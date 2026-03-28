#pragma once

#include <cstddef> // size_t
#include <memory>  // std::unique_ptr
#include <stdexcept>

// =============================================================================
// Storage
// =============================================================================
// Storage owns a flat array of floats on the heap. Nothing more.
//
// The key design principle: separate *ownership* from *interpretation*.
//   - Storage says: "I own this block of N floats."
//   - Tensor (see tensor.hpp) says: "Here is how to interpret a Storage
//     as an N-dimensional array."
//
// Why separate them?
//   Multiple Tensor objects can share one Storage — e.g. a transposed view
//   and the original tensor both point at the same data. If Tensor owned the
//   buffer directly, you'd need to copy on every view operation. With Storage
//   as a separate ref-counted object, views are free.
//
// This mirrors PyTorch's design: torch.Storage / torch.Tensor, and NumPy's
// ndarray, which separates the data buffer from the array metadata.

class Storage {
  public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    // Allocate a zero-initialized buffer of `size` floats.
    //
    // We default-initialize to zero so that freshly created tensors have
    // defined values. In a production system you might skip this for
    // performance, but correctness first.
    explicit Storage(size_t size);

    // -------------------------------------------------------------------------
    // Rule of Five
    // -------------------------------------------------------------------------
    // When a class manages a resource (here: heap memory), you must explicitly
    // decide what happens when the object is copied, moved, or destroyed.
    // This is the "Rule of Five": if you define any one of the following, you
    // should define all five.
    //
    // The five special member functions are:
    //   1. Destructor            — what happens when the object goes out of
    //   scope
    //   2. Copy constructor      — what happens with:  Storage b = a;
    //   3. Copy assignment       — what happens with:  b = a;  (both already
    //   exist)
    //   4. Move constructor      — what happens with:  Storage b =
    //   std::move(a);
    //   5. Move assignment       — what happens with:  b = std::move(a);
    //
    // Copy  = make an independent duplicate (two separate buffers, same
    // values). Move  = transfer ownership (no allocation, the source is left
    // empty).
    //
    // Why does move matter?
    //   Returning a large Storage from a function would copy gigabytes without
    //   move semantics. With move, the return just transfers the pointer —
    //   O(1).
    //
    // YOUR TASK: implement all five in storage.cpp.

    ~Storage();

    Storage(const Storage &other);            // copy constructor
    Storage &operator=(const Storage &other); // copy assignment

    Storage(Storage &&other) noexcept;            // move constructor
    Storage &operator=(Storage &&other) noexcept; // move assignment

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    // Total number of floats in the buffer.
    size_t size() const;

    // Raw pointer to the start of the buffer.
    //
    // Two overloads: one for non-const Storage (returns a mutable pointer so
    // callers can write), one for const Storage (returns a read-only pointer).
    //
    // Why const-overload?
    //   If you have `const Storage& s`, you can still call s.data() — but you
    //   should not be able to modify the contents through it. The const
    //   overload enforces that at compile time.
    float *data();
    const float *data() const;

  private:
    // A raw pointer rather than std::vector or std::array because:
    //   1. We want explicit control over allocation (relevant for later phases
    //      where you might swap in aligned_alloc or a custom allocator).
    //   2. It makes implementing the Rule of Five more instructive — you have
    //      to think about ownership explicitly.
    float *data_;
    size_t size_;
};
