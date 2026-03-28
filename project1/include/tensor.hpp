#pragma once

#include "storage.hpp"
#include <vector>
#include <string>
#include <initializer_list>

// =============================================================================
// Tensor
// =============================================================================
// A Tensor is a *view* into a Storage: it adds shape, strides, and an offset
// that describe how to interpret the flat buffer as an N-dimensional array.
//
// Vocabulary:
//   shape   — the size of each dimension, e.g. {3, 4, 5} for a 3D tensor
//   strides — how many elements to skip in the flat buffer when you increment
//             the index along each dimension by 1 (details below)
//   offset  — where in the buffer this view starts (nonzero after slicing)
//
// -----------------------------------------------------------------------------
// Strides in depth
// -----------------------------------------------------------------------------
// Given shape {3, 4, 5}, the default row-major strides are {20, 5, 1}.
//
//   element at logical index (i, j, k)  =  buffer[offset + i*20 + j*5 + k*1]
//
// General formula for row-major strides:
//   strides[n-1] = 1
//   strides[i]   = strides[i+1] * shape[i+1]
//
// Why strides enable zero-copy views:
//   Transpose dims 0 and 1:  swap strides -> {5, 20, 1}, swap shape -> {4,3,5}
//   The buffer is unchanged. You're just reinterpreting how to walk it.
//
//   Slice dim 1 from index 1 to 3:
//     shape[1] becomes 2, offset += 1 * strides[1]. Buffer unchanged.
//
// This is identical to how NumPy and PyTorch represent tensors internally.
// -----------------------------------------------------------------------------

class Tensor {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    // Allocate a new tensor with the given shape.
    // Creates a fresh Storage and computes row-major strides automatically.
    // The underlying buffer is zero-initialized.
    //
    // Example:
    //   Tensor t({3, 4, 5});   // shape (3,4,5), 60 floats, row-major strides
    explicit Tensor(std::vector<size_t> shape);

    // View constructor: wrap an existing Storage with explicit metadata.
    // Does NOT copy the buffer — this is how transpose/slice return views.
    //
    // This constructor is mostly called internally by view-producing methods,
    // but it's public so you can construct exotic layouts in tests.
    Tensor(std::shared_ptr<Storage> storage,
           std::vector<size_t>      shape,
           std::vector<size_t>      strides,
           size_t                   offset = 0);

    // -------------------------------------------------------------------------
    // Rule of Five
    // -------------------------------------------------------------------------
    // Think carefully about what "copy" means for a Tensor:
    //
    //   SHALLOW copy — two Tensors share the same Storage (same buffer).
    //                  This is what you want for views (cheap, no allocation).
    //   DEEP copy    — allocate a new Storage and copy all values.
    //                  This is what you want when you need an independent tensor.
    //
    // For this assignment: implement SHALLOW copy/move for Tensor.
    // (Deep copy will be a named method `clone()` added in a later phase.)
    //
    // Since Tensor holds a std::shared_ptr<Storage>, a shallow copy is simply
    // copying the shared_ptr — the ref count goes up, no buffer is duplicated.
    // Move transfers the shared_ptr and leaves the source in a valid empty state.
    //
    // YOUR TASK: implement all five in tensor.cpp.

    ~Tensor();

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // -------------------------------------------------------------------------
    // Metadata
    // -------------------------------------------------------------------------

    // Returns the size of each dimension, e.g. {3, 4, 5}.
    //
    // Returned by const-ref to avoid copying the vector. The method is const
    // because inspecting metadata does not modify the tensor.
    const std::vector<size_t>& shape() const;

    // Returns the stride for each dimension (in units of elements, not bytes).
    const std::vector<size_t>& strides() const;

    // Number of dimensions.
    size_t ndim() const;

    // Total number of elements (product of all shape dimensions).
    size_t numel() const;

    // Returns the offset into the Storage buffer where this view starts.
    size_t offset() const;

    // A tensor is contiguous if its elements are laid out sequentially in
    // memory with no gaps — i.e. strides match what row-major would produce.
    //
    // Formally: strides[i] == strides[i+1] * shape[i+1], strides[ndim-1] == 1.
    //
    // Why does this matter?
    //   Some operations (reshape, fast loops) require contiguous layout.
    //   After a transpose, is_contiguous() returns false.
    bool is_contiguous() const;

    // -------------------------------------------------------------------------
    // Element access
    // -------------------------------------------------------------------------

    // Read/write element at logical index `idx`.
    //
    // Computes: buffer[offset + sum_i(idx[i] * strides[i])]
    //
    // Two overloads for the same reason as Storage::data():
    //   - non-const returns float& so you can write: t.at({0,1}) = 3.14f;
    //   - const returns float  so you can read from a const Tensor.
    //
    // Throws std::out_of_range if any index exceeds its dimension.
    float  at(std::initializer_list<size_t> idx) const;
    float& at(std::initializer_list<size_t> idx);

    // -------------------------------------------------------------------------
    // Zero-copy views
    // -------------------------------------------------------------------------
    // Each of these returns a NEW Tensor object pointing at the SAME Storage.
    // No data is copied. The returned Tensor has different shape/strides/offset.
    //
    // All are marked const because they don't modify *this — they produce a
    // new view. (The underlying buffer is shared, but the Tensor object itself
    // is not mutated.)

    // Return a view with a different shape.
    // Precondition: is_contiguous() — reshape is undefined for strided tensors
    //               because the new shape's strides would be inconsistent.
    // Precondition: product of new_shape == numel().
    // Throws std::runtime_error if either precondition is violated.
    Tensor reshape(std::vector<size_t> new_shape) const;

    // Return a view with dims dim0 and dim1 swapped.
    // Shape and strides at those two positions are exchanged; everything else
    // is unchanged.
    // Throws std::out_of_range if dim0 or dim1 >= ndim().
    Tensor transpose(size_t dim0, size_t dim1) const;

    // Return a view into a slice of dimension `dim` from index `start`
    // (inclusive) to `end` (exclusive).
    //
    // Effect on metadata:
    //   offset  += start * strides[dim]
    //   shape[dim] = end - start
    //   strides unchanged
    //
    // Throws std::out_of_range if dim >= ndim() or end > shape[dim] or
    // start >= end.
    Tensor slice(size_t dim, size_t start, size_t end) const;

    // -------------------------------------------------------------------------
    // Raw data access
    // -------------------------------------------------------------------------

    // Returns a raw pointer to buffer[offset] — the start of *this* view.
    // Used by operator implementations (phase 2) for fast loops.
    //
    // Only valid to dereference up to numel() elements when the tensor is
    // contiguous. For non-contiguous tensors, use at() or index arithmetic.
    float*       data_ptr();
    const float* data_ptr() const;

    // Access the underlying Storage (needed for constructing views from ops).
    std::shared_ptr<Storage>       storage();
    std::shared_ptr<const Storage> storage() const;

    // -------------------------------------------------------------------------
    // Debug
    // -------------------------------------------------------------------------

    // Print the tensor contents in a nested bracket format, e.g.:
    //   [[1.0, 2.0],
    //    [3.0, 4.0]]
    // Uses logical index order (respects strides), so transposed tensors print
    // correctly.
    void print() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t>      shape_;
    std::vector<size_t>      strides_;
    size_t                   offset_;

    // Helper: compute row-major strides from a shape.
    // strides[n-1] = 1, strides[i] = strides[i+1] * shape[i+1]
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

    // Helper: compute flat buffer index from a logical multi-index.
    size_t flat_index(const std::vector<size_t>& idx) const;
};
