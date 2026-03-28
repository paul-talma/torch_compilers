# Phase 1: Tensor Abstraction

## Overview

You will implement a minimal tensor library in C++. The goal is not to produce
a polished ML framework — it is to deeply understand how multi-dimensional
arrays are represented in memory, and how frameworks like PyTorch and NumPy
achieve zero-copy operations like transpose and slice.

This is Phase 1 of a multi-phase project. The code you write here will be the
foundation for a computation graph, automatic differentiation, and an op fusion
pass in later phases. Write it carefully.

---

## What you're building

Two classes:

**`Storage`** — owns a flat array of floats on the heap. Nothing more. It knows
its size, and it gives you a pointer to its data. The class exists solely to
separate *ownership* from *interpretation*.

**`Tensor`** — a *view* into a Storage. It adds shape, strides, and an offset
that describe how to interpret the flat buffer as an N-dimensional array. A
Tensor does not own the data — the Storage does. Multiple Tensors can share
one Storage.

---

## Files

```
include/
    storage.hpp     — Storage class declaration (given; read carefully)
    tensor.hpp      — Tensor class declaration  (given; read carefully)
src/
    storage.cpp     — YOU IMPLEMENT THIS
    tensor.cpp      — YOU IMPLEMENT THIS
tests/
    test_tensor.cpp — test suite (given; do not modify)
CMakeLists.txt      — build config (given)
```

Create `src/storage.cpp` and `src/tensor.cpp`. Do not modify the headers or
the test file.

---

## Build & run

```bash
mkdir build && cd build
cmake ..
make
./tests
```

All tests should pass before you proceed to the written questions.

---

## Strides

The most important concept in this assignment. Read this carefully.

A tensor with shape `(d0, d1, d2)` stored in a flat buffer needs a rule for
mapping a logical index `(i, j, k)` to a position in the buffer.

The rule is:

    flat_index = offset + i * strides[0] + j * strides[1] + k * strides[2]

Strides are stored explicitly — one per dimension. For a freshly allocated
tensor in row-major (C) order:

    strides[ndim-1] = 1
    strides[i]      = strides[i+1] * shape[i+1]

Example: shape `(3, 4, 5)` → strides `{20, 5, 1}`.

**Why store strides explicitly rather than recomputing from shape?**

Because strides decouple the *logical* layout from the *physical* layout. After
a transpose, the logical shape changes but the buffer does not. Storing strides
directly lets you express any layout — row-major, column-major, transposed,
sliced — with the same data structure.

---

## Zero-copy views

`transpose`, `slice`, and `reshape` all return a new Tensor object that shares
the same underlying Storage. No data is copied.

- **transpose(dim0, dim1):** swap `shape[dim0]` ↔ `shape[dim1]` and
  `strides[dim0]` ↔ `strides[dim1]`. The buffer is untouched.

- **slice(dim, start, end):** set `offset += start * strides[dim]` and
  `shape[dim] = end - start`. Strides are unchanged.

- **reshape(new_shape):** only valid when the tensor is contiguous (strides
  are row-major). Recompute strides for the new shape. The buffer is untouched.

---

## Copy vs. move semantics

A core C++ concept you must implement correctly for both classes.

**Copy** means: make an independent duplicate.
- For `Storage`: allocate a new buffer, copy all values.
- For `Tensor`: copy the `shared_ptr` (shared ownership of same Storage), copy
  shape/strides/offset vectors.

**Move** means: transfer ownership without allocating.
- For `Storage`: transfer the raw pointer; set the source's pointer to nullptr
  and size to 0.
- For `Tensor`: move the `shared_ptr` and vectors; the source is left valid but
  empty.

You must implement the *Rule of Five* for both classes: destructor, copy
constructor, copy assignment, move constructor, move assignment.

Note that `Tensor` copy is *shallow* — two Tensors after a copy share the same
Storage. This is intentional (it is how views work). A deep copy (independent
buffer) will be added as a `clone()` method in a later phase.

---

## Implementation notes

**`is_contiguous()`:** a tensor is contiguous if its strides match what
row-major would produce. Check: `strides[ndim-1] == 1` and
`strides[i] == strides[i+1] * shape[i+1]` for all valid `i`.

**`data_ptr()`:** should return `storage_->data() + offset_`, not just
`storage_->data()`. The offset is nonzero after slicing.

**`at()` bounds checking:** throw `std::out_of_range` if any index exceeds its
dimension. Include a helpful message.

**`print()`:** iterate in logical index order using nested loops (or a
recursive helper). This is not performance-critical — use `at()` internally.

---

## Written questions

Answer these after all tests pass. Short answers (a few sentences each) are
sufficient.

**Q1.** After calling `t.transpose(0, 1)`, is the resulting tensor contiguous?
Why or why not? What would you have to do to get a contiguous tensor with the
transposed values?

**Q2.** `Storage` uses a raw pointer (`float*`) rather than `std::vector<float>`
or `std::unique_ptr<float[]>`. What is the tradeoff? What additional
responsibility does this place on you as the implementer?

**Q3.** `Tensor` copy is shallow (shared Storage), but `Storage` copy is deep
(new buffer). Explain why this asymmetry is intentional and correct.

**Q4.** Suppose you call `slice(0, 2, 5)` on a tensor with shape `(8, 4)`.
What are the shape, strides, and offset of the resulting tensor? Does the result
share storage with the original? Is it contiguous?

**Q5.** Why does `reshape` require `is_contiguous()`? Construct a concrete
example of a non-contiguous tensor where the requested reshape would produce
incorrect results if the check were absent.

---

## Resources

- NumPy's internal memory model (the definitive reference for this abstraction):
  https://numpy.org/doc/stable/reference/internals.html

- PyTorch's `as_strided` — experiment with strides interactively:
  https://pytorch.org/docs/stable/generated/torch.as_strided.html

- C++ Rule of Five:
  https://en.cppreference.com/w/cpp/language/rule_of_three

- `std::shared_ptr`:
  https://en.cppreference.com/w/cpp/memory/shared_ptr
