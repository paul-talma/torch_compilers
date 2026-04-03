#pragma once

#include "tensor.hpp"

// =============================================================================
// ops.hpp — Core tensor operations
// =============================================================================
// This file declares the compute operations that work on Tensors. Phase 2 ops
// are “dumb”: they compute results but do not record anything for autodiff.
// That comes in Phase 3.
//
// —————————————————————————–
// Design: preallocated output
// —————————————————————————–
// Every op comes in two forms:
//
//   void add(const Tensor& a, const Tensor& b, Tensor& out);   // core
//   Tensor add(const Tensor& a, const Tensor& b);              // convenience
//
// The core form writes into a caller-provided output buffer. The convenience
// form allocates a fresh Tensor and calls through.
//
// Why expose the preallocated form?
//   In real ML systems, memory allocation is expensive and carefully managed.
//   Frameworks like PyTorch and TVM separate “where does the output live” from
//   “what computation produces it”. The preallocated form lets the caller
//   control allocation — important for op fusion (Phase 5) where you want to
//   avoid intermediate buffers entirely.
//
// Why also provide the convenience form?
//   Usability. For tests and prototyping you don’t want to manually allocate
//   every output. The convenience form is a thin wrapper.
//
// —————————————————————————–
// Design: free functions vs methods
// —————————————————————————–
// Ops are free functions (not Tensor methods) because:
//   1. An op takes multiple tensors as input — there’s no natural “self”.
//   2. Free functions are easier to swap, wrap, or intercept (useful for the
//      graph in Phase 3).
//   3. This matches how ATen (PyTorch’s op library) works internally.
//
// Ops are also exposed as methods on Tensor for ergonomics:
//   t.add(other)  vs  add(t, other)
// The method simply calls the free function.
//
// —————————————————————————–
// Broadcasting
// —————————————————————————–
// Elementwise ops support NumPy-style broadcasting on the last dimensions.
// Rules:
//   - Dimensions are aligned from the right.
//   - A dimension of size 1 is expanded to match the other tensor’s size.
//   - Missing leading dimensions are treated as size 1.
//
// Examples:
//   (3, 4) + (4,)    -> (3, 4)      broadcast along dim 0
//   (3, 1) + (3, 4)  -> (3, 4)      broadcast along dim 1
//   (5, 1, 4) + (3, 4) -> (5, 3, 4) broadcast + prepend
//
// Throws std::runtime_error if shapes are incompatible.

namespace ops {

// =============================================================================
// Shape utilities
// =============================================================================

// Compute the output shape for two tensors under broadcasting rules.
// Throws if shapes are not broadcast-compatible.
std::vector<size_t> broadcast_shape(const std::vector<size_t> &a,
                                    const std::vector<size_t> &b);

// =============================================================================
// Elementwise ops
// =============================================================================
// All elementwise ops support broadcasting.
// `out` must have the correct broadcast output shape.

// out[i] = a[i] + b[i]
void add(const Tensor &a, const Tensor &b, Tensor &out);
Tensor add(const Tensor &a, const Tensor &b);

// out[i] = a[i] - b[i]
void sub(const Tensor &a, const Tensor &b, Tensor &out);
Tensor sub(const Tensor &a, const Tensor &b);

// out[i] = a[i] * b[i]
void mul(const Tensor &a, const Tensor &b, Tensor &out);
Tensor mul(const Tensor &a, const Tensor &b);

// out[i] = a[i] / b[i]
// Throws if any b[i] == 0.
void div(const Tensor &a, const Tensor &b, Tensor &out);
Tensor div(const Tensor &a, const Tensor &b);

// out[i] = max(0, a[i])
// Unary — no broadcasting needed.
void relu(const Tensor &a, Tensor &out);
Tensor relu(const Tensor &a);

// out[i] = exp(a[i])
void exp(const Tensor &a, Tensor &out);
Tensor exp(const Tensor &a);

// out[i] = log(a[i])
// Throws if any a[i] <= 0.
void log(const Tensor &a, Tensor &out);
Tensor log(const Tensor &a);

// =============================================================================
// Reductions
// =============================================================================
// Reduce along a single axis. The output drops that dimension entirely.
//   sum({3,4,5}, axis=1) -> shape {3,5}
//
// `out` must already have the correct reduced shape.

// out[…] = sum of a[…] along `axis`
void sum(const Tensor &a, size_t axis, Tensor &out);
Tensor sum(const Tensor &a, size_t axis);

// out[…] = mean of a[…] along `axis`
void mean(const Tensor &a, size_t axis, Tensor &out);
Tensor mean(const Tensor &a, size_t axis);

// Reduce over ALL elements, returning a scalar (shape {1}).
Tensor sum_all(const Tensor &a);
Tensor mean_all(const Tensor &a);

// =============================================================================
// Matrix multiplication
// =============================================================================
// 2D only: a is (M, K), b is (K, N), out is (M, N).
// Throws if inner dimensions don’t match or inputs are not 2D.
//
// Implementation note: loop order matters for cache performance.
// The naive i-j-k order causes cache misses on b. The i-k-j order
// (iterate over the contraction dimension in the middle) keeps b’s
// row in cache. You should implement i-k-j and measure the difference.
void matmul(const Tensor &a, const Tensor &b, Tensor &out);
Tensor matmul(const Tensor &a, const Tensor &b);

// helpers
void generic_op(const Tensor &a, Tensor &out, float op(const float &)); // unary
void generic_op(const Tensor &a, // binary
                const Tensor &b,
                Tensor &out,
                float op(const float &, const float &));
void broadcast_to(const Tensor &src, Tensor &out);
} // namespace ops
