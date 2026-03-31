#pragma once

#include "node.hpp"
#include "tensor.hpp"

// =============================================================================
// autograd.hpp — Design B: Implicit graph (PyTorch-style)
// =============================================================================
// In this design, the graph is implicit: each Tensor optionally holds a
// shared_ptr to the Node that produced it (its grad_fn). The graph exists
// in the pointers between nodes — there is no separate Graph object.
//
// This is how PyTorch’s autograd works. torch.Tensor has a .grad_fn
// attribute that points to the Function (our Node) that created it, and
// .grad that accumulates gradients during backward.
//
// The tradeoff vs Design A:
//   Pro: ergonomic — ops look like normal function calls, no graph object
//   Con: the graph structure is invisible; harder to inspect or reason about
//
// We introduce a new class, AGTensor (autograd tensor), that wraps Tensor
// and adds requires_grad, grad, and grad_fn fields. This keeps the Phase 1
// Tensor clean and avoids coupling it to autodiff.
//
// Usage:
//   AGTensor x(Tensor({3,4}), /*requires_grad=*/true);
//   AGTensor y(Tensor({3,4}), true);
//   AGTensor z = ag::add(x, y);    // builds graph implicitly
//   AGTensor loss = ag::mean_all(z);
//   loss.backward();               // populates .grad on x, y, z, loss
//   x.grad;                        // d(loss)/d(x)

struct AGTensor {
    // The underlying data.
    Tensor data;

    // If true, gradients flow through this tensor during backward.
    bool requires_grad;

    // Gradient of loss w.r.t. this tensor. Populated during backward.
    // Only meaningful if requires_grad == true.
    std::optional<Tensor> grad;

    // The node that produced this tensor. nullptr for leaf tensors.
    // This is the backward edge in the implicit graph.
    std::shared_ptr<Node> grad_fn;

    // Convenience: is this a leaf (no grad_fn)?
    bool is_leaf() const { return grad_fn == nullptr; }

    // Trigger backward pass from this tensor (must be scalar).
    // Walks the implicit graph via grad_fn pointers.
    void backward();

    // Constructors
    explicit AGTensor(Tensor data, bool requires_grad = false);
};

// =============================================================================
// ag namespace — autograd ops
// =============================================================================
// These mirror the ops in ops.hpp but operate on AGTensors, building the
// implicit graph as a side effect when requires_grad is true.
//
// If neither input requires grad, the result is a plain tensor with no
// grad_fn — no graph is built, no overhead.

namespace ag {

AGTensor add(const AGTensor &a, const AGTensor &b);
AGTensor sub(const AGTensor &a, const AGTensor &b);
AGTensor mul(const AGTensor &a, const AGTensor &b);
AGTensor relu(const AGTensor &a);
AGTensor exp(const AGTensor &a);
AGTensor log(const AGTensor &a);
AGTensor sum(const AGTensor &a, size_t axis);
AGTensor mean(const AGTensor &a, size_t axis);
AGTensor mean_all(const AGTensor &a);
AGTensor matmul(const AGTensor &a, const AGTensor &b);

} // namespace ag
