#pragma once

#include “tensor.hpp”
#include “ops.hpp”
#include <functional>
#include <memory>
#include <optional>
#include <vector>

// =============================================================================
// node.hpp — Computation graph node
// =============================================================================
//
// —————————————————————————–
// Two graph designs
// —————————————————————————–
// This phase implements autodiff twice, in increasing realism:
//
// DESIGN A — Explicit graph (instructive)
//   A separate `Graph` object owns all nodes. You explicitly register ops
//   with the graph. Backward is a method on Graph. The structure is fully
//   visible and inspectable.
//
// DESIGN B — Implicit graph (realistic, PyTorch-style)
//   Each Tensor optionally holds a pointer to the Node that produced it
//   (its “grad_fn”). The graph is implicit in these pointers. Backward
//   is triggered by calling .backward() on a scalar loss Tensor.
//
// You will implement Design A first, then Design B.
// The Node struct below is shared by both designs.
//
// —————————————————————————–
// How reverse-mode autodiff works
// —————————————————————————–
// Given a computation loss = f(x), reverse-mode AD computes d(loss)/d(x)
// for every intermediate and leaf tensor in one backward pass.
//
// Algorithm:
//   1. Build the forward graph (happens during forward pass).
//   2. Topological sort from the loss node.
//   3. Seed: set loss.grad = ones (d(loss)/d(loss) = 1).
//   4. Walk nodes in reverse topological order. For each node:
//        grad_input = vjp(grad_output)
//      where vjp is the vector-Jacobian product for that op.
//
// The VJP (vector-Jacobian product) for each op:
//   add:     grad_a = grad,  grad_b = grad
//   sub:     grad_a = grad,  grad_b = -grad
//   mul:     grad_a = grad * b,  grad_b = grad * a
//   relu:    grad_a = grad * (a > 0)
//   sum:     grad_a = broadcast(grad) back to a’s shape
//   mean:    grad_a = broadcast(grad) / N
//   matmul:  grad_a = grad @ b.T,  grad_b = a.T @ grad
//
// You should derive these yourself before implementing (see written questions).

// —————————————————————————–
// OpType — what operation produced this node’s output
// —————————————————————————–
enum class OpType {
    Leaf, // input tensor, no inputs, gradient accumulates here
    Add,
    Sub,
    Mul,
    Div,
    ReLU,
    Exp,
    Log,
    Sum,
    Mean,
    MatMul,
};

// —————————————————————————–
// Node
// —————————————————————————–
// A node in the computation graph. Represents one op application.
//
// Each node stores:
//   - The op that produced it
//   - Pointers to input nodes (the node’s “parents” in the graph)
//   - The output Tensor produced by the op
//   - The gradient of the loss w.r.t. this node’s output (populated during
//     backward)
//   - A backward function: given grad_output, compute grad_inputs and
//     write them into the input nodes’ grad fields
//
// Why store the backward function as a std::function?
//   Each op’s VJP needs to close over the *values* of the forward inputs
//   (e.g. matmul’s VJP needs a and b to compute grad_a = grad @ b.T).
//   A lambda captures these naturally. std::function erases the lambda type
//   so all nodes have the same type regardless of op.
struct Node {
    OpType op;

    ```
        // Input nodes. Empty for Leaf nodes.
        std::vector<std::shared_ptr<Node>>
            inputs;

    // The tensor this node produced during the forward pass.
    // Needed by VJPs that depend on forward values (e.g. relu, mul).
    Tensor output;

    // Gradient of loss w.r.t. output. Populated during backward.
    // std::optional because it starts unset and is filled in during backward.
    std::optional<Tensor> grad;

    // The VJP: given grad (d_loss/d_output), write gradients into
    // inputs[i]->grad for each input.
    // Empty for Leaf nodes (no inputs to propagate to).
    std::function<void(const Tensor &grad)> backward_fn;

    // axis stored for sum/mean nodes (needed for VJP shape reconstruction)
    std::optional<size_t> reduce_axis;
    ```
};
