# Phases 2 & 3: Ops, Computation Graph, and Autodiff

## Overview

Phase 2 implements the core tensor operations. Phase 3 builds automatic
differentiation on top of them — twice, in two different styles.

These phases build directly on the `Tensor` and `Storage` from Phase 1.

-----

## File structure

```
ml_systems/
├── project1_tensor/           # Phase 1 (unchanged)
│   ├── include/
│   │   ├── storage.hpp
│   │   └── tensor.hpp
│   └── src/
│       ├── storage.cpp
│       └── tensor.cpp
├── project2_ops/
│   ├── include/
│   │   └── ops.hpp            # given
│   └── src/
│       └── ops.cpp            # YOU IMPLEMENT
├── project3_autodiff/
│   ├── include/
│   │   ├── node.hpp           # given
│   │   ├── graph.hpp          # given (Design A)
│   │   └── autograd.hpp       # given (Design B)
│   └── src/
│       ├── graph.cpp          # YOU IMPLEMENT (Design A)
│       └── autograd.cpp       # YOU IMPLEMENT (Design B)
└── tests/
    └── test_ops_autodiff.cpp  # given
```

-----

## Phase 2: Ops

### What you’re building

Free functions in the `ops` namespace that compute results on Tensors.
Every op comes in two forms:

```cpp
void ops::add(const Tensor& a, const Tensor& b, Tensor& out);  // core
Tensor ops::add(const Tensor& a, const Tensor& b);             // convenience
```

The convenience form allocates an output of the correct shape and calls
through to the core form.

### Why preallocated output?

In real ML systems, memory allocation is expensive. Separating “what shape
does the output have” from “who allocates it” lets the caller control memory.
This matters later when op fusion (Phase 5) wants to eliminate intermediate
buffers entirely — the fused op writes directly into the final output,
skipping allocations that the unfused version would require.

### Broadcasting

Elementwise ops follow NumPy broadcasting rules:

- Align shapes from the right.
- A size-1 dimension expands to match the other tensor.
- Missing leading dimensions are treated as size-1.

```
(3, 4) + (4,)    -> (3, 4)
(3, 1) + (3, 4)  -> (3, 4)
(1,)   + (3, 4)  -> (3, 4)
```

Shapes are incompatible if a non-1 dimension doesn’t match.

Implement `ops::broadcast_shape` first — the elementwise ops call it.

### Matmul loop order

Implement matmul with the i-k-j loop order:

```cpp
for i in M:
  for k in K:       // <- K in the middle
    for j in N:
      out[i][j] += a[i][k] * b[k][j]
```

The naive i-j-k order causes a cache miss on `b[k][j]` for every j (striding
through rows of b). The i-k-j order keeps `b[k][*]` (a full row) in cache
across the inner loop. Benchmark both on a large matrix to see the difference.

-----

## Phase 3: Autodiff

### Background: reverse-mode AD

Given a scalar loss = f(x₁, x₂, …, xₙ), reverse-mode AD computes
∂loss/∂xᵢ for all inputs in a single backward pass. This is efficient
when there are many inputs and one output — exactly the ML case.

The algorithm:

1. Forward pass: compute the result, record the computation graph.
1. Topological sort the graph from the loss node.
1. Seed: set loss.grad = 1.
1. Walk in reverse topological order. For each node, compute the
   vector-Jacobian product (VJP) and accumulate into input gradients.

The VJP answers: given ∂loss/∂output, what is ∂loss/∂input?
It applies the chain rule at each op.

### VJP rules

Derive these yourself before implementing. The key ones:

|Op          |VJP                                                |
|------------|---------------------------------------------------|
|add(a,b)    |grad_a = grad, grad_b = grad                       |
|sub(a,b)    |grad_a = grad, grad_b = -grad                      |
|mul(a,b)    |grad_a = grad * b, grad_b = grad * a               |
|relu(a)     |grad_a = grad * (a > 0)                            |
|exp(a)      |grad_a = grad * exp(a)                             |
|log(a)      |grad_a = grad / a                                  |
|sum(a,axis) |grad_a = broadcast(grad) along axis back to a.shape|
|mean(a,axis)|grad_a = broadcast(grad) / N along axis            |
|matmul(A,B) |grad_A = grad @ Bᵀ, grad_B = Aᵀ @ grad             |

For reductions, “broadcast back” means: insert the reduced dimension back
(size 1) and expand it to match the original shape.

### Design A — Explicit graph

A `Graph` object owns all nodes. You call `graph.add(a, b)` to register ops.
`graph.backward(loss)` runs the backward pass.

The graph stores nodes in insertion order. Backward does a DFS topo-sort
from the loss node and walks in reverse.

Implement topological sort with DFS post-order:

```
topo_sort(node):
  if node in visited: return
  mark visited
  for each input of node:
    topo_sort(input)
  append node to order
```

Backward walks `order` in reverse.

### Design B — Implicit graph (PyTorch-style)

No `Graph` object. Each `AGTensor` holds a `grad_fn` pointer to the `Node`
that produced it. The graph lives in these pointers.

`AGTensor::backward()` does the same topo-sort, but traverses `grad_fn`
pointers instead of a node list.

Key rule: if neither input `requires_grad`, don’t build a node — return a
plain `AGTensor` with `grad_fn = nullptr`. This avoids overhead for tensors
that don’t need gradients.

### Gradient checker

`check_grad` is provided in the test file. It approximates the gradient
numerically using centered finite differences:

```
∂f/∂x[i] ≈ (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)
```

and compares against your analytic gradient. A max error below 1e-3
indicates a correct implementation (float32 limits precision).

Use this to debug VJP implementations — if a test fails, call `check_grad`
on that op in isolation to pinpoint which VJP is wrong.

-----

## Written questions

**Q1.** Derive the VJP for `matmul`. If C = A @ B and ∂loss/∂C = G, show
that ∂loss/∂A = G @ Bᵀ and ∂loss/∂B = Aᵀ @ G using the chain rule and
the definition of matrix derivatives.

**Q2.** In Design A, `backward_fn` is stored as a `std::function<void(const Tensor&)>`.
What does the lambda close over, and why is this necessary? What would go wrong
if you stored only the `OpType` and recomputed inputs from the graph?

**Q3.** In Design B, why does the `AGTensor` need to store both `data` (a
`Tensor`) and `grad_fn` (a `shared_ptr<Node>`)? Why not store the `Node`
only and retrieve `data` from `node->output`?

**Q4.** What is the asymptotic memory cost of the computation graph relative
to the forward computation? For a network with L layers each producing a
tensor of size N, how much memory does the graph consume during backward?

**Q5.** The current design does not handle gradient accumulation (a tensor
used more than once in the graph). Sketch how you would extend Design B to
support it. What data structure change is needed, and where in the backward
pass does accumulation happen?

-----

## Resources

- Andrej Karpathy’s micrograd — a minimal Python autograd engine that
  implements the same ideas: https://github.com/karpathy/micrograd
- The matrix cookbook (VJP derivations for matrix ops, section 2):
  https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
- PyTorch autograd internals (for Design B reference):
  https://pytorch.org/docs/stable/notes/autograd.html
