#pragma once

#include “node.hpp”

// =============================================================================
// graph.hpp — Design A: Explicit computation graph
// =============================================================================
// A Graph owns all nodes. You register ops by calling graph.add(),
// graph.matmul(), etc. The graph tracks the full computation and can
// run backward from any scalar node.
//
// This design makes the structure maximally visible: you can inspect
// graph.nodes(), print the DAG, and reason about the backward pass
// before it runs. It is less ergonomic than Design B (you have to pass
// the graph around) but more transparent.
//
// Usage:
//   Graph g;
//   auto x = g.leaf(Tensor({3, 4}));   // register input
//   auto y = g.leaf(Tensor({3, 4}));
//   auto z = g.add(x, y);              // register op, returns node
//   auto loss = g.mean_all(z);
//   g.backward(loss);                  // populates .grad on all nodes
//   loss->grad;                        // d(loss)/d(loss) = 1
//   x->grad;                           // d(loss)/d(x)

class Graph {
  public:
    // ———————————————————————––
    // Leaf registration
    // ———————————————————————––
    // Register a tensor as a leaf (input). Returns a Node whose .output is
    // the tensor and whose .grad will be populated during backward.
    //
    // requires_grad: if false, gradients are not propagated through this node.
    // This mirrors PyTorch’s tensor.requires_grad flag.
    std::shared_ptr<Node> leaf(Tensor t, bool requires_grad = true);

    ```
        // -------------------------------------------------------------------------
        // Op registration
        // -------------------------------------------------------------------------
        // Each method runs the op, creates a Node, registers it, and returns
        // it. The returned Node's .output holds the result tensor.

        std::shared_ptr<Node> add(std::shared_ptr<Node> a,
                                  std::shared_ptr<Node> b);
    std::shared_ptr<Node> sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    std::shared_ptr<Node> div(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
    std::shared_ptr<Node> relu(std::shared_ptr<Node> a);
    std::shared_ptr<Node> exp(std::shared_ptr<Node> a);
    std::shared_ptr<Node> log(std::shared_ptr<Node> a);
    std::shared_ptr<Node> sum(std::shared_ptr<Node> a, size_t axis);
    std::shared_ptr<Node> mean(std::shared_ptr<Node> a, size_t axis);
    std::shared_ptr<Node> mean_all(std::shared_ptr<Node> a);
    std::shared_ptr<Node> matmul(std::shared_ptr<Node> a,
                                 std::shared_ptr<Node> b);

    // -------------------------------------------------------------------------
    // Backward
    // -------------------------------------------------------------------------
    // Compute gradients for all nodes reachable from `loss`.
    // `loss` must be a scalar (numel() == 1).
    // After this call, every node's .grad is populated.
    //
    // Algorithm:
    //   1. Topological sort (DFS post-order from loss)
    //   2. Seed loss->grad = ones({1})
    //   3. Walk in reverse order, call backward_fn on each node
    void backward(std::shared_ptr<Node> loss);

    // All registered nodes, in insertion order.
    const std::vector<std::shared_ptr<Node>> &nodes() const;
  ```

      private : std::vector<std::shared_ptr<Node>>
                    nodes_;

    ```
        // DFS post-order traversal from `node`, appending to `order`.
        // `visited` prevents duplicate visits.
        void topo_sort(std::shared_ptr<Node> node,
                       std::vector<std::shared_ptr<Node>> &order,
                       std::unordered_set<Node *> &visited);

    // Helper: create and register a node.
    std::shared_ptr<Node>
    make_node(OpType op,
              std::vector<std::shared_ptr<Node>> inputs,
              Tensor output,
              std::function<void(const Tensor &)> backward_fn,
              std::optional<size_t> reduce_axis = std::nullopt);
    ```
};
