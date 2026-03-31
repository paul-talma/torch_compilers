#include “tensor.hpp”
#include “ops.hpp”
#include “graph.hpp”
#include “autograd.hpp”
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>

// =============================================================================
// Test harness (same as Phase 1)
// =============================================================================

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)
static void name();
struct name##_runner {
    name##_runner() {
        tests_run++;
        try {
            name();
            tests_passed++;
            std::cout << “ PASS  “ << #name << “\n”;
        } catch (const std::exception &e) {
            std::cout << “ FAIL  “ << #name << “\n         “ << e.what() << “\n”;
        }
    }
} name##_instance;
static void name()

#define ASSERT(cond)
    do {
    if (!(cond))
        throw std::runtime_error(“Assertion failed : “ #cond                
“ at line “ + std::to_string(**LINE **));
}
while (0)

#define ASSERT_EQ(a, b)
    do {
        auto _a = (a);
        auto _b = (b);
        if (_a != _b) {
            std::ostringstream _oss;
            _oss << “Expected equal
                : “ #a “ == “ #b << “ (” << _a << “ vs “ << _b << “)”
                                 << “ at line “ << **LINE * *;
            throw std::runtime_error(_oss.str());
        }
    } while (0)

#define ASSERT_NEAR(a, b, tol)
        do {
        auto _a = (a);
        auto _b = (b);
        if (std::fabs(_a - _b) > (tol)) {
            std::ostringstream _oss;
            _oss << “Expected near : “ #a “ ~ “ #b
                 << “ (” << _a << “ vs “ << _b << “)”
                 << “ at line “
                 << **LINE * *;
            throw std::runtime_error(_oss.str());
        }
    }
while (0)

#define ASSERT_THROWS(expr)
    do {
        bool threw = false;
        try {
            (expr);
        } catch (…) {
            threw = true;
        }
        if (!threw)
            throw std::runtime_error(                                 
“Expected exception from : “ #expr                     
“ at line “ + std::to_string(**LINE **));
    } while (0)

        // =============================================================================
        // Finite difference gradient checker
        // =============================================================================
        // For a scalar-valued function f(x), the numerical gradient at x[i] is:
        //
        //   df/dx[i] ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
        //
        // where e_i is the unit vector in direction i. This is the centered
        // difference approximation. We use it to verify analytic gradients from
        // autodiff.
        //
        // Returns the maximum absolute difference between analytic and numeric
        // grads. A value below 1e-4 generally indicates a correct
        // implementation.
        //
        // `f` takes a Tensor and returns a scalar Tensor (shape {1}).
        // `x` is the point at which to check gradients.
        // `analytic_grad` is the gradient computed by your backward pass.

        static float check_grad(std::function<Tensor(const Tensor &)> f,
                                const Tensor &x,
                                const Tensor &analytic_grad,
                                float eps = 1e-3f) {
        float max_err = 0.0f;
        Tensor x_pos = x; // will be mutated element by element
        Tensor x_neg = x;

        ``` for (size_t i = 0; i < x.numel(); ++i) {
            // Perturb element i
            float orig = x.data_ptr()[i];
            x_pos.data_ptr()[i] = orig + eps;
            x_neg.data_ptr()[i] = orig - eps;

            float f_pos = f(x_pos).data_ptr()[0];
            float f_neg = f(x_neg).data_ptr()[0];
            float numeric = (f_pos - f_neg) / (2.0f * eps);

            float analytic = analytic_grad.data_ptr()[i];
            float err = std::fabs(numeric - analytic);
            if (err > max_err)
                max_err = err;

            // Restore
            x_pos.data_ptr()[i] = orig;
            x_neg.data_ptr()[i] = orig;
        }
        return max_err;
        ```
    }

// =============================================================================
// Phase 2: Ops tests
// =============================================================================

// ––––––––––––––––––––
// add
// ––––––––––––––––––––

TEST(ops_add_basic) {
    Tensor a({2, 3}), b({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        a.data_ptr()[i] = static_cast<float>(i);
        b.data_ptr()[i] = 1.0f;
    }
    Tensor c = ops::add(a, b);
    for (size_t i = 0; i < 6; ++i)
        ASSERT_NEAR(c.data_ptr()[i], static_cast<float>(i) + 1.0f, 1e-6f);
}

TEST(ops_add_broadcast_row) {
    // (3,4) + (4,) -> (3,4): add a row vector to each row
    Tensor a({3, 4}), b({4});
    for (size_t i = 0; i < 12; ++i)
        a.data_ptr()[i] = 1.0f;
    for (size_t j = 0; j < 4; ++j)
        b.data_ptr()[j] = static_cast<float>(j);

    ``` Tensor c = ops::add(a, b);
    ASSERT_EQ(c.shape()[0], 3u);
    ASSERT_EQ(c.shape()[1], 4u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(c.at({i, j}), 1.0f + static_cast<float>(j), 1e-6f);
    ```
}

TEST(ops_add_broadcast_col) {
    // (3,1) + (3,4) -> (3,4)
    Tensor a({3, 1}), b({3, 4});
    for (size_t i = 0; i < 3; ++i)
        a.data_ptr()[i] = static_cast<float>(i);
    for (size_t i = 0; i < 12; ++i)
        b.data_ptr()[i] = 10.0f;

    ``` Tensor c = ops::add(a, b);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(c.at({i, j}), static_cast<float>(i) + 10.0f, 1e-6f);
    ```
}

TEST(ops_add_incompatible_shapes) {
    Tensor a({3, 4}), b({5});
    ASSERT_THROWS(ops::add(a, b));
}

TEST(ops_add_preallocated) {
    Tensor a({3}), b({3}), out({3});
    a.data_ptr()[0] = 1.0f;
    b.data_ptr()[0] = 2.0f;
    ops::add(a, b, out);
    ASSERT_NEAR(out.data_ptr()[0], 3.0f, 1e-6f);
}

// ––––––––––––––––––––
// mul
// ––––––––––––––––––––

TEST(ops_mul_basic) {
    Tensor a({4}), b({4});
    for (size_t i = 0; i < 4; ++i) {
        a.data_ptr()[i] = static_cast<float>(i + 1);
        b.data_ptr()[i] = 2.0f;
    }
    Tensor c = ops::mul(a, b);
    for (size_t i = 0; i < 4; ++i)
        ASSERT_NEAR(c.data_ptr()[i], static_cast<float>((i + 1) * 2), 1e-6f);
}

TEST(ops_mul_broadcast) {
    Tensor a({2, 3}), b({1});
    for (size_t i = 0; i < 6; ++i)
        a.data_ptr()[i] = static_cast<float>(i);
    b.data_ptr()[0] = 3.0f;
    Tensor c = ops::mul(a, b);
    for (size_t i = 0; i < 6; ++i)
        ASSERT_NEAR(c.data_ptr()[i], static_cast<float>(i) * 3.0f, 1e-6f);
}

// ––––––––––––––––––––
// relu
// ––––––––––––––––––––

TEST(ops_relu_basic) {
    Tensor a({5});
    float vals[] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    for (size_t i = 0; i < 5; ++i)
        a.data_ptr()[i] = vals[i];

    ``` Tensor b = ops::relu(a);
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f};
    for (size_t i = 0; i < 5; ++i)
        ASSERT_NEAR(b.data_ptr()[i], expected[i], 1e-6f);
    ```
}

// ––––––––––––––––––––
// exp / log
// ––––––––––––––––––––

TEST(ops_exp_basic) {
    Tensor a({3});
    a.data_ptr()[0] = 0.0f;
    a.data_ptr()[1] = 1.0f;
    a.data_ptr()[2] = 2.0f;
    Tensor b = ops::exp(a);
    ASSERT_NEAR(b.data_ptr()[0], 1.0f, 1e-5f);
    ASSERT_NEAR(b.data_ptr()[1], std::exp(1.0f), 1e-5f);
    ASSERT_NEAR(b.data_ptr()[2], std::exp(2.0f), 1e-5f);
}

TEST(ops_log_basic) {
    Tensor a({3});
    a.data_ptr()[0] = 1.0f;
    a.data_ptr()[1] = std::exp(1.0f);
    a.data_ptr()[2] = 4.0f;
    Tensor b = ops::log(a);
    ASSERT_NEAR(b.data_ptr()[0], 0.0f, 1e-5f);
    ASSERT_NEAR(b.data_ptr()[1], 1.0f, 1e-5f);
    ASSERT_NEAR(b.data_ptr()[2], std::log(4.0f), 1e-5f);
}

TEST(ops_log_nonpositive_throws) {
    Tensor a({3});
    a.data_ptr()[0] = 1.0f;
    a.data_ptr()[1] = 0.0f;
    a.data_ptr()[2] = 1.0f;
    ASSERT_THROWS(ops::log(a));
}

// ––––––––––––––––––––
// sum / mean
// ––––––––––––––––––––

TEST(ops_sum_axis0) {
    // (3,4) summed over axis 0 -> (4,)
    Tensor a({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            a.at({i, j}) = static_cast<float>(i * 4 + j);

    ``` Tensor s = ops::sum(a, 0);
    ASSERT_EQ(s.shape().size(), 1u);
    ASSERT_EQ(s.shape()[0], 4u);
    // column j: 0+j, 4+j, 8+j = 12+3j
    for (size_t j = 0; j < 4; ++j)
        ASSERT_NEAR(s.data_ptr()[j], 12.0f + 3.0f * static_cast<float>(j),
                    1e-5f);
    ```
}

TEST(ops_sum_axis1) {
    // (3,4) summed over axis 1 -> (3,)
    Tensor a({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            a.at({i, j}) = 1.0f;

    ``` Tensor s = ops::sum(a, 1);
    ASSERT_EQ(s.shape()[0], 3u);
    for (size_t i = 0; i < 3; ++i)
        ASSERT_NEAR(s.data_ptr()[i], 4.0f, 1e-5f);
    ```
}

TEST(ops_mean_axis0) {
    Tensor a({4, 2});
    for (size_t i = 0; i < 8; ++i)
        a.data_ptr()[i] = static_cast<float>(i);
    Tensor m = ops::mean(a, 0);
    ASSERT_EQ(m.shape()[0], 2u);
    // col 0: 0,2,4,6 -> mean 3.0; col 1: 1,3,5,7 -> mean 4.0
    ASSERT_NEAR(m.data_ptr()[0], 3.0f, 1e-5f);
    ASSERT_NEAR(m.data_ptr()[1], 4.0f, 1e-5f);
}

TEST(ops_sum_all) {
    Tensor a({3, 3});
    for (size_t i = 0; i < 9; ++i)
        a.data_ptr()[i] = 1.0f;
    Tensor s = ops::sum_all(a);
    ASSERT_EQ(s.numel(), 1u);
    ASSERT_NEAR(s.data_ptr()[0], 9.0f, 1e-5f);
}

TEST(ops_mean_all) {
    Tensor a({2, 4});
    for (size_t i = 0; i < 8; ++i)
        a.data_ptr()[i] = 2.0f;
    Tensor m = ops::mean_all(a);
    ASSERT_NEAR(m.data_ptr()[0], 2.0f, 1e-5f);
}

// ––––––––––––––––––––
// matmul
// ––––––––––––––––––––

TEST(ops_matmul_basic) {
    // (2,3) x (3,2) = (2,2)
    Tensor a({2, 3}), b({3, 2});
    // a = [[1,2,3],[4,5,6]]
    for (size_t i = 0; i < 6; ++i)
        a.data_ptr()[i] = static_cast<float>(i + 1);
    // b = [[1,0],[0,1],[0,0]]  (first two rows of identity padded)
    b.data_ptr()[0] = 1;
    b.data_ptr()[1] = 0;
    b.data_ptr()[2] = 0;
    b.data_ptr()[3] = 1;
    b.data_ptr()[4] = 0;
    b.data_ptr()[5] = 0;

    ``` Tensor c = ops::matmul(a, b);
    ASSERT_EQ(c.shape()[0], 2u);
    ASSERT_EQ(c.shape()[1], 2u);
    ASSERT_NEAR(c.at({0, 0}), 1.0f, 1e-5f);
    ASSERT_NEAR(c.at({0, 1}), 2.0f, 1e-5f);
    ASSERT_NEAR(c.at({1, 0}), 4.0f, 1e-5f);
    ASSERT_NEAR(c.at({1, 1}), 5.0f, 1e-5f);
    ```
}

TEST(ops_matmul_square) {
    // (3,3) x (3,3), verify against naive reference
    Tensor a({3, 3}), b({3, 3});
    for (size_t i = 0; i < 9; ++i) {
        a.data_ptr()[i] = static_cast<float>(i + 1);
        b.data_ptr()[i] = static_cast<float>(9 - i);
    }
    Tensor c = ops::matmul(a, b);

    ```
        // Naive reference
        for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) {
        float ref = 0.0f;
        for (size_t k = 0; k < 3; ++k)
            ref += a.at({i, k}) * b.at({k, j});
        ASSERT_NEAR(c.at({i, j}), ref, 1e-4f);
    }
    ```
}

TEST(ops_matmul_dim_mismatch) {
    Tensor a({2, 3}), b({4, 2});
    ASSERT_THROWS(ops::matmul(a, b));
}

TEST(ops_matmul_non_2d) {
    Tensor a({2, 3, 4}), b({4, 2});
    ASSERT_THROWS(ops::matmul(a, b));
}

// =============================================================================
// Phase 3A: Explicit graph (Design A) tests
// =============================================================================

TEST(graph_leaf_forward) {
    Graph g;
    Tensor t({2, 3});
    for (size_t i = 0; i < 6; ++i)
        t.data_ptr()[i] = static_cast<float>(i);
    auto x = g.leaf(t);
    ASSERT_NEAR(x->output.at({0, 0}), 0.0f, 1e-6f);
    ASSERT_NEAR(x->output.at({1, 2}), 5.0f, 1e-6f);
}

TEST(graph_add_forward) {
    Graph g;
    Tensor ta({3}), tb({3});
    ta.data_ptr()[0] = 1;
    tb.data_ptr()[0] = 2;
    auto a = g.leaf(ta);
    auto b = g.leaf(tb);
    auto c = g.add(a, b);
    ASSERT_NEAR(c->output.data_ptr()[0], 3.0f, 1e-6f);
}

TEST(graph_backward_add) {
    // loss = sum(a + b), grad_a = grad_b = ones
    Graph g;
    Tensor ta({3}), tb({3});
    for (size_t i = 0; i < 3; ++i) {
        ta.data_ptr()[i] = 1.0f;
        tb.data_ptr()[i] = 2.0f;
    }
    auto a = g.leaf(ta);
    auto b = g.leaf(tb);
    auto c = g.add(a, b);
    auto loss = g.mean_all(c);
    g.backward(loss);

    ```
        // d(mean(a+b))/d(a[i]) = 1/3
        for (size_t i = 0; i < 3; ++i) {
        ASSERT_NEAR(a->grad->data_ptr()[i], 1.0f / 3.0f, 1e-5f);
        ASSERT_NEAR(b->grad->data_ptr()[i], 1.0f / 3.0f, 1e-5f);
    }
    ```
}

TEST(graph_backward_mul) {
    // loss = mean(a * b), grad_a = b/N, grad_b = a/N
    Graph g;
    Tensor ta({4}), tb({4});
    for (size_t i = 0; i < 4; ++i) {
        ta.data_ptr()[i] = static_cast<float>(i + 1);
        tb.data_ptr()[i] = 2.0f;
    }
    auto a = g.leaf(ta);
    auto b = g.leaf(tb);
    auto c = g.mul(a, b);
    auto loss = g.mean_all(c);
    g.backward(loss);

    ``` for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(a->grad->data_ptr()[i], 2.0f / 4.0f, 1e-5f);
        ASSERT_NEAR(b->grad->data_ptr()[i], static_cast<float>(i + 1) / 4.0f,
                    1e-5f);
    }
    ```
}

TEST(graph_backward_relu) {
    Graph g;
    Tensor ta({5});
    float vals[] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.0f};
    for (size_t i = 0; i < 5; ++i)
        ta.data_ptr()[i] = vals[i];
    auto a = g.leaf(ta);
    auto r = g.relu(a);
    auto loss = g.mean_all(r);
    g.backward(loss);

    ```
        // grad_a[i] = (a[i] > 0) / 5
        float expected[] = {0.0f, 0.0f, 0.0f, 0.2f, 0.2f};
    for (size_t i = 0; i < 5; ++i)
        ASSERT_NEAR(a->grad->data_ptr()[i], expected[i], 1e-5f);
    ```
}

TEST(graph_backward_matmul) {
    // loss = mean_all(A @ B)
    // grad_A = grad @ B.T / N,  grad_B = A.T @ grad / N
    Graph g;
    Tensor tA({2, 3}), tB({3, 2});
    for (size_t i = 0; i < 6; ++i)
        tA.data_ptr()[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < 6; ++i)
        tB.data_ptr()[i] = static_cast<float>(i + 1);
    auto A = g.leaf(tA);
    auto B = g.leaf(tB);
    auto C = g.matmul(A, B);
    auto loss = g.mean_all(C);
    g.backward(loss);

    ```
        // Verify with finite differences
        auto f_A = [&](const Tensor &a) -> Tensor {
        return ops::mean_all(ops::matmul(a, tB));
    };
    float err_A = check_grad(f_A, tA, *A->grad);
    ASSERT(err_A < 1e-3f);

    auto f_B = [&](const Tensor &b) -> Tensor {
        return ops::mean_all(ops::matmul(tA, b));
    };
    float err_B = check_grad(f_B, tB, *B->grad);
    ASSERT(err_B < 1e-3f);
    ```
}

TEST(graph_no_grad_leaf) {
    // requires_grad=false: gradient should not be populated
    Graph g;
    Tensor ta({3}), tb({3});
    for (size_t i = 0; i < 3; ++i) {
        ta.data_ptr()[i] = 1.0f;
        tb.data_ptr()[i] = 1.0f;
    }
    auto a = g.leaf(ta, /*requires_grad=*/false);
    auto b = g.leaf(tb, /*requires_grad=*/true);
    auto c = g.add(a, b);
    auto loss = g.mean_all(c);
    g.backward(loss);

    ``` ASSERT(!a->grad.has_value());
    ASSERT(b->grad.has_value());
    ```
}

// =============================================================================
// Phase 3B: Implicit graph (Design B) tests
// =============================================================================

TEST(ag_add_forward) {
    AGTensor a(Tensor({3}), true);
    AGTensor b(Tensor({3}), true);
    for (size_t i = 0; i < 3; ++i) {
        a.data.data_ptr()[i] = 1.0f;
        b.data.data_ptr()[i] = 2.0f;
    }
    AGTensor c = ag::add(a, b);
    for (size_t i = 0; i < 3; ++i)
        ASSERT_NEAR(c.data.data_ptr()[i], 3.0f, 1e-6f);
}

TEST(ag_add_has_grad_fn) {
    AGTensor a(Tensor({3}), true);
    AGTensor b(Tensor({3}), true);
    AGTensor c = ag::add(a, b);
    ASSERT(!c.is_leaf());
    ASSERT(c.grad_fn != nullptr);
}

TEST(ag_no_grad_fn_when_not_required) {
    AGTensor a(Tensor({3}), false);
    AGTensor b(Tensor({3}), false);
    AGTensor c = ag::add(a, b);
    ASSERT(c.is_leaf());
}

TEST(ag_backward_add) {
    AGTensor a(Tensor({4}), true);
    AGTensor b(Tensor({4}), true);
    for (size_t i = 0; i < 4; ++i) {
        a.data.data_ptr()[i] = static_cast<float>(i);
        b.data.data_ptr()[i] = 1.0f;
    }
    AGTensor loss = ag::mean_all(ag::add(a, b));
    loss.backward();

    ``` for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(a.grad->data_ptr()[i], 0.25f, 1e-5f);
        ASSERT_NEAR(b.grad->data_ptr()[i], 0.25f, 1e-5f);
    }
    ```
}

TEST(ag_backward_chain) {
    // loss = mean_all(relu(a * b + c))
    AGTensor a(Tensor({4}), true);
    AGTensor b(Tensor({4}), true);
    AGTensor c(Tensor({4}), true);
    for (size_t i = 0; i < 4; ++i) {
        a.data.data_ptr()[i] = static_cast<float>(i) - 1.5f; // mix of pos/neg
        b.data.data_ptr()[i] = 2.0f;
        c.data.data_ptr()[i] = 0.5f;
    }
    AGTensor loss = ag::mean_all(ag::relu(ag::add(ag::mul(a, b), c)));
    loss.backward();

    ```
        // Verify a's gradient with finite differences
        auto f = [&](const Tensor &x) -> Tensor {
        Tensor bdata = b.data, cdata = c.data;
        return ops::mean_all(ops::relu(ops::add(ops::mul(x, bdata), cdata)));
    };
    float err = check_grad(f, a.data, *a.grad);
    ASSERT(err < 1e-3f);
    ```
}

TEST(ag_backward_matmul) {
    AGTensor A(Tensor({3, 4}), true);
    AGTensor B(Tensor({4, 2}), true);
    for (size_t i = 0; i < 12; ++i)
        A.data.data_ptr()[i] = static_cast<float>(i + 1) * 0.1f;
    for (size_t i = 0; i < 8; ++i)
        B.data.data_ptr()[i] = static_cast<float>(i + 1) * 0.1f;

    ``` AGTensor loss = ag::mean_all(ag::matmul(A, B));
    loss.backward();

    auto f_A = [&](const Tensor &a) -> Tensor {
        return ops::mean_all(ops::matmul(a, B.data));
    };
    ASSERT(check_grad(f_A, A.data, *A.grad) < 1e-3f);

    auto f_B = [&](const Tensor &b) -> Tensor {
        return ops::mean_all(ops::matmul(A.data, b));
    };
    ASSERT(check_grad(f_B, B.data, *B.grad) < 1e-3f);
    ```
}

TEST(ag_backward_log) {
    AGTensor a(Tensor({4}), true);
    for (size_t i = 0; i < 4; ++i)
        a.data.data_ptr()[i] = static_cast<float>(i + 1); // all positive

    ``` AGTensor loss = ag::mean_all(ag::log(a));
    loss.backward();

    auto f = [](const Tensor &x) -> Tensor {
        return ops::mean_all(ops::log(x));
    };
    ASSERT(check_grad(f, a.data, *a.grad) < 1e-3f);
    ```
}

TEST(ag_linear_regression_step) {
    // One gradient descent step on loss = mean((Xw - y)^2)
    // Verify loss decreases.
    size_t N = 8, D = 4;
    AGTensor X(Tensor({N, D}), false);
    AGTensor w(Tensor({D, 1}), true);
    AGTensor y(Tensor({N, 1}), false);

    ``` for (size_t i = 0; i < N * D; ++i) X.data.data_ptr()[i] =
        static_cast<float>(i) * 0.1f;
    for (size_t i = 0; i < D; ++i)
        w.data.data_ptr()[i] = 0.1f;
    for (size_t i = 0; i < N; ++i)
        y.data.data_ptr()[i] = 1.0f;

    auto forward = [&]() {
        AGTensor pred = ag::matmul(X, w);
        AGTensor diff = ag::sub(pred, y);
        AGTensor sq = ag::mul(diff, diff);
        return ag::mean_all(sq);
    };

    AGTensor loss0 = forward();
    loss0.backward();

    float lr = 0.01f;
    for (size_t i = 0; i < D; ++i)
        w.data.data_ptr()[i] -= lr * w.grad->data_ptr()[i];

    // Reset grad for next forward pass
    w.grad.reset();

    AGTensor loss1 = forward();
    ASSERT(loss1.data.data_ptr()[0] < loss0.data.data_ptr()[0]);
    ```
}

// =============================================================================
// main
// =============================================================================

int main() {
    std::cout << “Running Phase 2 & 3 tests…\n\n”;
    std::cout << “\n” << tests_passed << “ / “ << tests_run << “ passed\n”;
    return (tests_passed == tests_run) ? 0 : 1;
}
