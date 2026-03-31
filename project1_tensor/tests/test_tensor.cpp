#include "storage.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>

// =============================================================================
// Test helpers
// =============================================================================

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                             \
    static void name();                                                        \
    struct name##_runner {                                                     \
        name##_runner() {                                                      \
            tests_run++;                                                       \
            try {                                                              \
                name();                                                        \
                tests_passed++;                                                \
                std::cout << "  PASS  " << #name << "\n";                      \
            } catch (const std::exception &e) {                                \
                std::cout << "  FAIL  " << #name << "\n         " << e.what()  \
                          << "\n";                                             \
            }                                                                  \
        }                                                                      \
    } name##_instance;                                                         \
    static void name()

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond))                                                           \
            throw std::runtime_error("Assertion failed: " #cond " at line " +  \
                                     std::to_string(__LINE__));                \
    } while (0)

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                       \
        auto _a = (a);                                                         \
        auto _b = (b);                                                         \
        if (_a != _b) {                                                        \
            std::ostringstream _oss;                                           \
            _oss << "Expected equal: " #a " == " #b << " (" << _a << " vs "    \
                 << _b << ")"                                                  \
                 << " at line " << __LINE__;                                   \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        auto _a = (a);                                                         \
        auto _b = (b);                                                         \
        if (std::fabs(_a - _b) > (tol)) {                                      \
            std::ostringstream _oss;                                           \
            _oss << "Expected near: " #a " ~ " #b << " (" << _a << " vs "      \
                 << _b << ")"                                                  \
                 << " at line " << __LINE__;                                   \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

#define ASSERT_THROWS(expr)                                                    \
    do {                                                                       \
        bool threw = false;                                                    \
        try {                                                                  \
            (expr);                                                            \
        } catch (...) {                                                        \
            threw = true;                                                      \
        }                                                                      \
        if (!threw)                                                            \
            throw std::runtime_error("Expected exception from: " #expr         \
                                     " at line " +                             \
                                     std::to_string(__LINE__));                \
    } while (0)

// =============================================================================
// Storage tests
// =============================================================================

TEST(storage_size) {
    Storage s(16);
    ASSERT_EQ(s.size(), 16u);
}

TEST(storage_zero_initialized) {
    Storage s(8);
    for (size_t i = 0; i < s.size(); ++i)
        ASSERT_EQ(s.data()[i], 0.0f);
}

TEST(storage_read_write) {
    Storage s(4);
    s.data()[0] = 1.0f;
    s.data()[3] = 9.0f;
    ASSERT_EQ(s.data()[0], 1.0f);
    ASSERT_EQ(s.data()[3], 9.0f);
}

TEST(storage_copy_constructor) {
    Storage a(4);
    a.data()[0] = 42.0f;

    Storage b = a; // copy constructor

    // Same values
    ASSERT_EQ(b.data()[0], 42.0f);
    ASSERT_EQ(b.size(), 4u);

    // Independent buffers — mutating b does not affect a
    b.data()[0] = 0.0f;
    ASSERT_EQ(a.data()[0], 42.0f);
}

TEST(storage_copy_assignment) {
    Storage a(4);
    a.data()[1] = 7.0f;

    Storage b(2); // different size
    b = a;        // copy assignment

    ASSERT_EQ(b.size(), 4u);
    ASSERT_EQ(b.data()[1], 7.0f);

    // Independent
    b.data()[1] = 0.0f;
    ASSERT_EQ(a.data()[1], 7.0f);
}

TEST(storage_move_constructor) {
    Storage a(4);
    a.data()[2] = 5.0f;
    float *original_ptr = a.data();

    Storage b = std::move(a); // move constructor

    // b owns the original buffer
    ASSERT_EQ(b.data(), original_ptr);
    ASSERT_EQ(b.data()[2], 5.0f);
    ASSERT_EQ(b.size(), 4u);

    // a is left in a valid but empty state
    // (data() == nullptr and size() == 0 is the expected contract)
    ASSERT_EQ(a.data(), nullptr);
    ASSERT_EQ(a.size(), 0u);
}

TEST(storage_move_assignment) {
    Storage a(4);
    a.data()[0] = 3.0f;
    float *original_ptr = a.data();

    Storage b(2);
    b = std::move(a);

    ASSERT_EQ(b.data(), original_ptr);
    ASSERT_EQ(b.data()[0], 3.0f);
    ASSERT_EQ(a.data(), nullptr);
    ASSERT_EQ(a.size(), 0u);
}

TEST(storage_self_assignment) {
    Storage a(4);
    a.data()[0] = 1.0f;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
    a = a; // should not crash or corrupt
#pragma clang diagnostic pop
    ASSERT_EQ(a.data()[0], 1.0f);
    ASSERT_EQ(a.size(), 4u);
}

// =============================================================================
// Tensor construction & metadata
// =============================================================================

TEST(tensor_shape_1d) {
    Tensor t({5});
    ASSERT_EQ(t.ndim(), 1u);
    ASSERT_EQ(t.shape()[0], 5u);
    ASSERT_EQ(t.numel(), 5u);
}

TEST(tensor_shape_2d) {
    Tensor t({3, 4});
    ASSERT_EQ(t.ndim(), 2u);
    ASSERT_EQ(t.shape()[0], 3u);
    ASSERT_EQ(t.shape()[1], 4u);
    ASSERT_EQ(t.numel(), 12u);
}

TEST(tensor_shape_3d) {
    Tensor t({3, 4, 5});
    ASSERT_EQ(t.ndim(), 3u);
    ASSERT_EQ(t.numel(), 60u);
}

TEST(tensor_row_major_strides_1d) {
    Tensor t({7});
    ASSERT_EQ(t.strides()[0], 1u);
}

TEST(tensor_row_major_strides_2d) {
    Tensor t({3, 4});
    // strides: {4, 1}
    ASSERT_EQ(t.strides()[0], 4u);
    ASSERT_EQ(t.strides()[1], 1u);
}

TEST(tensor_row_major_strides_3d) {
    Tensor t({3, 4, 5});
    // strides: {20, 5, 1}
    ASSERT_EQ(t.strides()[0], 20u);
    ASSERT_EQ(t.strides()[1], 5u);
    ASSERT_EQ(t.strides()[2], 1u);
}

TEST(tensor_zero_initialized) {
    Tensor t({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_EQ(t.at({i, j}), 0.0f);
}

TEST(tensor_is_contiguous_fresh) {
    Tensor t({3, 4, 5});
    ASSERT(t.is_contiguous());
}

TEST(tensor_offset_fresh) {
    Tensor t({3, 4});
    ASSERT_EQ(t.offset(), 0u);
}

// =============================================================================
// Element access
// =============================================================================

TEST(tensor_at_write_read_1d) {
    Tensor t({5});
    t.at({2}) = 99.0f;
    ASSERT_EQ(t.at({2}), 99.0f);
}

TEST(tensor_at_write_read_2d) {
    Tensor t({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            t.at({i, j}) = static_cast<float>(i * 4 + j);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_EQ(t.at({i, j}), static_cast<float>(i * 4 + j));
}

TEST(tensor_at_write_read_3d) {
    Tensor t({2, 3, 4});
    t.at({1, 2, 3}) = 42.0f;
    ASSERT_EQ(t.at({1, 2, 3}), 42.0f);
}

TEST(tensor_at_out_of_range) {
    Tensor t({3, 4});
    ASSERT_THROWS(t.at({3, 0})); // dim 0 out of range
    ASSERT_THROWS(t.at({0, 4})); // dim 1 out of range
}

TEST(tensor_at_row_major_layout) {
    // Verify that at() uses the correct flat index.
    // For shape (2,3), row-major: element (i,j) is at buffer[i*3 + j].
    Tensor t({2, 3});
    float *p = t.data_ptr();
    for (int k = 0; k < 6; ++k)
        p[k] = static_cast<float>(k);

    ASSERT_EQ(t.at({0, 0}), 0.0f);
    ASSERT_EQ(t.at({0, 2}), 2.0f);
    ASSERT_EQ(t.at({1, 0}), 3.0f);
    ASSERT_EQ(t.at({1, 2}), 5.0f);
}

// =============================================================================
// Transpose
// =============================================================================

TEST(transpose_shape) {
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    ASSERT_EQ(tr.shape()[0], 4u);
    ASSERT_EQ(tr.shape()[1], 3u);
}

TEST(transpose_strides) {
    Tensor t({3, 4});
    // original strides: {4, 1}
    Tensor tr = t.transpose(0, 1);
    // transposed strides: {1, 4}
    ASSERT_EQ(tr.strides()[0], 1u);
    ASSERT_EQ(tr.strides()[1], 4u);
}

TEST(transpose_shares_storage) {
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    // Both point at the same Storage
    ASSERT_EQ(t.storage().get(), tr.storage().get());
}

TEST(transpose_values) {
    Tensor t({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            t.at({i, j}) = static_cast<float>(i * 4 + j);

    Tensor tr = t.transpose(0, 1);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_EQ(tr.at({j, i}), t.at({i, j}));
}

TEST(transpose_not_contiguous) {
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    ASSERT(!tr.is_contiguous());
}

TEST(transpose_3d) {
    Tensor t({2, 3, 4});
    Tensor tr = t.transpose(0, 2);
    ASSERT_EQ(tr.shape()[0], 4u);
    ASSERT_EQ(tr.shape()[1], 3u);
    ASSERT_EQ(tr.shape()[2], 2u);
    ASSERT_EQ(tr.strides()[0], 1u);  // was strides[2]
    ASSERT_EQ(tr.strides()[2], 12u); // was strides[0]
}

TEST(transpose_out_of_range) {
    Tensor t({3, 4});
    ASSERT_THROWS(t.transpose(0, 2));
}

TEST(transpose_write_through) {
    // Writing through the transposed view should be visible in the original.
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    tr.at({2, 1}) = 77.0f;
    ASSERT_EQ(t.at({1, 2}), 77.0f);
}

// =============================================================================
// Reshape
// =============================================================================

TEST(reshape_1d_to_2d) {
    Tensor t({12});
    Tensor r = t.reshape({3, 4});
    ASSERT_EQ(r.shape()[0], 3u);
    ASSERT_EQ(r.shape()[1], 4u);
    ASSERT_EQ(r.numel(), 12u);
}

TEST(reshape_shares_storage) {
    Tensor t({12});
    Tensor r = t.reshape({3, 4});
    ASSERT_EQ(t.storage().get(), r.storage().get());
}

TEST(reshape_values_preserved) {
    Tensor t({6});
    for (size_t i = 0; i < 6; ++i)
        t.at({i}) = static_cast<float>(i);

    Tensor r = t.reshape({2, 3});
    ASSERT_EQ(r.at({0, 0}), 0.0f);
    ASSERT_EQ(r.at({0, 2}), 2.0f);
    ASSERT_EQ(r.at({1, 0}), 3.0f);
    ASSERT_EQ(r.at({1, 2}), 5.0f);
}

TEST(reshape_wrong_numel) {
    Tensor t({6});
    ASSERT_THROWS(t.reshape({2, 4})); // 8 != 6
}

TEST(reshape_non_contiguous_fails) {
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    ASSERT_THROWS(tr.reshape({12}));
}

TEST(reshape_contiguous) {
    Tensor t({3, 4});
    Tensor r = t.reshape({12});
    ASSERT(r.is_contiguous());
}

// =============================================================================
// Slice
// =============================================================================

TEST(slice_shape) {
    Tensor t({6, 4});
    Tensor s = t.slice(0, 1, 4); // rows 1..3
    ASSERT_EQ(s.shape()[0], 3u);
    ASSERT_EQ(s.shape()[1], 4u);
}

TEST(slice_shares_storage) {
    Tensor t({6, 4});
    Tensor s = t.slice(0, 1, 4);
    ASSERT_EQ(t.storage().get(), s.storage().get());
}

TEST(slice_offset) {
    Tensor t({6, 4});
    // Slicing dim 0 from index 2: offset should be 2 * strides[0] = 2*4 = 8
    Tensor s = t.slice(0, 2, 5);
    ASSERT_EQ(s.offset(), 8u);
}

TEST(slice_values) {
    Tensor t({5, 3});
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 3; ++j)
            t.at({i, j}) = static_cast<float>(i * 10 + j);

    Tensor s = t.slice(0, 2, 4); // rows 2 and 3
    ASSERT_EQ(s.at({0, 0}), 20.0f);
    ASSERT_EQ(s.at({0, 2}), 22.0f);
    ASSERT_EQ(s.at({1, 1}), 31.0f);
}

TEST(slice_dim1) {
    Tensor t({4, 6});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 6; ++j)
            t.at({i, j}) = static_cast<float>(j);

    Tensor s = t.slice(1, 2, 5); // columns 2,3,4
    ASSERT_EQ(s.shape()[0], 4u);
    ASSERT_EQ(s.shape()[1], 3u);
    ASSERT_EQ(s.at({0, 0}), 2.0f);
    ASSERT_EQ(s.at({0, 2}), 4.0f);
}

TEST(slice_write_through) {
    Tensor t({4, 4});
    Tensor s = t.slice(0, 1, 3);
    s.at({0, 0}) = 55.0f;
    ASSERT_EQ(t.at({1, 0}), 55.0f);
}

TEST(slice_out_of_range) {
    Tensor t({5, 4});
    ASSERT_THROWS(t.slice(0, 3, 6)); // end > shape[0]
    ASSERT_THROWS(t.slice(0, 3, 3)); // start >= end
    ASSERT_THROWS(t.slice(2, 0, 1)); // dim >= ndim
}

// =============================================================================
// Copy / move semantics for Tensor
// =============================================================================

TEST(tensor_copy_constructor_shares_storage) {
    Tensor a({3, 4});
    a.at({0, 0}) = 7.0f;

    Tensor b = a; // copy constructor — shallow copy

    // Same storage
    ASSERT_EQ(a.storage().get(), b.storage().get());
    ASSERT_EQ(b.at({0, 0}), 7.0f);

    // Mutation through b is visible in a (shared buffer)
    b.at({0, 0}) = 99.0f;
    ASSERT_EQ(a.at({0, 0}), 99.0f);
}

TEST(tensor_copy_constructor_independent_metadata) {
    Tensor a({3, 4});
    Tensor b = a;

    // Shape vectors are independent copies — modifying one's view metadata
    // won't affect the other. (We can't mutate shape directly, so we verify
    // via transpose producing independent objects.)
    Tensor ta = a.transpose(0, 1);
    Tensor tb = b.transpose(0, 1);
    ASSERT_EQ(ta.shape()[0], tb.shape()[0]);
}

TEST(tensor_copy_assignment) {
    Tensor a({3, 4});
    a.at({1, 1}) = 5.0f;

    Tensor b({2, 2});
    b = a;

    ASSERT_EQ(b.shape()[0], 3u);
    ASSERT_EQ(b.at({1, 1}), 5.0f);
    ASSERT_EQ(a.storage().get(), b.storage().get());
}

TEST(tensor_move_constructor) {
    Tensor a({3, 4});
    a.at({0, 0}) = 42.0f;
    Storage *original_storage = a.storage().get();

    Tensor b = std::move(a);

    ASSERT_EQ(b.storage().get(), original_storage);
    ASSERT_EQ(b.at({0, 0}), 42.0f);
    ASSERT_EQ(b.shape()[0], 3u);
}

TEST(tensor_move_assignment) {
    Tensor a({3, 4});
    a.at({2, 3}) = 13.0f;
    Storage *original_storage = a.storage().get();

    Tensor b({1});
    b = std::move(a);

    ASSERT_EQ(b.storage().get(), original_storage);
    ASSERT_EQ(b.at({2, 3}), 13.0f);
}

TEST(tensor_self_copy_assignment) {
    Tensor a({3, 4});
    a.at({0, 0}) = 1.0f;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
    a = a;
#pragma clang diagnostic pop
    ASSERT_EQ(a.at({0, 0}), 1.0f);
    ASSERT_EQ(a.shape()[0], 3u);
}

// =============================================================================
// data_ptr
// =============================================================================

TEST(data_ptr_contiguous) {
    Tensor t({3});
    float *p = t.data_ptr();
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    ASSERT_EQ(t.at({0}), 1.0f);
    ASSERT_EQ(t.at({1}), 2.0f);
    ASSERT_EQ(t.at({2}), 3.0f);
}

TEST(data_ptr_after_slice_offset) {
    Tensor t({6});
    for (size_t i = 0; i < 6; ++i)
        t.at({i}) = static_cast<float>(i);

    Tensor s = t.slice(0, 2, 5);
    // data_ptr() should point to element 2 in the buffer
    ASSERT_EQ(s.data_ptr()[0], 2.0f);
}

// =============================================================================
// main
// =============================================================================

int main() {
    std::cout << "Running tests...\n\n";
    std::cout << tests_passed << " / " << tests_run << " passed\n";

    // pretty printing tests
    // Tensor t({2, 2, 3});
    // for (size_t i = 0; i < 2; ++i) {
    //     for (size_t j = 0; j < 2; ++j) {
    //         for (size_t k = 0; k < 3; ++k) {
    //             t.at({i, j, k}) = static_cast<float>(i * 6 + j * 3 + k);
    //         }
    //     }
    // }
    // t.print();
    //
    // Tensor t1({2, 3});
    // for (size_t i = 0; i < 2; ++i) {
    //     for (size_t j = 0; j < 2; ++j) {
    //         t1.at({i, j}) = static_cast<float>(i * 3 + j);
    //     }
    // }
    // t1.print();

    return (tests_passed == tests_run) ? 0 : 1;
}
