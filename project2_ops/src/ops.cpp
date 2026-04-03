#include "ops.hpp"
#include "storage.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

std::vector<size_t> ops::broadcast_shape(const std::vector<size_t> &a,
                                         const std::vector<size_t> &b) {
    std::vector<size_t> const &min = a.size() < b.size() ? a : b;
    std::vector<size_t> const &max = a.size() < b.size() ? b : a;
    size_t min_len = min.size();
    size_t max_len = max.size();
    std::vector<size_t> output_dims = max;
    for (size_t i = 0; i < min_len; ++i) {
        size_t offset = i + 1;
        size_t min_dim = min[min_len - offset];
        size_t max_dim = max[max_len - offset];
        if (min_dim == 1) {
            output_dims[max_len - offset] = max_dim;
        } else if (max_dim == 1) {
            output_dims[max_len - offset] = min_dim;
        } else if (min_dim == max_dim) {
            output_dims[max_len - offset] = min_dim;
        } else {
            std::ostringstream oss{"Could not broadcast at dim -"};
            oss << offset << std::endl;
            throw std::runtime_error(oss.str());
        }
    }
    return output_dims;
}

// elementwise
void ops::add(const Tensor &a, const Tensor &b, Tensor &out) {
    auto op = [](const float &a_, const float &b_) { return a_ + b_; };
    generic_op(a, b, out, op);
}
Tensor ops::add(const Tensor &a, const Tensor &b) {
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());
    Tensor out = Tensor(out_dims);
    add(a, b, out);
    return out;
}

void ops::sub(const Tensor &a, const Tensor &b, Tensor &out) {
    auto op = [](const float &a_, const float &b_) { return a_ - b_; };
    generic_op(a, b, out, op);
}
Tensor ops::sub(const Tensor &a, const Tensor &b) {
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());
    Tensor out = Tensor(out_dims);
    add(a, b, out);
    return out;
}

void ops::mul(const Tensor &a, const Tensor &b, Tensor &out) {
    auto op = [](const float &a_, const float &b_) { return a_ * b_; };
    generic_op(a, b, out, op);
}
Tensor ops::mul(const Tensor &a, const Tensor &b) {
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());
    Tensor out = Tensor(out_dims);
    mul(a, b, out);
    return out;
}

void ops::div(const Tensor &a, const Tensor &b, Tensor &out) {
    auto op = [](const float &a_, const float &b_) { return a_ / b_; };
    generic_op(a, b, out, op);
}
Tensor ops::div(const Tensor &a, const Tensor &b) {
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());
    Tensor out = Tensor(out_dims);
    div(a, b, out);
    return out;
}

void ops::relu(const Tensor &a, Tensor &out) {
    auto op = [](const float &a_) { return std::max(a_, 0.0f); };
    generic_op(a, out, op);
}
Tensor ops::relu(const Tensor &a) {
    Tensor out = Tensor(a.shape());
    relu(a, out);
    return out;
}

void ops::exp(const Tensor &a, Tensor &out) {
    auto op = [](const float &a_) { return std::exp(a_); };
    generic_op(a, out, op);
}
Tensor ops::exp(const Tensor &a) {
    Tensor out = Tensor(a.shape());
    exp(a, out);
    return out;
}

void ops::log(const Tensor &a, Tensor &out) {
    auto op = [](const float &a_) { return std::log(a_); };
    generic_op(a, out, op);
}
Tensor ops::log(const Tensor &a) {
    Tensor out = Tensor(a.shape());
    log(a, out);
    return out;
}

// reductions

std::vector<size_t> drop_axis(const Tensor &a, size_t axis) {
    std::vector<size_t> new_shape = a.shape();
    new_shape.erase(new_shape.begin() + axis);
    return new_shape;
}

void reduction_axis_and_shape_check(const Tensor &a, size_t axis, Tensor &out) {
    if (a.ndim() <= axis) {
        throw std::runtime_error("Invalid axis (too large)!");
    }
    if (axis < 0) {
        throw std::runtime_error("Invalid axis (negative)!");
    }
    // check dims match
    std::vector<size_t> new_shape = drop_axis(a, axis);
    if (new_shape != out.shape()) {
        throw std::runtime_error("Output shape doesn't match input + axis!");
    }
}

void ops::sum(const Tensor &a, size_t axis, Tensor &out) {
    reduction_axis_and_shape_check(a, axis, out);

    std::vector<size_t> idx = std::vector<size_t>(out.ndim());
    for (size_t n = 0; n < out.numel(); ++n) {
        std::vector<size_t> idx_a = idx;
        idx_a.insert(idx_a.begin() + axis, 0);

        // accumulate over axis
        float acc = 0;
        for (size_t id = 0; id < a.shape()[axis]; ++id) {
            acc += a.at(idx_a);
        }
        // store acc in out
        out.at(idx) = acc;

        // increment out idx
        for (int d = out.ndim() - 1; d >= 0; --d) {
            ++idx[d];
            if (idx[d] < out.shape()[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
Tensor ops::sum(const Tensor &a, size_t axis) {
    std::vector<size_t> new_shape = drop_axis(a, axis);
    Tensor out = Tensor(new_shape);
    sum(a, axis, out);
    return out;
}

void ops::mean(const Tensor &a, size_t axis, Tensor &out) {
    reduction_axis_and_shape_check(a, axis, out);

    std::vector<size_t> idx = std::vector<size_t>(out.ndim());
    for (size_t n = 0; n < out.numel(); ++n) {
        std::vector<size_t> idx_a = idx;
        idx_a.insert(idx_a.begin() + axis, 0);

        // accumulate over axis
        float acc = 0;
        for (size_t id = 0; id < a.shape()[axis]; ++id) {
            acc += a.at(idx_a);
        }
        // store average of acc in out
        out.at(idx) = acc / a.shape()[axis];

        // increment out idx
        for (int d = out.ndim() - 1; d >= 0; --d) {
            ++idx[d];
            if (idx[d] < out.shape()[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}
Tensor ops::mean(const Tensor &a, size_t axis) {
    std::vector<size_t> new_shape = drop_axis(a, axis);
    Tensor out = Tensor(new_shape);
    mean(a, axis, out);
    return out;
}

Tensor sum_all(const Tensor &a) {
    float acc = 0;
    const float *base = a.data_ptr();
    for (size_t i = 0; i < a.numel(); ++i) {
        acc += base[i];
    }
    Tensor out = Tensor({1});
    out.at({0}) = acc;
    return out;
}
Tensor mean_all(const Tensor &a) {
    float acc = 0;
    const float *base = a.data_ptr();
    for (size_t i = 0; i < a.numel(); ++i) {
        acc += base[i];
    }
    Tensor out = Tensor({1});
    out.at({0}) = acc / a.numel();
    return out;
}

// matmul
void ops::matmul(const Tensor &a, const Tensor &b, Tensor &out) {}
Tensor ops::matmul(const Tensor &a, const Tensor &b) { return Tensor({1}); }

// helpers
void ops::generic_op(const Tensor &a, Tensor &out, float (*op)(const float &)) {
    // throw if output has wrong shape
    if (a.shape() != out.shape()) {
        throw std::runtime_error("Output and input dims don't match!");
    }

    // iterate through a and out
    std::vector<size_t> idx(a.ndim());
    for (size_t n = 0; n < a.numel(); ++n) {
        out.at(idx) = op(a.at(idx));

        for (int d = a.ndim() - 1; d >= 0; --d) {
            ++idx[d];
            if (idx[d] < a.shape()[d])
                break;
            idx[d] = 0;
        }
    }
}

void ops::generic_op(const Tensor &a,
                     const Tensor &b,
                     Tensor &out,
                     float (*op)(const float &, const float &)) {
    // broadcast_shape throws error if a and b are not broadcastable
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());

    // throw if output has wrong shape
    if (out.shape() != out_dims) {
        throw std::runtime_error("Output dims don't match broadcast dims");
    }

    // operation will be performed on broadcast tensors a_, b_
    Tensor a_ = Tensor(out_dims);
    Tensor b_ = Tensor(out_dims);
    int dim_diff = a_.ndim() - b_.ndim();

    // could save on copying at the cost of reusing code in each branch
    if (dim_diff < 0) { // broadcast a to a_
        broadcast_to(a, a_);
        b_ = b;
    } else if (dim_diff > 0) { // broadcast b to b_
        a_ = a;
        broadcast_to(b, b_);
    } else {
        a_ = a;
        b_ = b;
    }

    // iterate through
    std::vector<size_t> idx(a.ndim());
    for (size_t n = 0; n < a.numel(); ++n) {
        // perform op
        out.at(idx) = op(a.at(idx), b.at(idx));

        // increment idx
        for (int d = a.ndim() - 1; d >= 0; --d) {
            ++idx[d];
            if (idx[d] < a.shape()[d])
                break;
            idx[d] = 0;
        }
    }
}
void ops::broadcast_to(const Tensor &src, Tensor &out) {
    size_t mod = src.numel();
    size_t bound = out.numel();
    for (size_t id = 0; id < bound; ++id) {
        out.at({id}) = src.at({id % mod});
    }
}
