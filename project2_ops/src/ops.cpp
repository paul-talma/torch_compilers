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

void ops::add(const Tensor &a, const Tensor &b, Tensor &out) {
    // broadcast_shape throws error if a and b are not broadcastable
    std::vector<size_t> out_dims = broadcast_shape(a.shape(), b.shape());

    // throw if output has wrong shape
    if (out.shape() != out_dims) {
        throw std::runtime_error("Output dims don't broadcast dims");
    }

    // operation will be performed on broadcast tensors a_, b_
    Tensor a_ = Tensor(out_dims);
    Tensor b_ = Tensor(out_dims);
    int dim_diff = a_.ndim() - b_.ndim();

    // could save on copying at the cost of reusing code in each branch
    if (dim_diff < 0) {
        broadcast_to(a, a_);
        b_ = b;
    } else if (dim_diff > 0) {
        a_ = a;
        broadcast_to(b, b_);
    } else {
        a_ = a;
        b_ = b;
    }

    // iterate through
    size_t bound = out.numel();
    for (size_t id = 0; id < bound; ++id) {
        out.at({id}) = a_.at({id}) + b_.at({id});
    }
}

Tensor ops::add(const Tensor &a, const Tensor &b) {}

// helpers
void ops::broadcast_to(const Tensor &src, Tensor &out) {
    // get
    size_t mod = src.numel();
    size_t bound = out.numel();
    for (size_t id = 0; id < bound; ++id) {
        out.at({id}) = src.at({id % mod});
    }
}
