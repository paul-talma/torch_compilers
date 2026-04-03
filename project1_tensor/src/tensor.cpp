#include "tensor.hpp"
#include "storage.hpp"
#include "types.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

Tensor::Tensor(std::vector<size_t> shape) {
    // shape
    shape_ = shape;

    // strides
    strides_ = compute_strides(shape);

    // offset
    offset_ = 0;

    // storage
    size_t size = 1;
    for (size_t d : shape) {
        size *= d;
    }
    storage_ = std::make_shared<Storage>(size);
}
Tensor::Tensor(std::shared_ptr<Storage> storage,
               std::vector<size_t> shape,
               std::vector<size_t> strides,
               size_t offset) {
    storage_ = storage;
    shape_ = shape;
    strides_ = strides;
    offset_ = offset;
}

Tensor::~Tensor() {}

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), strides_(other.strides_),
      offset_(other.offset_) { // copy constructor
    storage_ = other.storage_;
}
Tensor &Tensor::operator=(const Tensor &other) { // copy assignment
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        offset_ = other.offset_;
        storage_ = other.storage_;
    }
    return *this;
}

Tensor::Tensor(Tensor &&other) noexcept // move constructor
    : storage_(std::move(other.storage_)), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), offset_(other.offset_) {
    other.offset_ = 0;
}
Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        offset_ = other.offset_;

        other.offset_ = 0;
    }
    return *this;
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }
const std::vector<size_t> &Tensor::strides() const { return strides_; }
size_t Tensor::ndim() const { return shape_.size(); }
size_t Tensor::numel() const {
    u32 count = 1;
    for (auto i : shape_) {
        count *= i;
    }
    return count;
}
size_t Tensor::offset() const { return offset_; }
bool Tensor::is_contiguous() const {
    u32 stride = 1;
    for (int i = strides_.size() - 1; i >= 0; --i) {
        if (stride != strides_[i]) {
            return false;
        }
        stride *= shape_[i];
    }
    return true;
}

float Tensor::at(std::vector<size_t> idx) const {
    size_t pos = flat_index(idx);
    return storage_->data()[pos];
}

float &Tensor::at(std::vector<size_t> idx) {
    size_t pos = flat_index(idx);
    return storage_->data()[pos];
}

float Tensor::at(std::initializer_list<size_t> idx) const {
    size_t pos = flat_index(idx);
    return storage_->data()[pos];
}

float &Tensor::at(std::initializer_list<size_t> idx) {
    size_t pos = flat_index(idx);
    return storage_->data()[pos];
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Reshaping a non-contiguous array!");
    }
    u32 prod = 1;
    for (auto s : new_shape) {
        prod *= s;
    }
    if (prod != numel()) {
        std::ostringstream oss;
        oss << "New shape doesn't match number of elements! Expected: "
            << numel() << ". Received: " << prod;
        throw std::runtime_error(oss.str());
    }

    std::vector<size_t> new_strides = compute_strides(new_shape);
    return Tensor(storage_, new_shape, new_strides, offset_);
}

std::vector<size_t>
swap_dims(std::vector<size_t> vec, size_t dim0, size_t dim1) {
    std::vector<size_t> new_vec = vec;
    size_t temp = new_vec[dim0];
    new_vec[dim0] = new_vec[dim1];
    new_vec[dim1] = temp;
    return new_vec;
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    size_t dims = ndim();
    if (dim0 >= dims) {
        std::ostringstream oss;
        oss << "Dimension 0 out of range! Given: " << dim0 << ". Max: " << dims;
        throw std::runtime_error(oss.str());
    }
    if (dim1 >= dims) {
        std::ostringstream oss;
        oss << "Dimension 1 out of range! Given: " << dim1 << ". Max: " << dims;
        throw std::runtime_error(oss.str());
    }

    std::vector<size_t> new_shape = swap_dims(shape_, dim0, dim1);
    std::vector<size_t> new_strides = swap_dims(strides_, dim0, dim1);
    return Tensor(storage_, new_shape, new_strides, offset_);
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t dims = ndim();
    if (dim >= dims) {
        std::ostringstream oss;
        oss << "Dimension out of bounds! Given: " << dim << ". Max: " << dims;
        throw std::out_of_range(oss.str());
    }
    if (end >= shape_[dim]) {
        std::ostringstream oss;
        oss << "End index out of bounds! Given: " << end
            << ". Max: " << shape_[dim];
        throw std::out_of_range(oss.str());
    }
    if (start >= end) {
        throw std::out_of_range("Start index must be <= end index!");
    }

    size_t new_offset = offset_ + start * strides_[dim];
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = end - start;

    return Tensor(storage_, new_shape, strides_, new_offset);
}

float *Tensor::data_ptr() { return storage_->data() + offset_; }

const float *Tensor::data_ptr() const { return storage_->data(); }

std::shared_ptr<Storage> Tensor::storage() { return storage_; }

std::shared_ptr<const Storage> Tensor::storage() const { return storage_; }

u32 width(u32 x) {
    // TODO: compute width of str representation of x
    return 0;
}

void Tensor::print() const {
    std::vector<size_t> idx = std::vector<size_t>(ndim());
    std::ostringstream oss;
    u32 max_elem_width = width(max());
    print_rec(0, idx, oss, max_elem_width);
    oss << "\n";
    std::cout << oss.str();
}

// TODO: implement
std::string to_str(float elem, u32 max_elem_width) { return "0"; }

void Tensor::print_rec(size_t dim,
                       std::vector<size_t> idx,
                       std::ostringstream &oss,
                       u32 max_elem_width) const {
    // base case
    if (dim == ndim()) {
        // TODO: this should use `at` but need to cast idx to initializer list
        size_t pos = flat_index(idx);
        float elem = storage_->data()[pos];
        oss << to_str(elem, max_elem_width);
        return;
    }

    // open bracket
    oss << "[";
    for (size_t d = 0; d < shape_[dim] - 1; ++d) {
        // display next element of lower dim
        print_rec(dim + 1, idx, oss, max_elem_width);

        // commas, whitespace
        oss << ",";
        if (dim + 1 == ndim()) {
            oss << " ";
        } else if (dim + 2 == ndim()) {
            oss << "\n";
        } else {
            oss << "\n\n";
        }

        if (dim + 1 < ndim()) {
            for (size_t i = 0; i <= dim; ++i) {
                oss << " ";
            }
        }

        // update idx
        idx[dim]++;
    }
    // final element (no whitespace/commas)
    print_rec(dim + 1, idx, oss, max_elem_width);
    idx[dim]++; // update

    // close bracket
    oss << "]";
}

size_t Tensor::flat_index(const std::vector<size_t> &idx) const {
    size_t pos = offset_;
    for (u32 i = 0; i < strides_.size(); ++i) {
        size_t id = idx.begin()[i];
        if (id >= shape_[i]) {
            std::ostringstream oss;
            oss << "Dimension: " << i << ". Access index: " << id
                << ". Max index: " << shape_[id];
            throw std::out_of_range(oss.str());
        }
        pos += strides_[i] * idx.begin()[i];
    }
    return pos;
}

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides = std::vector<size_t>(shape.size());
    strides[strides.size() - 1] = 1;
    for (int i = strides.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}
