#include "storage.hpp"
#include <algorithm>
#include <cstddef>

Storage::Storage(size_t size) : size_(size) { data_ = new float[size]; }
Storage::~Storage() { delete[] data_; }

Storage::Storage(const Storage &other)
    : data_(new float[other.size_]), size_(other.size_) { // copy constructor
    std::copy(other.data_, other.data_ + size_, data_);
}
Storage &Storage::operator=(const Storage &other) { // copy assignment
    if (this != &other) {
        size_ = other.size_;
        data_ = new float[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_) { // move constructor

    other.data_ = nullptr;
    other.size_ = 0;
}
Storage &Storage::operator=(Storage &&other) noexcept { // move assignment
    if (this != &other) {
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

size_t       Storage::size() const { return size_; };
float       *Storage::data() { return data_; };
const float *Storage::data() const { return data_; };
