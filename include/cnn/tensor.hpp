#pragma once

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace cnn {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, std::vector<float> data);

    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    std::vector<float>& values() { return data_; }
    const std::vector<float>& values() const { return data_; }

    float& operator[](std::size_t index) { return data_[index]; }
    const float& operator[](std::size_t index) const { return data_[index]; }

    float& at(std::initializer_list<std::size_t> indices);
    const float& at(std::initializer_list<std::size_t> indices) const;

    std::size_t offset(std::initializer_list<std::size_t> indices) const;
    void reshape(std::vector<std::size_t> new_shape);
    void fill(float value);

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace cnn
