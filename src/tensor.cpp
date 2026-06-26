#include "cnn/tensor.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cnn {
namespace {

std::size_t product(const std::vector<std::size_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

} // namespace

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(product(shape_), 0.0f)
{
}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data))
{
    if (product(shape_) != data_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape");
    }
}

float& Tensor::at(std::initializer_list<std::size_t> indices)
{
    return data_[offset(indices)];
}

const float& Tensor::at(std::initializer_list<std::size_t> indices) const
{
    return data_[offset(indices)];
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index rank does not match shape rank");
    }

    std::size_t flat_index = 0;
    std::size_t stride = 1;

    auto index_it = indices.end();
    auto shape_it = shape_.end();
    while (index_it != indices.begin()) {
        --index_it;
        --shape_it;
        if (*index_it >= *shape_it) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        flat_index += (*index_it) * stride;
        stride *= *shape_it;
    }

    return flat_index;
}

void Tensor::reshape(std::vector<std::size_t> new_shape)
{
    if (product(new_shape) != data_.size()) {
        throw std::invalid_argument("New tensor shape must preserve element count");
    }
    shape_ = std::move(new_shape);
}

void Tensor::fill(float value)
{
    std::fill(data_.begin(), data_.end(), value);
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
    os << "Tensor(shape=[";
    for (std::size_t i = 0; i < tensor.shape().size(); ++i) {
        os << tensor.shape()[i];
        if (i + 1 < tensor.shape().size()) {
            os << ", ";
        }
    }
    os << "], values=[";

    const std::size_t preview = std::min<std::size_t>(tensor.size(), 12);
    for (std::size_t i = 0; i < preview; ++i) {
        os << tensor[i];
        if (i + 1 < preview) {
            os << ", ";
        }
    }
    if (preview < tensor.size()) {
        os << ", ...";
    }
    os << "])";
    return os;
}

} // namespace cnn
