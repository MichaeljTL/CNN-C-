#include "cnn/layers/max_pool2d.hpp"

#include "cnn/backend/cpu_ops.hpp"

namespace cnn::layers {

MaxPool2D::MaxPool2D(std::size_t kernel_size, std::size_t stride)
    : kernel_size_(kernel_size), stride_(stride)
{
}

Tensor MaxPool2D::forward(const Tensor& input) const
{
    return backend::cpu::max_pool2d(input, kernel_size_, stride_);
}

} // namespace cnn::layers
