#pragma once

#include "cnn/tensor.hpp"

namespace cnn::backend::cpu {

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding);

Tensor dense(const Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor flatten(const Tensor& input);
Tensor max_pool2d(const Tensor& input, std::size_t kernel_size, std::size_t stride);
Tensor relu(const Tensor& input);
Tensor softmax(const Tensor& input);

} // namespace cnn::backend::cpu
