#pragma once

#include "cnn/layers/layer.hpp"

namespace cnn::layers {

class MaxPool2D final : public Layer {
public:
    MaxPool2D(std::size_t kernel_size, std::size_t stride);
    Tensor forward(const Tensor& input) const override;

private:
    std::size_t kernel_size_;
    std::size_t stride_;
};

} // namespace cnn::layers
