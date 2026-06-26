#pragma once

#include "cnn/layers/layer.hpp"

namespace cnn::layers {

class Conv2D final : public Layer {
public:
    Conv2D(std::size_t in_channels,
           std::size_t out_channels,
           std::size_t kernel_size,
           std::size_t stride = 1,
           std::size_t padding = 0);

    Tensor forward(const Tensor& input) const override;

private:
    std::size_t stride_;
    std::size_t padding_;
    Tensor weights_;
    Tensor bias_;
};

} // namespace cnn::layers
