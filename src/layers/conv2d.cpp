#include "cnn/layers/conv2d.hpp"

#include "cnn/backend/cpu_ops.hpp"

#include <cmath>

namespace cnn::layers {

Conv2D::Conv2D(std::size_t in_channels,
               std::size_t out_channels,
               std::size_t kernel_size,
               std::size_t stride,
               std::size_t padding)
    : stride_(stride),
      padding_(padding),
      weights_({out_channels, in_channels, kernel_size, kernel_size}),
      bias_({out_channels})
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel_size * kernel_size));
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * scale * 0.1f;
    }
    bias_.fill(0.0f);
}

Tensor Conv2D::forward(const Tensor& input) const
{
    return backend::cpu::conv2d(input, weights_, bias_, stride_, padding_);
}

} // namespace cnn::layers
