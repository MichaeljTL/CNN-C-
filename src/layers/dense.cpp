#include "cnn/layers/dense.hpp"

#include "cnn/backend/cpu_ops.hpp"

#include <cmath>

namespace cnn::layers {

Dense::Dense(std::size_t in_features, std::size_t out_features)
    : weights_({out_features, in_features}), bias_({out_features})
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(in_features));
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * scale * 0.05f;
    }
    bias_.fill(0.0f);
}

Tensor Dense::forward(const Tensor& input) const
{
    return backend::cpu::dense(input, weights_, bias_);
}

} // namespace cnn::layers
