#pragma once

#include "cnn/layers/layer.hpp"

namespace cnn::layers {

class Dense final : public Layer {
public:
    Dense(std::size_t in_features, std::size_t out_features);
    Tensor forward(const Tensor& input) const override;

private:
    Tensor weights_;
    Tensor bias_;
};

} // namespace cnn::layers
