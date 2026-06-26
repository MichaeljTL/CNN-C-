#pragma once

#include "cnn/layers/layer.hpp"

namespace cnn::layers {

class Flatten final : public Layer {
public:
    Tensor forward(const Tensor& input) const override;
};

} // namespace cnn::layers
