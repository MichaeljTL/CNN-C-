#pragma once

#include "cnn/tensor.hpp"

namespace cnn::layers {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
};

} // namespace cnn::layers
