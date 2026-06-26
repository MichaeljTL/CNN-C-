#pragma once

#include "cnn/layers/layer.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace cnn {

class Sequential {
public:
    template <typename LayerType, typename... Args>
    Sequential& add(Args&&... args)
    {
        layers_.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
        return *this;
    }

    Tensor forward(const Tensor& input) const;

private:
    std::vector<std::unique_ptr<layers::Layer>> layers_;
};

} // namespace cnn
