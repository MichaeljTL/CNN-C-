#include "cnn/sequential.hpp"

namespace cnn {

Tensor Sequential::forward(const Tensor& input) const
{
    Tensor current = input;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

} // namespace cnn
