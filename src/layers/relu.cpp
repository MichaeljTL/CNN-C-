#include "cnn/layers/relu.hpp"

#include "cnn/backend/cpu_ops.hpp"

namespace cnn::layers {

Tensor ReLU::forward(const Tensor& input) const
{
    return backend::cpu::relu(input);
}

} // namespace cnn::layers
