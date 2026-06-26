#include "cnn/layers/softmax.hpp"

#include "cnn/backend/cpu_ops.hpp"

namespace cnn::layers {

Tensor Softmax::forward(const Tensor& input) const
{
    return backend::cpu::softmax(input);
}

} // namespace cnn::layers
