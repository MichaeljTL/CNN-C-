#include "cnn/layers/flatten.hpp"

#include "cnn/backend/cpu_ops.hpp"

namespace cnn::layers {

Tensor Flatten::forward(const Tensor& input) const
{
    return backend::cpu::flatten(input);
}

} // namespace cnn::layers
