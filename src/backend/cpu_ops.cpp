#include "cnn/backend/cpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cnn::backend::cpu {
namespace {

void require_rank(const Tensor& tensor, std::size_t rank, const char* name)
{
    if (tensor.rank() != rank) {
        throw std::invalid_argument(std::string{name} + " has invalid rank");
    }
}

} // namespace

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding)
{
    require_rank(input, 4, "input");
    require_rank(weights, 4, "weights");
    require_rank(bias, 1, "bias");

    const auto& in_shape = input.shape();
    const auto& w_shape = weights.shape();
    const std::size_t batch = in_shape[0];
    const std::size_t in_channels = in_shape[1];
    const std::size_t in_h = in_shape[2];
    const std::size_t in_w = in_shape[3];
    const std::size_t out_channels = w_shape[0];
    const std::size_t kernel_channels = w_shape[1];
    const std::size_t kernel_h = w_shape[2];
    const std::size_t kernel_w = w_shape[3];

    if (in_channels != kernel_channels || bias.shape()[0] != out_channels) {
        throw std::invalid_argument("Conv2D channel dimensions do not match");
    }
    if (stride == 0) {
        throw std::invalid_argument("Conv2D stride must be greater than zero");
    }

    const std::size_t padded_h = in_h + 2 * padding;
    const std::size_t padded_w = in_w + 2 * padding;
    if (padded_h < kernel_h || padded_w < kernel_w) {
        throw std::invalid_argument("Conv2D kernel is larger than padded input");
    }

    const std::size_t out_h = (padded_h - kernel_h) / stride + 1;
    const std::size_t out_w = (padded_w - kernel_w) / stride + 1;
    Tensor output({batch, out_channels, out_h, out_w});

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    float sum = bias.at({oc});

                    for (std::size_t ic = 0; ic < in_channels; ++ic) {
                        for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                                const int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                                const int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                                if (ih < 0 || iw < 0 || ih >= static_cast<int>(in_h) || iw >= static_cast<int>(in_w)) {
                                    continue;
                                }
                                sum += input.at({n, ic, static_cast<std::size_t>(ih), static_cast<std::size_t>(iw)})
                                    * weights.at({oc, ic, kh, kw});
                            }
                        }
                    }

                    output.at({n, oc, oh, ow}) = sum;
                }
            }
        }
    }

    return output;
}

Tensor dense(const Tensor& input, const Tensor& weights, const Tensor& bias)
{
    require_rank(input, 2, "input");
    require_rank(weights, 2, "weights");
    require_rank(bias, 1, "bias");

    const std::size_t batch = input.shape()[0];
    const std::size_t in_features = input.shape()[1];
    const std::size_t out_features = weights.shape()[0];

    if (weights.shape()[1] != in_features || bias.shape()[0] != out_features) {
        throw std::invalid_argument("Dense dimensions do not match");
    }

    Tensor output({batch, out_features});
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t out = 0; out < out_features; ++out) {
            float sum = bias.at({out});
            for (std::size_t in = 0; in < in_features; ++in) {
                sum += input.at({n, in}) * weights.at({out, in});
            }
            output.at({n, out}) = sum;
        }
    }
    return output;
}

Tensor flatten(const Tensor& input)
{
    if (input.rank() < 2) {
        throw std::invalid_argument("Flatten expects at least batch and feature dimensions");
    }

    const std::size_t batch = input.shape()[0];
    const std::size_t features = input.size() / batch;
    Tensor output({batch, features}, input.values());
    return output;
}

Tensor max_pool2d(const Tensor& input, std::size_t kernel_size, std::size_t stride)
{
    require_rank(input, 4, "input");
    if (kernel_size == 0 || stride == 0) {
        throw std::invalid_argument("MaxPool2D kernel and stride must be greater than zero");
    }

    const auto& shape = input.shape();
    const std::size_t batch = shape[0];
    const std::size_t channels = shape[1];
    const std::size_t in_h = shape[2];
    const std::size_t in_w = shape[3];
    if (in_h < kernel_size || in_w < kernel_size) {
        throw std::invalid_argument("MaxPool2D kernel is larger than input");
    }

    const std::size_t out_h = (in_h - kernel_size) / stride + 1;
    const std::size_t out_w = (in_w - kernel_size) / stride + 1;
    Tensor output({batch, channels, out_h, out_w});

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    float best = -std::numeric_limits<float>::infinity();
                    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
                        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
                            best = std::max(best, input.at({n, c, oh * stride + kh, ow * stride + kw}));
                        }
                    }
                    output.at({n, c, oh, ow}) = best;
                }
            }
        }
    }

    return output;
}

Tensor relu(const Tensor& input)
{
    Tensor output(input.shape(), input.values());
    for (float& value : output.values()) {
        value = std::max(0.0f, value);
    }
    return output;
}

Tensor softmax(const Tensor& input)
{
    require_rank(input, 2, "input");

    const std::size_t batch = input.shape()[0];
    const std::size_t classes = input.shape()[1];
    Tensor output(input.shape());

    for (std::size_t n = 0; n < batch; ++n) {
        float max_value = input.at({n, 0});
        for (std::size_t c = 1; c < classes; ++c) {
            max_value = std::max(max_value, input.at({n, c}));
        }

        float denominator = 0.0f;
        for (std::size_t c = 0; c < classes; ++c) {
            const float value = std::exp(input.at({n, c}) - max_value);
            output.at({n, c}) = value;
            denominator += value;
        }

        for (std::size_t c = 0; c < classes; ++c) {
            output.at({n, c}) /= denominator;
        }
    }

    return output;
}

} // namespace cnn::backend::cpu
