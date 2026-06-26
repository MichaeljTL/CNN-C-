#include "cnn/sequential.hpp"
#include "cnn/layers/conv2d.hpp"
#include "cnn/layers/dense.hpp"
#include "cnn/layers/flatten.hpp"
#include "cnn/layers/max_pool2d.hpp"
#include "cnn/layers/relu.hpp"
#include "cnn/layers/softmax.hpp"

#include <iostream>

int main()
{
    using namespace cnn;
    using namespace cnn::layers;

    Tensor image({1, 1, 8, 8});
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>(i % 16) / 15.0f;
    }

    Sequential model;
    model
        .add<Conv2D>(1, 2, 3, 1, 1)
        .add<ReLU>()
        .add<MaxPool2D>(2, 2)
        .add<Flatten>()
        .add<Dense>(2 * 4 * 4, 3)
        .add<Softmax>();

    const Tensor probabilities = model.forward(image);
    std::cout << probabilities << '\n';

    return 0;
}
