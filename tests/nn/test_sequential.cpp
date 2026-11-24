#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mini_torch/nn/linear.h"
#include "mini_torch/nn/relu.h"
#include "mini_torch/nn/sequential.h"
#include "mini_torch/core/tensor.h"

void assert_close(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) < tolerance);
}

void test_sequential()
{
    linear layer1(2, 3);
    relu activation;
    linear layer2(3, 1);

    sequential model;

    model.add(std::make_unique<linear>(layer1));
    model.add(std::make_unique<relu>(activation));
    model.add(std::make_unique<linear>(layer2));

    tensor x({4, 2}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x.data_at(i) = static_cast<float>(i + 1);
    }

    tensor output = model.forward(x);

    // 4 examples, 1 output each
    assert(output.shape() == std::vector<int>({4, 1}));

    // The two linear layers each have weights + bias.
    // ReLU has no parameters.
    std::vector<tensor*> parameters = model.parameters();

    assert(parameters.size() == 4);

    // Parameter shapes
    assert(parameters[0]->shape() == std::vector<int>({2, 3}));
    assert(parameters[1]->shape() == std::vector<int>({3}));
    assert(parameters[2]->shape() == std::vector<int>({3, 1}));
    assert(parameters[3]->shape() == std::vector<int>({1}));

    // Make sure these are actually trainable parameters.
    for (tensor* parameter : parameters)
    {
        assert(parameter != nullptr);
        assert(parameter->requires_grad());
    }

    // Run the entire computation graph backward.
    tensor loss = output.sum();

    loss.backward();

    // Every parameter in the model should have received a gradient.
    for (tensor* parameter : parameters)
    {
        assert(parameter->has_grad());
        assert(parameter->grad().shape() == parameter->shape());
    }

    std::cout << "Sequential test passed!\n";
}

int main(){
    test_sequential();
};