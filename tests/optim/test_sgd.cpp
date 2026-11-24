
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "tensor.h"
#include "sgd.h"

bool is_close(float actual, float expected, float tolerance = 1e-5f)
{
    return std::abs(actual - expected) < tolerance;
}

void test_sgd_step()
{
    // Create two trainable parameters.
    tensor weights({2}, true);
    tensor bias({1}, true);

    weights.data_at(0) = 2.0f;
    weights.data_at(1) = -3.0f;
    bias.data_at(0) = 1.0f;

    // Manually assign gradients so we can test SGD independently
    // of autograd.
    tensor weights_grad({2});
    weights_grad.data_at(0) = 0.5f;
    weights_grad.data_at(1) = -1.0f;

    tensor bias_grad({1});
    bias_grad.data_at(0) = 2.0f;

    weights.set_grad(weights_grad);
    bias.set_grad(bias_grad);

    std::vector<tensor*> parameters = {&weights, &bias};
    const float learning_rate = 0.1f;

    sgd optimizer(parameters, learning_rate);

    optimizer.step();

    // SGD: parameter = parameter - learning_rate * gradient
    assert(is_close(weights.data_at(0), 1.95f));
    assert(is_close(weights.data_at(1), -2.9f));
    assert(is_close(bias.data_at(0), 0.8f));

    std::cout << "SGD step test passed!\n";
}

void test_sgd_zero_grad()
{
    tensor weights({2}, true);

    tensor gradient({2});
    gradient.data_at(0) = 0.5f;
    gradient.data_at(1) = -1.0f;
    weights.set_grad(gradient);

    std::vector<tensor*> parameters = {&weights};
    sgd optimizer(parameters, 0.1f);

    optimizer.zero_grad();

    assert(weights.has_grad());

    tensor temporary = weights.grad();

    for(std::size_t i = 0; i < temporary.size();++i){
        assert(is_close(temporary.data_at(i), 0.0f));
    }

    std::cout << "SGD zero_grad test passed!\n";
}

int main()
{
    test_sgd_step();
    test_sgd_zero_grad();

    std::cout << "All SGD tests passed!\n";
    return 0;
}
