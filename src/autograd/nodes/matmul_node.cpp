#include "matmul_node.h"
#include <chrono>
#include <iostream>

matmul_node::matmul_node(const tensor &x_input, const tensor &y_input)
    : x(x_input), y(y_input)
{
}

void matmul_node::backward(const tensor &upstream_gradient)
{

    if (x.requires_grad())
    {

        tensor x_gradient = upstream_gradient.matmul(y.transpose());
        x.accumulate_grad(x_gradient);
        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }

    if (y.requires_grad())
    {
        tensor y_gradient = x.transpose().matmul(upstream_gradient);

        y.accumulate_grad(y_gradient);

        if (y.has_grad_fn())
        {
            y.grad_fn()->backward(y_gradient);
        }
    }
}