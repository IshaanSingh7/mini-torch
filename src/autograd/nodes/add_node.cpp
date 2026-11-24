#include "mini_torch/autograd/nodes/add_node.h"
#include "mini_torch/core/broadcasting.h"

add_node::add_node(const tensor &x_input, const tensor &y_input)
    : x(x_input), y(y_input)

{
}

void add_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor x_gradient =
            sum_to_shape(upstream_gradient, x.shape());

        x.accumulate_grad(x_gradient);

        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }

    if (y.requires_grad())
    {
        tensor y_gradient =
            sum_to_shape(upstream_gradient, y.shape());

        y.accumulate_grad(y_gradient);

        if (y.has_grad_fn())
        {
            y.grad_fn()->backward(y_gradient);
        }
    }
}