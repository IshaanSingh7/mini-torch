#include "mul_node.h"
#include "broadcasting.h"

mul_node::mul_node(const tensor &x_input, const tensor &y_input)
    : x(x_input), y(y_input)
{
}

void mul_node::backward(const tensor &upstream_gradient)
{

    if (x.is_same(y) && x.requires_grad())
    {
        tensor gradient =
            sum_to_shape((upstream_gradient * x) * 2.0f, x.shape());

        x.accumulate_grad(gradient);

        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(gradient);
        }

        return;
    }

    if (x.requires_grad())
    {
        tensor x_gradient = sum_to_shape(upstream_gradient * y, x.shape());
        x.accumulate_grad(x_gradient);
        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }
    if (y.requires_grad())
    {
        tensor y_gradient = sum_to_shape(upstream_gradient * x, y.shape());

        y.accumulate_grad(y_gradient);

        if (y.has_grad_fn())
        {
            y.grad_fn()->backward(y_gradient);
        }
    }
}