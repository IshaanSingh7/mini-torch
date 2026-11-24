#include "sub_node.h"
#include "broadcasting.h"
sub_node::sub_node(const tensor &x_input, const tensor &y_input)
    : x(x_input), y(y_input)

{
}

void sub_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor x_gradient = sum_to_shape(upstream_gradient, x.shape());
        x.accumulate_grad(x_gradient);

        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }
    if (y.requires_grad())
    {
        tensor y_gradient = sum_to_shape(upstream_gradient, y.shape()) * -1.0f;
        y.accumulate_grad(y_gradient);

        if (y.has_grad_fn())
        {
            y.grad_fn()->backward(y_gradient);
        }
    }
}