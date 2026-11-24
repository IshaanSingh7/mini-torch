#include "scalar_div_node.h"

scalar_div_node::scalar_div_node(
    const tensor &x_input, float scalar)
    : x(x_input), scalar_(scalar)
{
}

void scalar_div_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor x_gradient = upstream_gradient / scalar_;

        x.accumulate_grad(x_gradient);

        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }
}