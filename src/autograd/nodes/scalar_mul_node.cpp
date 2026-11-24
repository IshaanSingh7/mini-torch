#include "scalar_mul_node.h"

scalar_mul_node::scalar_mul_node(
    const tensor &x_input, float scalar)
    : x(x_input), scalar_(scalar)
{
}

void scalar_mul_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor x_gradient = upstream_gradient * scalar_;

        x.accumulate_grad(x_gradient);

        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(x_gradient);
        }
    }
}