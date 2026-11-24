#include "relu_node.h"

relu_node::relu_node(const tensor &x_input)
    : x(x_input)

{
}

void relu_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor result(x.shape());

        for (std::size_t i = 0; i < upstream_gradient.size(); ++i)
        {
            if (x.data_at(i) > 0)
            {
                result.data_at(i) = upstream_gradient.data_at(i);
            }
            else
            {
                result.data_at(i) = 0.0f;
            }
        }
        x.accumulate_grad(result);
        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(result);
        }
    }
}