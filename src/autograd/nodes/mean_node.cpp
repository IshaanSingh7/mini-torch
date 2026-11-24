
#include "mean_node.h"

mean_node::mean_node(const tensor &x_input)
    : x(x_input)

{
}

void mean_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor result(x.shape());

        float gradient = upstream_gradient.data_at(0) / x.size();

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            result.data_at(i) = gradient;
        }
        x.accumulate_grad(result);
        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(result);
        }
    }
}