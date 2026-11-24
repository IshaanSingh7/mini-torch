#include "sum_node.h"

sum_node::sum_node(const tensor &x_input)
    : x(x_input)

{
}

void sum_node::backward(const tensor &upstream_gradient)
{
    if (x.requires_grad())
    {
        tensor result(x.shape());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            result.data_at(i) = upstream_gradient.data_at(0);
        }
        x.accumulate_grad(result);
        if (x.has_grad_fn())
        {
            x.grad_fn()->backward(result);
        }
    }
}