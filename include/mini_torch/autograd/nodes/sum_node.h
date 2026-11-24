#include "mini_torch/autograd/node.h"

class sum_node : public node
{
public:
    sum_node(const tensor &x_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
};