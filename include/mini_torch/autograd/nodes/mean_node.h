#include "mini_torch/autograd/node.h"

class mean_node : public node
{
public:
    mean_node(const tensor &x_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
};