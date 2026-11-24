#include "mini_torch/autograd/node.h"

class relu_node : public node
{
public:
    relu_node(const tensor &x_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
};