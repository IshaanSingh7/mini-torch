#include "mini_torch/autograd/node.h"

class mul_node : public node
{
public:
    mul_node(const tensor &x_input, const tensor &y_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    tensor y;
};