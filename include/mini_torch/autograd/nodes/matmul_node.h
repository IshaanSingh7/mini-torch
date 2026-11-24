#include "mini_torch/autograd/node.h"

class matmul_node : public node
{
public:
    matmul_node(const tensor &x_input, const tensor &y_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    tensor y;
};