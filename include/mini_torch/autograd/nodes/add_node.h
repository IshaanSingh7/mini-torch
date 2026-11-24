#include "mini_torch/autograd/node.h"

class add_node : public node
{
public:
    add_node(const tensor &x_input, const tensor &y_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    tensor y;
};