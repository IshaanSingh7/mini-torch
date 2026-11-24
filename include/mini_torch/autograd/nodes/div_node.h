#include "node.h"

class div_node : public node
{
public:
    div_node(const tensor &x_input, const tensor &y_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    tensor y;
};