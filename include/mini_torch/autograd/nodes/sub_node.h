#include "node.h"

class sub_node : public node
{
public:
    sub_node(const tensor &x_input, const tensor &y_input);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    tensor y;
};