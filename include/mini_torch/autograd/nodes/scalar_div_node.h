#pragma once
#include "node.h"

class scalar_div_node : public node
{
public:
    scalar_div_node(const tensor &x_input, float scalar);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    float scalar_;
};