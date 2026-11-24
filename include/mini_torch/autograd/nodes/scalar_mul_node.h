#pragma once
#include "node.h"

class scalar_mul_node : public node
{
public:
    scalar_mul_node(const tensor &x_input, float scalar);

    void backward(const tensor &upstream_gradient) override;

private:
    tensor x;
    float scalar_;
};