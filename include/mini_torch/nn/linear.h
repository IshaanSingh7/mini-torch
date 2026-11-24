#pragma once
#include "mini_torch/core/tensor.h"
#include "mini_torch/nn/module.h"
class linear : public module
{
private:
    tensor weights;
    tensor bias;

public:
    linear(std::size_t input_features, std::size_t output_features);

    tensor forward(const tensor& x) override;
    std::vector<tensor*> parameters() override;
    tensor& get_weights();
    tensor& get_bias();
};