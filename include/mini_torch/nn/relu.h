#pragma once

#include "mini_torch/nn/module.h"

class relu : public module
{
public:
    tensor forward(const tensor &x) override;

    std::vector<tensor*> parameters() override;
};