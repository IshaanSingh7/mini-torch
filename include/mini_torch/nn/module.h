#pragma once
#include "mini_torch/core/tensor.h"

class module
{
public:
    virtual ~module() = default;
    virtual tensor forward(const tensor &x) = 0;
    virtual std::vector<tensor *> parameters() = 0;

    virtual void train();
    virtual void eval();
    bool is_training() const;

protected:
    bool training = true;
};