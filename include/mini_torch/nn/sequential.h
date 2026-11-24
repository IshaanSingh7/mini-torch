#pragma once
#include <vector>
#include "mini_torch/nn/module.h"

class sequential : public module
{
private:
    std::vector<std::unique_ptr<module>> modules;

public:
    sequential();
    
    void add(std::unique_ptr<module> module);

    tensor forward(const tensor &x) override;
    std::vector<tensor*> parameters() override;

    void train() override;
    void eval() override;
};