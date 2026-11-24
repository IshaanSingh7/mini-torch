#pragma once
#include "optimizer.h"

class sgd : public optimizer
{
public:
    sgd(const std::vector<tensor*> &parameters, float learning_rate);

    void step() override;

private:
    float learning_rate;

};