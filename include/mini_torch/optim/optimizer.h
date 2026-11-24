#include "mini_torch/core/tensor.h"

class optimizer
{
public:
    optimizer(const std::vector<tensor*> &parameters);

    virtual void step() = 0;
    void zero_grad();

protected:
    std::vector<tensor*> parameters;


};