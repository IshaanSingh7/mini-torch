#include "mini_torch/optim/optimizer.h"

optimizer::optimizer(const std::vector<tensor *> &parameters)
    : parameters(parameters)
{
}


void optimizer::zero_grad(){
    for(std::size_t i = 0; i < parameters.size(); ++i){
        parameters[i]->zero_grad();
    }
}

