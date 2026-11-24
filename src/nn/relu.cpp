#include "mini_torch/nn/relu.h"

tensor relu::forward(const tensor &x){
    return x.relu();
}

std::vector<tensor*> relu::parameters(){
    return {};
}