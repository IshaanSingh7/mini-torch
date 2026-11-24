#include <vector>
#include "sequential.h"
#include "tensor.h"

sequential::sequential(){

}
void sequential::add(std::unique_ptr<module> module)
{
    modules.push_back(std::move(module));
}

std::vector<tensor *> sequential::parameters()
{
    std::vector<tensor *> parameters;
    for (std::size_t i = 0; i < modules.size(); ++i)
    {
        std::vector<tensor*> module_params = modules[i]->parameters();

        for(std::size_t j = 0; j < module_params.size(); ++j){ // module_params is almost always 2 (weights + bias)
            parameters.push_back(module_params[j]);
        }
    }
    return parameters;
}

tensor sequential::forward(const tensor &x)
{
    tensor predictions = modules[0]->forward(x);
    for (std::size_t i = 1; i < modules.size(); ++i)
    {
        predictions = modules[i]->forward(predictions);
    }

    return predictions;
}

void sequential::train(){
    training = true;
}

void sequential::eval(){
    training = false;
}