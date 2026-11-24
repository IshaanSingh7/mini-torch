#include "sgd.h"

sgd::sgd(const std::vector<tensor *> &parameters, float lr)
    : optimizer(parameters), learning_rate(lr)
{
}

void sgd::step()
{
    for(tensor *parameter : parameters){
        if(!parameter->requires_grad() || !parameter->has_grad()){
            continue;
        }

        tensor gradient = parameter->grad();

        for(std::size_t i = 0; i< parameter->size(); ++i){
            parameter->data_at(i) -= learning_rate * gradient.data_at(i);
        }
        
    }
}

