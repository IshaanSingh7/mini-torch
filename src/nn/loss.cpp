#include "mini_torch/nn/loss.h"

// tensor mse_loss(const tensor &predictions, const tensor &targets)
// {

//     tensor difference = predictions - targets;
//     return (difference * difference).mean();
// }

tensor mse_loss(const tensor &predictions, const tensor &targets)
{

    tensor difference = predictions - targets;

    tensor squared = difference * difference;

    tensor result = squared.mean();

    return result;
}