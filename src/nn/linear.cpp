#include "mini_torch/nn/linear.h"
#include <random>

// This layer will output a matrix with shape (#samples n, number of outputs you want)

linear::linear(std::size_t input_features, std::size_t output_features)
    : weights(tensor({static_cast<int>(input_features), static_cast<int>(output_features)}, true)),
      bias(tensor({static_cast<int>(output_features)}, true))
{
    std::size_t fan_in = input_features;
    float stddev = std::sqrt(2.0/static_cast<float>(fan_in));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    for(std::size_t i = 0; i < weights.size(); ++i){
        weights.data_at(i) = dist(gen);
    }

    for(std::size_t i = 0; i<bias.size(); ++i){
        bias.data_at(i) = 0;
    }

}

tensor &linear::get_weights()
{
    return weights;
}
tensor &linear::get_bias()
{
    return bias;
}

tensor linear::forward(const tensor &x)
{
    return x.matmul(weights) + bias;
}

std::vector<tensor *> linear::parameters()
{
    return {&weights, &bias};
}