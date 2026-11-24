#pragma once

#include <vector>
#include "mini_torch/core/tensor.h"

std::vector<int> broadcast_shape(
    const std::vector<int> &x_shape,
    const std::vector<int> &y_shape);

std::vector<std::size_t> broadcast_strides(const std::vector<int> &x_shape, const std::vector<int> &result_shape);

tensor sum_to_shape(
    const tensor &input,
    const std::vector<int> &target_shape);