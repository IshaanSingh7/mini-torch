#include "broadcasting.h"
#include "tensor.h"
#include <stdexcept>
#include <chrono>
#include <iostream>

std::vector<int> broadcast_shape(
    const std::vector<int> &x_shape,
    const std::vector<int> &y_shape)
{
    std::size_t rank =
        std::max(x_shape.size(), y_shape.size());

    std::vector<int> result_shape(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        int x_dim = 1;
        int y_dim = 1;

        if (i >= rank - x_shape.size())
        {
            x_dim = x_shape[i - (rank - x_shape.size())];
        }

        if (i >= rank - y_shape.size())
        {
            y_dim = y_shape[i - (rank - y_shape.size())];
        }

        if (x_dim != y_dim && x_dim != 1 && y_dim != 1)
        {
            throw std::invalid_argument(
                "Tensor shapes are not broadcastable");
        }

        result_shape[i] =
            x_dim == 1 ? y_dim : x_dim;
    }

    return result_shape;
}

std::vector<std::size_t> broadcast_strides(
    const std::vector<int> &x_shape,
    const std::vector<int> &result_shape)
{
    std::size_t rank = result_shape.size();

    if (x_shape.size() > rank)
    {
        throw std::invalid_argument(
            "Input rank exceeds result rank");
    }

    std::vector<std::size_t> strides(rank, 0);

    std::size_t stride = 1;
    std::size_t rank_difference =
        rank - x_shape.size();

    for (std::size_t i = x_shape.size(); i-- > 0;)
    {
        std::size_t d = rank_difference + i;

        if (x_shape[i] != result_shape[d] &&
            x_shape[i] != 1)
        {
            throw std::invalid_argument(
                "Incompatible broadcast dimensions");
        }

        if (x_shape[i] != 1)
        {
            strides[d] = stride;
        }

        stride *= static_cast<std::size_t>(x_shape[i]);
    }

    return strides;
}

tensor sum_to_shape(
    const tensor &input,
    const std::vector<int> &target_shape)
{
    if (input.shape() == target_shape)
    {
        return input;
    }

    const std::vector<int> &input_shape = input.shape();

    if (input_shape.size() < target_shape.size())
    {
        throw std::invalid_argument(
            "Cannot sum tensor to a higher-rank shape");
    }

    std::size_t rank = input_shape.size();

    std::size_t rank_difference =
        rank - target_shape.size();

    for (std::size_t i = 0; i < target_shape.size(); ++i)
    {
        int input_dim = input_shape[rank_difference + i];
        int target_dim = target_shape[i];

        if (target_dim != 1 && target_dim != input_dim)
        {
            throw std::invalid_argument(
                "Incompatible reduction dimensions");
        }
    }

    tensor result(target_shape);

    // Calculate contiguous strides for the input.
    std::vector<std::size_t> input_strides(rank);

    std::size_t stride = 1;

    for (std::size_t i = rank; i-- > 0;)
    {
        input_strides[i] = stride;
        stride *= static_cast<std::size_t>(input_shape[i]);
    }

    // Calculate strides for the target, aligned to input rank.
    // Reduced dimensions receive stride zero.
    std::vector<std::size_t> target_strides(rank, 0);

    stride = 1;

    for (std::size_t i = target_shape.size(); i-- > 0;)
    {
        std::size_t d = rank_difference + i;

        if (target_shape[i] != 1)
        {
            target_strides[d] = stride;
        }

        stride *= static_cast<std::size_t>(target_shape[i]);
    }

    // Accumulate input elements into the target.
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        std::size_t remaining = i;
        std::size_t target_offset = 0;

        for (std::size_t d = 0; d < rank; ++d)
        {
            std::size_t coordinate =
                remaining / input_strides[d];

            remaining %= input_strides[d];

            target_offset += coordinate * target_strides[d];
        }

        result.data_at(target_offset) += input.data_at(i);
    }


    return result;
}