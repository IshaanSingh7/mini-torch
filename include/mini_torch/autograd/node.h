
#pragma once

#include "mini_torch/core/tensor.h"
#include <atomic>
#include <cstddef>

class node
{
public:
    inline static std::atomic<std::size_t> live_count{0};

    node()
    {
        ++live_count;
    }

    virtual ~node()
    {
        --live_count;
    }

    virtual void backward(const tensor &upstream_gradient) = 0;
};
