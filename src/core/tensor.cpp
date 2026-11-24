#include "tensor.h"
#include "tensor_impl.h"
#include <utility>
#include <iostream>
#include <stdexcept>
#include "add_node.h"
#include "mul_node.h"
#include "div_node.h"
#include "sub_node.h"
#include "relu_node.h"
#include "matmul_node.h"
#include "sum_node.h"
#include "mean_node.h"
#include "broadcasting.h"
#include "grad_mode.h"
#include "scalar_mul_node.h"
#include "scalar_div_node.h"
tensor::tensor()
    : impl(nullptr)
{
}

tensor::tensor(
    std::vector<int> dims,
    bool requires_grad)
    : impl(std::make_shared<tensor_impl>(
          std::move(dims),
          requires_grad))
{
}

// for grad
tensor::tensor(std::shared_ptr<tensor_impl> impl)
    : impl(std::move(impl))
{
}
template <typename BinaryOp>
tensor broadcast_binary_op(
    const tensor &x,
    const tensor &y,
    BinaryOp op)
{
    const std::vector<int> &x_shape = x.shape();
    const std::vector<int> &y_shape = y.shape();

    std::vector<int> result_shape =
        broadcast_shape(x_shape, y_shape);

    tensor result(result_shape);

    if (x_shape == y_shape)
    {
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result.data_at(i) =
                op(x.data_at(i), y.data_at(i));
        }

        return result;
    }

    std::size_t rank = result_shape.size();

    std::vector<std::size_t> x_strides =
        broadcast_strides(x_shape, result_shape);

    std::vector<std::size_t> y_strides =
        broadcast_strides(y_shape, result_shape);

    std::vector<std::size_t> result_strides(rank);

    std::size_t stride = 1;

    for (std::size_t d = rank; d-- > 0;)
    {
        result_strides[d] = stride;
        stride *= static_cast<std::size_t>(result_shape[d]);
    }

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::size_t remaining = i;
        std::size_t x_offset = 0;
        std::size_t y_offset = 0;

        for (std::size_t d = 0; d < rank; ++d)
        {
            std::size_t coordinate =
                remaining / result_strides[d];

            remaining %= result_strides[d];

            x_offset += coordinate * x_strides[d];
            y_offset += coordinate * y_strides[d];
        }

        result.data_at(i) =
            op(x.data_at(x_offset), y.data_at(y_offset));
    }

    return result;
}

tensor tensor::grad() const
{
    return tensor(impl->grad_);
}

bool tensor::has_grad() const
{
    return impl->has_grad();
}

size_t tensor::total_elements() const
{
    return impl->total_elements();
}

size_t tensor::size() const
{
    return impl->size();
}

bool tensor::requires_grad() const
{
    return impl->requires_grad();
}

bool tensor::has_grad_fn() const
{
    return impl->has_grad_fn();
}

std::shared_ptr<node> tensor::grad_fn() const
{
    return impl->grad_fn();
}

float &tensor::data_at(std::size_t index)
{
    return impl->data()[index];
}

const float &tensor::data_at(std::size_t index) const
{
    return impl->data()[index];
}

const std::vector<int> &tensor::shape() const
{
    return impl->shape();
}

void tensor::backward()
{
    if (!impl->grad_fn())
    {
        return;
    }

    no_grad_guard guard;

    tensor upstream(impl->shape());

    for (std::size_t i = 0; i < impl->size(); ++i)
    {
        upstream.data_at(i) = 1.0f;
    }

    impl->grad_fn()->backward(upstream);
}

void tensor::set_grad(const tensor &grad)
{
    impl->set_grad(grad.impl);
}

std::vector<float> tensor::get(const std::vector<int> &index)
{
    return impl->get(index);
}

void tensor::set(const std::vector<int> &index, float value)
{
    impl->set(index, value);
}

float &tensor::at(const std::vector<int> &index)
{
    return impl->at(index);
}

const float &tensor::at(const std::vector<int> &index) const
{
    return impl->at(index);
}

tensor tensor::operator+(const tensor &other) const
{
    tensor result = broadcast_binary_op(
        *this, other,
        [](float a, float b)
        { return a + b; });

    bool track_grad =
        grad_enabled && (requires_grad() || other.requires_grad());

    if (track_grad)
    {
        result.impl->set_requires_grad(true);
        result.impl->set_grad_fn(
            std::make_shared<add_node>(*this, other));
    }

    return result;
}

tensor tensor::operator-(const tensor &other) const
{
    tensor result = broadcast_binary_op(
        *this, other,
        [](float a, float b)
        { return a - b; });

    bool track_grad =
        grad_enabled && (requires_grad() || other.requires_grad());

    if (track_grad)
    {
        result.impl->set_requires_grad(true);
        result.impl->set_grad_fn(
            std::make_shared<sub_node>(*this, other));
    }

    return result;
}

tensor tensor::operator*(const tensor &other) const
{
    tensor result = broadcast_binary_op(
        *this, other,
        [](float a, float b)
        { return a * b; });

    bool track_grad =
        grad_enabled && (requires_grad() || other.requires_grad());

    if (track_grad)
    {
        result.impl->set_requires_grad(true);
        result.impl->set_grad_fn(
            std::make_shared<mul_node>(*this, other));
    }

    return result;
}

tensor tensor::operator/(const tensor &other) const
{
    tensor result = broadcast_binary_op(
        *this, other,
        [](float a, float b)
        { return a / b; });

    bool track_grad =
        grad_enabled && (requires_grad() || other.requires_grad());

    if (track_grad)
    {
        result.impl->set_requires_grad(true);
        result.impl->set_grad_fn(
            std::make_shared<div_node>(*this, other));
    }

    return result;
}

tensor tensor::operator*(float scalar) const
{
    tensor result(impl->shape(), (grad_enabled && requires_grad()));

    auto &destination = result.impl->data();
    auto &source = impl->data();

    for (std::size_t i = 0; i < impl->size(); ++i)
    {
        destination[i] = source[i] * scalar;
    }

    if (grad_enabled && requires_grad())
    {
        auto node = std::make_shared<scalar_mul_node>(*this, scalar);
        result.impl->set_grad_fn(node);
    }
    return result;
}

tensor tensor::operator/(float scalar) const
{
    if (scalar == 0.0f)
        throw std::invalid_argument("Division by zero");

    tensor result(impl->shape(), (grad_enabled && requires_grad()));

    auto &destination = result.impl->data();
    const auto &source = impl->data();

    for (std::size_t i = 0; i < impl->size(); ++i)
    {
        destination[i] = source[i] / scalar;
    }

    if (grad_enabled && requires_grad())
    {
        auto node = std::make_shared<scalar_div_node>(*this, scalar);
        result.impl->set_grad_fn(node);
    }

    return result;
}

std::vector<int> tensor::index_from_flat(
    std::size_t index) const
{
    return impl->index_from_flat(index);
}

tensor tensor::sum() const
{
    if (size() == 0)
        throw std::runtime_error("Cannot calculate mean of empty tensor");
    float total = 0.0f;

    const auto &source = impl->data();
    std::size_t size = source.size();

    for (std::size_t i = 0; i < size; ++i)
    {
        total += source[i];
    }
    tensor sum = tensor({1}, requires_grad());
    sum.data_at(0) = total;

    if (grad_enabled && requires_grad())
    {
        auto node = std::make_shared<sum_node>(*this);
        sum.impl->set_grad_fn(node);
    }
    return sum;
}

tensor tensor::mean() const
{
    const auto &source = impl->data();
    std::size_t size = source.size();
    float total = 0.0f;

    for (std::size_t i = 0; i < size; ++i)
    {
        total += source[i];
    }
    tensor mean = tensor({1}, requires_grad());
    mean.data_at(0) = total / size;

    if (grad_enabled && requires_grad())
    {
        auto node = std::make_shared<mean_node>(*this);
        mean.impl->set_grad_fn(node);
    }
    return mean;
}

tensor tensor::matmul(const tensor &other) const
{
    const std::vector<int> &x_shape = shape();
    const std::vector<int> &y_shape = other.shape();

    std::size_t x_dims = x_shape.size();
    std::size_t y_dims = y_shape.size();

    if (x_dims < 2 || y_dims < 2)
    {
        throw std::invalid_argument(
            "matmul requires tensors with at least 2 dimensions");
    }

    if (x_dims != y_dims)
    {
        throw std::invalid_argument(
            "matmul requires matching tensor ranks");
    }

    std::size_t rank = x_dims;

    int m = x_shape[rank - 2];
    int k = x_shape[rank - 1];
    int n = y_shape[rank - 1];

    if (k != y_shape[rank - 2])
    {
        throw std::invalid_argument(
            "Incompatible matrix dimensions");
    }

    // Validate matching batch dimensions.
    std::size_t batches = 1;

    for (std::size_t i = 0; i < rank - 2; ++i)
    {
        if (x_shape[i] != y_shape[i])
        {
            throw std::invalid_argument(
                "Incompatible batch dimensions");
        }

        batches *= static_cast<std::size_t>(x_shape[i]);
    }

    std::vector<int> result_shape = x_shape;
    result_shape[rank - 2] = m;
    result_shape[rank - 1] = n;

    bool track_grad =
        grad_enabled && (requires_grad() || other.requires_grad());

    tensor result(result_shape, track_grad);

    const auto &x_source = impl->data();
    const auto &y_source = other.impl->data();
    auto &destination = result.impl->data();

    // Number of elements in each matrix.
    std::size_t x_matrix_size =
        static_cast<std::size_t>(m) * k;

    std::size_t y_matrix_size =
        static_cast<std::size_t>(k) * n;

    std::size_t result_matrix_size =
        static_cast<std::size_t>(m) * n;

    for (std::size_t b = 0; b < batches; ++b)
    {
        std::size_t x_offset = b * x_matrix_size;
        std::size_t y_offset = b * y_matrix_size;
        std::size_t result_offset = b * result_matrix_size;

// In the reordered algorithm (ikj order), you compute the matrix multiplication as a 
// sum of outer products or partial contributions:Pick a row $i$ of Matrix $A$ and a column position $p$.
// Grab the single scalar value $x = A[i][p]$.Multiply that 
// scalar $x$ across the entire $p$-th row of Matrix $B$ ($B[p][0 \dots n-1]$) and 
// add those values directly into the corresponding positions in row $i$ of the result matrix.

        for (int i = 0; i < m; ++i)
        {
            // printf("Here");
            // std::fflush(stdout);
            for (int p = 0; p < k; ++p)
            {
                std::size_t x_index =
                    x_offset + static_cast<std::size_t>(i) * k + p;

                float x_value = x_source[x_index];

                for (int j = 0; j < n; ++j)
                {
                    std::size_t y_index =
                        y_offset + static_cast<std::size_t>(p) * n + j;

                    std::size_t result_index =
                        result_offset + static_cast<std::size_t>(i) * n + j;

                    destination[result_index] +=
                        x_value * y_source[y_index];
                }
            }
        }
    }

    if (track_grad)
    {
        auto node =
            std::make_shared<matmul_node>(*this, other);

        result.impl->set_grad_fn(node);
    }

    return result;
}

tensor tensor::relu() const
{
    bool track_grad = grad_enabled && requires_grad();
    tensor result(impl->shape(), track_grad);

    const auto &source = impl->data();
    auto &destination = result.impl->data();


    for (std::size_t i = 0; i < impl->size(); i++)
    {
        if (impl->data()[i] > 0)
        {
            destination[i] = source[i];
        }
        else
        {
            destination[i] = 0;
        }
    }

    if (track_grad)
    {
        result.impl->set_grad_fn(std::make_shared<relu_node>(*this));
    }

    return result;
}

void tensor::accumulate_grad(const tensor &grad)
{
    impl->accumulate_grad(grad.impl);
}

void tensor::zero_grad()
{
    impl->zero_grad();
}

tensor tensor::transpose() const
{
    const std::vector<int> &x_shape = shape();

    if (x_shape.size() < 2)
    {
        throw std::invalid_argument(
            "Tensor must have at least 2 dimensions!");
    }

    std::vector<int> result_shape = x_shape;

    std::size_t rank = x_shape.size();

    std::swap(
        result_shape[rank - 1],
        result_shape[rank - 2]);

    tensor result(result_shape);

    const auto &source = impl->data();
    auto &destination = result.impl->data();

    std::size_t rows = x_shape[rank - 2];
    std::size_t cols = x_shape[rank - 1];

    std::size_t num_batches = 1;

    for (std::size_t i = 0; i < rank - 2; ++i)
    {
        num_batches *= x_shape[i];
    }

    std::size_t matrix_size = rows * cols;

    for (std::size_t batch = 0; batch < num_batches; ++batch)
    {
        std::size_t offset = batch * matrix_size;

        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                destination[offset + j * rows + i] =
                    source[offset + i * cols + j];
            }
        }
    }

    return result;
}

tensor one_hot(const tensor &labels, int num_classes)
{
    tensor encoded({static_cast<int>(labels.size()), num_classes});
    // const auto &source = labels.impl->data();
    // auto &destination = encoded.impl->data();

    for (std::size_t row = 0; row < labels.size(); ++row)
    {
        int label = static_cast<int>(labels.data_at(row));
        if (label < 0 || label > num_classes)
        {
            throw std::range_error("Label is <0 or >num_classes");
        }
        encoded.data_at(row * num_classes + label) = 1.0f;
    }

    return encoded;
}

tensor tensor::select_rows(const std::vector<size_t> &indices){
    int row_size = 1;
    for(std::size_t i = shape().size()-1; i>0; --i){
        row_size*=shape()[i];
    }

    std::vector<int> output_shape = shape();
    output_shape[0] = indices.size();
    tensor result(output_shape, false);

    const auto &source = impl->data();
    auto &destination = result.impl->data();

    for(std::size_t i = 0; i<indices.size(); ++i){
        for(std::size_t j = 0; j<row_size;++j){
            destination[i*row_size + j] = source[indices[i]*row_size + j];
        }
    }

    return result;
}

bool tensor::is_same(const tensor &other) const
{
    return impl == other.impl;
}