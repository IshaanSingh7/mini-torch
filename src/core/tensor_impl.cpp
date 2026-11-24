#include "tensor_impl.h"

#include <numeric>
#include <functional>
#include <utility>
#include <stdexcept>

tensor_impl::tensor_impl(
    std::vector<int> dims,
    bool requires_grad)
    : shape_(std::move(dims)),
      requires_grad_(requires_grad)
{
    data_.resize(total_elements(), 0.0f);

    strides_.resize(shape_.size());

    std::size_t stride = 1;

    for (std::size_t i = shape_.size(); i-- > 0;)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }

    ++live_count;
}

tensor_impl::~tensor_impl()
{
    --live_count;
}

// Accessors implementations for shape, data, and strides here

const std::vector<int> &tensor_impl::shape() const
{
    return this->shape_;
}

std::vector<float> &tensor_impl::data()
{
    return this->data_;
}
const std::vector<float> &tensor_impl::data() const
{
    return this->data_;
}
const std::vector<int> &tensor_impl::strides() const
{
    return this->strides_;
}

std::size_t tensor_impl::total_elements() const
{
    if (shape_.empty())
    {
        return 0;
    }

    return std::accumulate(
        shape_.begin(),
        shape_.end(),
        1ULL,
        std::multiplies<std::size_t>());
}

std::size_t tensor_impl::size() const
{
    return data_.size();
}

bool tensor_impl::requires_grad() const
{
    return this->requires_grad_;
}

void tensor_impl::set_requires_grad(const bool condition)
{
    requires_grad_ = condition;
}

bool tensor_impl::has_grad() const
{
    return grad_ != nullptr;
}

std::shared_ptr<tensor_impl> tensor_impl::grad() const
{
    return grad_;
}

void tensor_impl::add_grad()
{
    if (!requires_grad_)
    {
        throw std::runtime_error(
            "Cannot create gradient for tensor with requires_grad=false");
    }
    if (grad_ == nullptr)
    {
        grad_ = std::make_shared<tensor_impl>(shape_, false);
    }
}

std::shared_ptr<node> tensor_impl::grad_fn() const
{
    return grad_fn_;
}

void tensor_impl::set_grad_fn(std::shared_ptr<node> node)
{
    grad_fn_ = std::move(node);
}

void tensor_impl::set_grad(std::shared_ptr<tensor_impl> grad)
{
    grad_ = std::move(grad);
}

void tensor_impl::zero_grad()
{
    if (!has_grad())
    {
        return;
    }

    auto &destination = grad_->data();
    for (std::size_t i = 0; i < grad_->size(); ++i)
    {
        destination[i] = 0.0f;
    }
}

std::vector<int> tensor_impl::index_from_flat(
    std::size_t flat_index) const
{
    std::vector<int> index(shape_.size());

    for (std::size_t i = shape_.size(); i-- > 0;)
    {
        index[i] = static_cast<int>(
            flat_index % shape_[i]);

        flat_index /= shape_[i];
    }

    return index;
}

void tensor_impl::accumulate_grad(const std::shared_ptr<tensor_impl> &grad)
{

    if (grad_ == nullptr)
    {
        grad_ = std::make_shared<tensor_impl>(shape(), false);
    }

    auto &destination = grad_->data();
    const auto &source = grad->data();

    for (std::size_t i = 0; i < size(); ++i)
    {
        destination[i] += source[i];
    }
}

bool tensor_impl::has_grad_fn() const
{
    return grad_fn_ != nullptr;
}

std::size_t tensor_impl::compute_offset(const std::vector<int> &index) const
{
    if (index.size() > shape_.size())
    {
        throw std::out_of_range("tensor index out of bounds");
    }

    std::size_t offset = 0;

    for (std::size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] < 0 || index[i] >= shape_[i])
        {
            throw std::out_of_range("tensor index out of bounds");
        }

        offset += index[i] * strides_[i];
    }

    return offset;
}

std::vector<float> tensor_impl::get(const std::vector<int> &index)
{

    std::size_t offset = compute_offset(index);

    std::size_t slice_size = 1;

    for (std::size_t i = index.size(); i < shape_.size(); ++i)
    {
        slice_size *= shape_[i];
    }

    return std::vector<float>(
        data_.begin() + offset,
        data_.begin() + offset + slice_size);
}

// implement set here

void tensor_impl::set(const std::vector<int> &index, float value)
{
    if (index.size() != shape_.size())
    {
        throw std::out_of_range("at() requires a full index");
    }

    data_[compute_offset(index)] = value;
}

float &tensor_impl::at(const std::vector<int> &index)
{
    if (index.size() != shape_.size())
    {
        throw std::out_of_range("at() requires a full index");
    }

    return data_[compute_offset(index)];
}

const float &tensor_impl::at(const std::vector<int> &index) const
{
    if (index.size() != shape_.size())
    {
        throw std::out_of_range("at() requires a full index");
    }

    return data_[compute_offset(index)];
}

std::atomic<std::size_t> tensor_impl::live_count{0};