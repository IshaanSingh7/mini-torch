#pragma once

#include <vector>
#include <cstddef>
#include <memory>

class tensor_impl;
class node;
class tensor
{
public:
    tensor();
    explicit tensor(
        std::vector<int> dimensions,
        bool requires_grad = false);

    std::size_t total_elements() const;
    std::size_t size() const;
    bool requires_grad() const;
    const std::vector<int> &shape() const;

    std::vector<float> get(const std::vector<int> &index);
    void set(const std::vector<int> &index, float value);
    float &at(const std::vector<int> &index);
    const float &at(const std::vector<int> &index) const;

    tensor operator+(const tensor &other) const;
    tensor operator-(const tensor &other) const;
    tensor operator*(const tensor &other) const;
    tensor operator/(const tensor &other) const;
    tensor operator*(float scalar) const;
    tensor operator/(float scalar) const;

    tensor sum() const;
    tensor mean() const;
    tensor matmul(const tensor &other) const;
    tensor relu() const;
    tensor transpose() const;

    tensor grad() const;
    bool has_grad() const;
    bool has_grad_fn() const;
    std::shared_ptr<node> grad_fn() const;

    void backward();

    float &data_at(std::size_t index);
    const float &data_at(std::size_t index) const;
    void set_grad(const tensor &grad);
    void accumulate_grad(const tensor &grad);
    void zero_grad();
    std::vector<int> index_from_flat(std::size_t index) const;
    tensor select_rows(const std::vector<size_t> &indices);
    bool is_same(const tensor &other) const;

private:
    explicit tensor(std::shared_ptr<tensor_impl> impl);
    std::shared_ptr<tensor_impl> impl;
};

tensor one_hot(const tensor &labels, int num_classes);