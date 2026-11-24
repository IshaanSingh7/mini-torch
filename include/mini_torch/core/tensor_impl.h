#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <atomic>

class node;
class tensor;
class tensor_impl
{
    friend class tensor;

public:
    explicit tensor_impl(
        std::vector<int> dims,
        bool requires_grad = false);

    static std::atomic<std::size_t> live_count;
    ~tensor_impl();

    std::size_t total_elements() const;
    std::size_t size() const;
    bool requires_grad() const;
    std::vector<float> get(const std::vector<int> &index);
    void set(const std::vector<int> &index, float value);
    float &at(const std::vector<int> &index);
    const float &at(const std::vector<int> &index) const;

    const std::vector<int> &shape() const;
    std::vector<float> &data();
    const std::vector<float> &data() const;
    const std::vector<int> &strides() const;
    std::shared_ptr<tensor_impl> grad() const;
    bool has_grad() const;
    bool has_grad_fn() const;

    std::shared_ptr<node> grad_fn() const;
    void set_grad_fn(std::shared_ptr<node> node);
    void set_grad(std::shared_ptr<tensor_impl> grad);
    void accumulate_grad(const std::shared_ptr<tensor_impl> &grad);
    void zero_grad();
    void set_requires_grad(const bool condition);
    std::vector<int> index_from_flat(std::size_t flat_index) const;

private:
    std::vector<float> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    bool requires_grad_;

    std::size_t compute_offset(const std::vector<int> &index) const;

    std::shared_ptr<tensor_impl> grad_;
    std::shared_ptr<node> grad_fn_;
    void add_grad();
};