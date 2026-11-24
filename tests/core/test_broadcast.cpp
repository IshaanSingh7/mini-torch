
#include "tensor.h"
#include "linear.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void assert_close(float actual, float expected, float tolerance = 1e-5f)
{
    assert(std::fabs(actual - expected) < tolerance);
}

void assert_shape(
    const tensor& x,
    const std::vector<int>& expected)
{
    assert(x.shape() == expected);
}

void assert_all_close(
    const tensor& x,
    float expected,
    float tolerance = 1e-5f)
{
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        assert_close(x.data_at(i), expected, tolerance);
    }
}

/*
 * x: {2,3}
 * y: {3}
 *
 *        x              y
 * [1 2 3]              [10 20 30]
 * [4 5 6]       +      [10 20 30]
 *
 * Result:
 * [11 22 33]
 * [14 25 36]
 *
 * loss = sum(result)
 *
 * dx = [1 1 1
 *       1 1 1]
 *
 * dy = [2 2 2]
 */
void test_add_broadcast_gradient_lower_rank()
{
    tensor x({2, 3}, true);
    tensor y({3}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;

    tensor z = x + y;
    tensor loss = z.sum();

    loss.backward();

    assert_shape(z, {2, 3});
    assert_all_close(x.grad(), 1.0f);
    assert_all_close(y.grad(), 2.0f);
}

/*
 * x: {2,3}
 * y: {1,3}
 *
 * y is broadcast across dimension 0.
 *
 * Therefore every y element is used twice.
 */
void test_add_broadcast_gradient_singleton_dimension()
{
    tensor x({2, 3}, true);
    tensor y({1, 3}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = static_cast<float>(10 + i);

    tensor loss = (x + y).sum();

    loss.backward();

    assert_shape(y.grad(), {1, 3});

    assert_all_close(x.grad(), 1.0f);
    assert_all_close(y.grad(), 2.0f);
}

/*
 * x: {2,1,3}
 * y: {2,4,3}
 *
 * x is broadcast along dimension 1.
 *
 * Each x element is used 4 times.
 *
 * Therefore:
 *
 * dx = every element = 4
 * dy = every element = 1
 */
void test_add_broadcast_gradient_middle_dimension()
{
    tensor x({2, 1, 3}, true);
    tensor y({2, 4, 3}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = static_cast<float>(i + 1);

    tensor loss = (x + y).sum();

    loss.backward();

    assert_shape(x.grad(), {2, 1, 3});
    assert_shape(y.grad(), {2, 4, 3});

    assert_all_close(x.grad(), 4.0f);
    assert_all_close(y.grad(), 1.0f);
}

/*
 * x: {2,1,3}
 * y: {1,4,1}
 *
 * Result: {2,4,3}
 *
 * x is broadcast 4 times.
 * y is broadcast 2 * 3 = 6 times.
 */
void test_add_broadcast_gradient_multiple_dimensions()
{
    tensor x({2, 1, 3}, true);
    tensor y({1, 4, 1}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = static_cast<float>(i + 1);

    tensor z = x + y;
    tensor loss = z.sum();

    loss.backward();

    assert_shape(z, {2, 4, 3});

    // x is repeated across the dimension of size 4.
    assert_all_close(x.grad(), 4.0f);

    // y is repeated across dimensions 2 and 3.
    assert_all_close(y.grad(), 6.0f);
}

/*
 * Subtraction:
 *
 * loss = sum(x - y)
 *
 * dx = +1
 * dy = -number_of_broadcast_repetitions
 */
void test_sub_broadcast_gradient()
{
    tensor x({2, 3}, true);
    tensor y({3}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = static_cast<float>(i + 1);

    tensor loss = (x - y).sum();

    loss.backward();

    assert_all_close(x.grad(), 1.0f);
    assert_all_close(y.grad(), -2.0f);
}

/*
 * Multiplication:
 *
 * x: {2,3}
 * y: {3}
 *
 * loss = sum(x * y)
 *
 * dx = y, broadcast to {2,3}
 * dy = sum over dimension 0 of x
 */
void test_mul_broadcast_gradient()
{
    tensor x({2, 3}, true);
    tensor y({3}, true);

    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;
    x.data_at(4) = 5.0f;
    x.data_at(5) = 6.0f;

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;

    tensor loss = (x * y).sum();

    loss.backward();

    // dx = y repeated across dimension 0
    assert_close(x.grad().data_at(0), 10.0f);
    assert_close(x.grad().data_at(1), 20.0f);
    assert_close(x.grad().data_at(2), 30.0f);
    assert_close(x.grad().data_at(3), 10.0f);
    assert_close(x.grad().data_at(4), 20.0f);
    assert_close(x.grad().data_at(5), 30.0f);

    // dy = sum(x, dimension 0)
    assert_close(y.grad().data_at(0), 5.0f);
    assert_close(y.grad().data_at(1), 7.0f);
    assert_close(y.grad().data_at(2), 9.0f);
}

/*
 * Division:
 *
 * loss = sum(x / y)
 *
 * dx = 1/y
 *
 * dy = -x/y^2, summed over the broadcast dimension.
 */
void test_div_broadcast_gradient()
{
    tensor x({2, 3}, true);
    tensor y({3}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 4.0f;
    x.data_at(2) = 6.0f;
    x.data_at(3) = 8.0f;
    x.data_at(4) = 10.0f;
    x.data_at(5) = 12.0f;

    y.data_at(0) = 2.0f;
    y.data_at(1) = 4.0f;
    y.data_at(2) = 6.0f;

    tensor loss = (x / y).sum();

    loss.backward();

    // dx = 1/y
    assert_close(x.grad().data_at(0), 0.5f);
    assert_close(x.grad().data_at(1), 0.25f);
    assert_close(x.grad().data_at(2), 1.0f / 6.0f);
    assert_close(x.grad().data_at(3), 0.5f);
    assert_close(x.grad().data_at(4), 0.25f);
    assert_close(x.grad().data_at(5), 1.0f / 6.0f);

    // dy = -(x1 + x2) / y^2
    assert_close(y.grad().data_at(0), -(2.0f + 8.0f) / 4.0f);
    assert_close(y.grad().data_at(1), -(4.0f + 10.0f) / 16.0f);
    assert_close(y.grad().data_at(2), -(6.0f + 12.0f) / 36.0f);
}

/*
 * Important regression test:
 *
 * If only one operand requires gradients, the other operand
 * should not receive a gradient.
 */
void test_broadcast_gradient_requires_grad()
{
    tensor x({2, 3}, true);
    tensor y({3}, false);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = static_cast<float>(i + 1);

    tensor loss = (x + y).sum();

    loss.backward();

    assert(x.has_grad());
    assert(!y.has_grad());

    assert_all_close(x.grad(), 1.0f);
}

/*
 * Test gradient accumulation through a broadcasted operation.
 *
 * loss = sum((x + y) * (x + y))
 *
 * Both paths contribute to the same tensors.
 *
 * This catches bugs where a node overwrites an existing gradient
 * instead of accumulating it.
 */
void test_broadcast_gradient_accumulation()
{
    tensor x({2, 3}, true);
    tensor y({3}, true);

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = 1.0f;

    for (std::size_t i = 0; i < y.size(); ++i)
        y.data_at(i) = 2.0f;

    tensor z = x + y;
    tensor loss = (z * z).sum();

    loss.backward();

    /*
     * z = 3 everywhere.
     *
     * d(z^2)/dz = 2z = 6.
     *
     * x is not broadcast:
     * dx = 6.
     *
     * y is broadcast over dimension 0:
     * dy = 6 + 6 = 12.
     */
    assert_all_close(x.grad(), 6.0f);
    assert_all_close(y.grad(), 12.0f);
}


// void test_linear(){
//     linear layer = linear(4, 2);
//     tensor x = tensor({8, 4});
//     tensor predictions = layer.forward(x);

//     layer.get_weights().grad();
//     layer.get_bias().grad();
// }

void test_linear()
{
    linear layer(4, 2);

    tensor x({8, 4});

    for (std::size_t i = 0; i < x.size(); ++i)
        x.data_at(i) = static_cast<float>(i + 1);

    tensor predictions = layer.forward(x);

    // Check output shape.
    assert(predictions.shape() ==
           std::vector<int>({8, 2}));

    // Check that the parameters have the expected shapes.
    assert(layer.get_weights().shape() ==
           std::vector<int>({4, 2}));

    assert(layer.get_bias().shape() ==
           std::vector<int>({2}));

    // Create loss and run backward.
    tensor loss = predictions.sum();
    loss.backward();

    // Check gradients exist.
    assert(layer.get_weights().has_grad());
    assert(layer.get_bias().has_grad());

    // Check gradient shapes.
    assert(layer.get_weights().grad().shape() ==
           layer.get_weights().shape());

    assert(layer.get_bias().grad().shape() ==
           layer.get_bias().shape());

    // Check bias gradient.
    assert(layer.get_bias().grad().data_at(0) == 8.0f);
    assert(layer.get_bias().grad().data_at(1) == 8.0f);

    // Check weight gradients.
    assert(layer.get_weights().grad().data_at(0) == 120.0f);
    assert(layer.get_weights().grad().data_at(1) == 120.0f);

    assert(layer.get_weights().grad().data_at(2) == 128.0f);
    assert(layer.get_weights().grad().data_at(3) == 128.0f);

    assert(layer.get_weights().grad().data_at(4) == 136.0f);
    assert(layer.get_weights().grad().data_at(5) == 136.0f);

    assert(layer.get_weights().grad().data_at(6) == 144.0f);
    assert(layer.get_weights().grad().data_at(7) == 144.0f);
}

int main()
{
    test_add_broadcast_gradient_lower_rank();
    test_add_broadcast_gradient_singleton_dimension();
    test_add_broadcast_gradient_middle_dimension();
    test_add_broadcast_gradient_multiple_dimensions();

    test_sub_broadcast_gradient();
    test_mul_broadcast_gradient();
    test_div_broadcast_gradient();

    test_broadcast_gradient_requires_grad();
    test_broadcast_gradient_accumulation();

    std::cout << "All broadcasting gradient tests passed!\n";
}