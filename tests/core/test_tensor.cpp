#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "mini_torch/core/tensor_impl.h"
#include "mini_torch/core/tensor.h"

bool is_close(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) <= tolerance;
}

void test_construction()
{
    tensor x({2, 3, 4});

    assert(x.total_elements() == 24);
    assert(x.size() == 24);

    std::cout << "construction: PASSED\n";
}

void test_set_and_at()
{
    tensor x({2, 3});

    x.set({0, 0}, 1.0f);
    x.set({0, 1}, 2.0f);
    x.set({0, 2}, 3.0f);

    x.set({1, 0}, 4.0f);
    x.set({1, 1}, 5.0f);
    x.set({1, 2}, 6.0f);

    assert(is_close(x.at({0, 0}), 1.0f));
    assert(is_close(x.at({0, 1}), 2.0f));
    assert(is_close(x.at({0, 2}), 3.0f));
    assert(is_close(x.at({1, 0}), 4.0f));
    assert(is_close(x.at({1, 1}), 5.0f));
    assert(is_close(x.at({1, 2}), 6.0f));

    std::cout << "set/at: PASSED\n";
}

void test_get()
{
    tensor x({2, 3});

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            x.set({i, j}, static_cast<float>(i * 3 + j + 1));
        }
    }

    std::vector<float> first_row = x.get({0});

    assert(first_row.size() == 3);
    assert(is_close(first_row[0], 1.0f));
    assert(is_close(first_row[1], 2.0f));
    assert(is_close(first_row[2], 3.0f));

    std::vector<float> second_row = x.get({1});

    assert(second_row.size() == 3);
    assert(is_close(second_row[0], 4.0f));
    assert(is_close(second_row[1], 5.0f));
    assert(is_close(second_row[2], 6.0f));

    std::cout << "get: PASSED\n";
}

void test_three_dimensional_indexing()
{
    tensor x({2, 3, 4});

    int value = 0;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                x.set({i, j, k}, static_cast<float>(value));
                ++value;
            }
        }
    }

    value = 0;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                assert(is_close(
                    x.at({i, j, k}),
                    static_cast<float>(value)));

                ++value;
            }
        }
    }

    std::cout << "3D indexing: PASSED\n";
}

void test_addition()
{
    tensor a({2, 3});
    tensor b({2, 3});

    float value = 1.0f;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            a.set({i, j}, value);
            b.set({i, j}, value * 10.0f);
            value += 1.0f;
        }
    }

    tensor c = a + b;

    assert(is_close(c.at({0, 0}), 11.0f));
    assert(is_close(c.at({0, 1}), 22.0f));
    assert(is_close(c.at({0, 2}), 33.0f));
    assert(is_close(c.at({1, 0}), 44.0f));
    assert(is_close(c.at({1, 1}), 55.0f));
    assert(is_close(c.at({1, 2}), 66.0f));

    std::cout << "addition: PASSED\n";
}

void test_subtraction()
{
    tensor a({2, 2});
    tensor b({2, 2});

    a.set({0, 0}, 10.0f);
    a.set({0, 1}, 20.0f);
    a.set({1, 0}, 30.0f);
    a.set({1, 1}, 40.0f);

    b.set({0, 0}, 1.0f);
    b.set({0, 1}, 2.0f);
    b.set({1, 0}, 3.0f);
    b.set({1, 1}, 4.0f);

    tensor c = a - b;

    assert(is_close(c.at({0, 0}), 9.0f));
    assert(is_close(c.at({0, 1}), 18.0f));
    assert(is_close(c.at({1, 0}), 27.0f));
    assert(is_close(c.at({1, 1}), 36.0f));

    std::cout << "subtraction: PASSED\n";
}

void test_multiplication()
{
    tensor a({2, 2});
    tensor b({2, 2});

    a.set({0, 0}, 1.0f);
    a.set({0, 1}, 2.0f);
    a.set({1, 0}, 3.0f);
    a.set({1, 1}, 4.0f);

    b.set({0, 0}, 5.0f);
    b.set({0, 1}, 6.0f);
    b.set({1, 0}, 7.0f);
    b.set({1, 1}, 8.0f);

    tensor c = a * b;

    assert(is_close(c.at({0, 0}), 5.0f));
    assert(is_close(c.at({0, 1}), 12.0f));
    assert(is_close(c.at({1, 0}), 21.0f));
    assert(is_close(c.at({1, 1}), 32.0f));

    std::cout << "multiplication: PASSED\n";
}

void test_division()
{
    tensor a({2, 2});
    tensor b({2, 2});

    a.set({0, 0}, 10.0f);
    a.set({0, 1}, 20.0f);
    a.set({1, 0}, 30.0f);
    a.set({1, 1}, 40.0f);

    b.set({0, 0}, 2.0f);
    b.set({0, 1}, 4.0f);
    b.set({1, 0}, 5.0f);
    b.set({1, 1}, 8.0f);

    tensor c = a / b;

    assert(is_close(c.at({0, 0}), 5.0f));
    assert(is_close(c.at({0, 1}), 5.0f));
    assert(is_close(c.at({1, 0}), 6.0f));
    assert(is_close(c.at({1, 1}), 5.0f));

    std::cout << "division: PASSED\n";
}

void test_sum()
{
    tensor x({2, 3});

    x.set({0, 0}, 1.0f);
    x.set({0, 1}, 2.0f);
    x.set({0, 2}, 3.0f);
    x.set({1, 0}, 4.0f);
    x.set({1, 1}, 5.0f);
    x.set({1, 2}, 6.0f);

    assert(is_close(x.sum().data_at(0), 21.0f));

    std::cout << "sum: PASSED\n";
}

// void test_mean()
// {
//     tensor x({2, 3});

//     x.set({0, 0}, 1.0f);
//     x.set({0, 1}, 2.0f);
//     x.set({0, 2}, 3.0f);
//     x.set({1, 0}, 4.0f);
//     x.set({1, 1}, 5.0f);
//     x.set({1, 2}, 6.0f);

//     assert(is_close(x.mean(), 3.5f));

//     std::cout << "mean: PASSED\n";
// }

void test_relu()
{
    tensor x({2, 3});

    x.set({0, 0}, -2.0f);
    x.set({0, 1}, 0.0f);
    x.set({0, 2}, 3.0f);

    x.set({1, 0}, -4.5f);
    x.set({1, 1}, 5.5f);
    x.set({1, 2}, -1.0f);

    tensor y = x.relu();

    assert(is_close(y.at({0, 0}), 0.0f));
    assert(is_close(y.at({0, 1}), 0.0f));
    assert(is_close(y.at({0, 2}), 3.0f));
    assert(is_close(y.at({1, 0}), 0.0f));
    assert(is_close(y.at({1, 1}), 5.5f));
    assert(is_close(y.at({1, 2}), 0.0f));

    std::cout << "relu: PASSED\n";
}

void test_invalid_index()
{
    tensor x({2, 3});

    bool threw = false;

    try
    {
        x.at({2, 0});
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }

    assert(threw);

    threw = false;

    try
    {
        x.at({0, 3});
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }

    assert(threw);

    threw = false;

    try
    {
        x.at({0});
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }

    assert(threw);

    std::cout << "invalid indexing: PASSED\n";
}

void test_const_access()
{
    tensor x({2, 2});

    x.set({0, 0}, 1.0f);
    x.set({0, 1}, 2.0f);
    x.set({1, 0}, 3.0f);
    x.set({1, 1}, 4.0f);

    const tensor &const_x = x;

    assert(is_close(const_x.at({0, 0}), 1.0f));
    assert(is_close(const_x.at({1, 1}), 4.0f));

    std::cout << "const access: PASSED\n";
}

void test_requires_grad()
{
    tensor x({2, 3}, true);
    tensor y({2, 3});

    assert(x.requires_grad() == true);
    assert(y.requires_grad() == false);

    std::cout << "requires grad: PASSED \n";
}

void test_null_grad()
{
    tensor x = tensor({2, 3}, true);
    assert(!x.has_grad());
    std::cout << "null grad: PASSED \n";
}

void test_backward_add()
{
    tensor x({2, 3}, true);
    tensor y({2, 3}, true);

    tensor z = x + y;

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    tensor x_grad = x.grad();
    tensor y_grad = y.grad();

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        assert(x_grad.data_at(i) == 1.0f);
        assert(y_grad.data_at(i) == 1.0f);
    }

    std::cout << "add backward: PASSED\n";
}

int main()
{
    test_construction();
    test_set_and_at();
    test_get();
    test_three_dimensional_indexing();

    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();

    test_sum();
    // test_mean();
    test_relu();

    test_invalid_index();
    test_const_access();

    test_requires_grad();
    test_null_grad();
    test_backward_add();
    std::cout << "\nAll tensor tests passed!\n";

    tensor a({1});
    a.data_at(0) = 2.0f;

    tensor b = a * a;

    printf("a: %f\n", a.data_at(0));
    printf("a * a: %f\n", b.data_at(0));
    return 0;
}