#include "mini_torch/autograd/nodes/add_node.h"
#include "mini_torch/autograd/nodes/mul_node.h"
#include "mini_torch/autograd/nodes/sub_node.h"
#include "mini_torch/autograd/nodes/div_node.h"
#include "mini_torch/autograd/nodes/relu_node.h"
#include "mini_torch/core/tensor.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

bool is_close(float a, float b, float tolerance = 1e-5f)
{
    return std::fabs(a - b) < tolerance;
}

void assert_tensor_values(
    const tensor &t,
    const std::vector<float> &expected,
    float tolerance = 1e-5f)
{
    assert(t.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        assert(is_close(t.data_at(i), expected[i], tolerance));
    }
}

// ---------------------------------------------------------
// ADD
// z = x + y
//
// dz/dx = 1
// dz/dy = 1
// ---------------------------------------------------------

void test_add_node()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;

    y.data_at(0) = 5.0f;
    y.data_at(1) = 6.0f;
    y.data_at(2) = 7.0f;
    y.data_at(3) = 8.0f;

    // Test node backward
    add_node node(x, y);

    tensor upstream({2, 2});

    for (std::size_t i = 0; i < upstream.size(); ++i)
    {
        upstream.data_at(i) = 1.0f;
    }

    node.backward(upstream);

    assert(x.has_grad());
    assert(y.has_grad());

    assert_tensor_values(
        x.grad(),
        {1.0f, 1.0f, 1.0f, 1.0f});

    assert_tensor_values(
        y.grad(),
        {1.0f, 1.0f, 1.0f, 1.0f});

    std::cout << "add_node: PASSED\n";
}

// ---------------------------------------------------------
// MUL
// z = x * y
//
// dz/dx = y
// dz/dy = x
// ---------------------------------------------------------

void test_mul_node()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = 4.0f;
    x.data_at(3) = 5.0f;

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;
    y.data_at(3) = 40.0f;

    mul_node node(x, y);

    tensor upstream({2, 2});

    upstream.data_at(0) = 1.0f;
    upstream.data_at(1) = 2.0f;
    upstream.data_at(2) = 3.0f;
    upstream.data_at(3) = 4.0f;

    node.backward(upstream);

    assert(x.has_grad());
    assert(y.has_grad());

    // x.grad = upstream * y
    assert_tensor_values(
        x.grad(),
        {10.0f,
         40.0f,
         90.0f,
         160.0f});

    // y.grad = upstream * x
    assert_tensor_values(
        y.grad(),
        {2.0f,
         6.0f,
         12.0f,
         20.0f});

    std::cout << "mul_node: PASSED\n";
}

// ---------------------------------------------------------
// SUB
// z = x - y
//
// dz/dx = 1
// dz/dy = -1
// ---------------------------------------------------------

void test_sub_node()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 10.0f;
    x.data_at(1) = 20.0f;
    x.data_at(2) = 30.0f;
    x.data_at(3) = 40.0f;

    y.data_at(0) = 1.0f;
    y.data_at(1) = 2.0f;
    y.data_at(2) = 3.0f;
    y.data_at(3) = 4.0f;

    sub_node node(x, y);

    tensor upstream({2, 2});

    upstream.data_at(0) = 2.0f;
    upstream.data_at(1) = 4.0f;
    upstream.data_at(2) = 6.0f;
    upstream.data_at(3) = 8.0f;

    node.backward(upstream);

    assert(x.has_grad());
    assert(y.has_grad());

    // x.grad = upstream
    assert_tensor_values(
        x.grad(),
        {2.0f,
         4.0f,
         6.0f,
         8.0f});

    // y.grad = -upstream
    assert_tensor_values(
        y.grad(),
        {-2.0f,
         -4.0f,
         -6.0f,
         -8.0f});

    std::cout << "sub_node: PASSED\n";
}

// ---------------------------------------------------------
// DIV
// z = x / y
//
// dz/dx = 1/y
// dz/dy = -x/y^2
// ---------------------------------------------------------

void test_div_node()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 10.0f;
    x.data_at(1) = 20.0f;
    x.data_at(2) = 30.0f;
    x.data_at(3) = 40.0f;

    y.data_at(0) = 2.0f;
    y.data_at(1) = 4.0f;
    y.data_at(2) = 5.0f;
    y.data_at(3) = 8.0f;

    div_node node(x, y);

    tensor upstream({2, 2});

    upstream.data_at(0) = 1.0f;
    upstream.data_at(1) = 2.0f;
    upstream.data_at(2) = 3.0f;
    upstream.data_at(3) = 4.0f;

    node.backward(upstream);

    assert(x.has_grad());
    assert(y.has_grad());

    // x.grad = upstream / y
    assert_tensor_values(
        x.grad(),
        {0.5f,
         0.5f,
         0.6f,
         0.5f});

    // y.grad = upstream * (-x / y^2)
    assert_tensor_values(
        y.grad(),
        {-2.5f,
         -2.5f,
         -3.6f,
         -2.5f});

    std::cout << "div_node: PASSED\n";
}

void test_full_mul_backward()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = 4.0f;
    x.data_at(3) = 5.0f;

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;
    y.data_at(3) = 40.0f;

    tensor z = x * y;

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    assert_tensor_values(
        x.grad(),
        {10.0f, 20.0f, 30.0f, 40.0f});

    assert_tensor_values(
        y.grad(),
        {2.0f, 3.0f, 4.0f, 5.0f});

    std::cout << "full mul backward: PASSED\n";
}

void test_full_add_backward()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = 4.0f;
    x.data_at(3) = 5.0f;

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;
    y.data_at(3) = 40.0f;

    tensor z = x + y;

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    // dz/dx = 1.0f for all elements
    assert_tensor_values(
        x.grad(),
        {1.0f, 1.0f, 1.0f, 1.0f});

    // dz/dy = 1.0f for all elements
    assert_tensor_values(
        y.grad(),
        {1.0f, 1.0f, 1.0f, 1.0f});

    std::cout << "full add backward: PASSED\n";
}

void test_full_sub_backward()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = 4.0f;
    x.data_at(3) = 5.0f;

    y.data_at(0) = 10.0f;
    y.data_at(1) = 20.0f;
    y.data_at(2) = 30.0f;
    y.data_at(3) = 40.0f;

    tensor z = x - y;

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    // dz/dx = 1.0f for all elements
    assert_tensor_values(
        x.grad(),
        {1.0f, 1.0f, 1.0f, 1.0f});

    // dz/dy = -1.0f for all elements
    assert_tensor_values(
        y.grad(),
        {-1.0f, -1.0f, -1.0f, -1.0f});

    std::cout << "full sub backward: PASSED\n";
}

void test_full_div_backward()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 10.0f;
    x.data_at(1) = 12.0f;
    x.data_at(2) = 16.0f;
    x.data_at(3) = 20.0f;

    y.data_at(0) = 2.0f;
    y.data_at(1) = 4.0f;
    y.data_at(2) = 4.0f;
    y.data_at(3) = 5.0f;

    tensor z = x / y;

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    // dz/dx = 1 / y -> {1/2, 1/4, 1/4, 1/5}
    assert_tensor_values(
        x.grad(),
        {0.5f, 0.25f, 0.25f, 0.2f});

    // dz/dy = -x / (y^2) -> {-10/4, -12/16, -16/16, -20/25}
    assert_tensor_values(
        y.grad(),
        {-2.5f, -0.75f, -1.0f, -0.8f});

    std::cout << "full div backward: PASSED\n";
}

void test_relu_node()
{
    tensor x({2, 3}, true);

    x.data_at(0) = -2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = 0.0f;
    x.data_at(3) = 5.0f;
    x.data_at(4) = -1.0f;
    x.data_at(5) = 4.0f;

    relu_node node(x);

    tensor upstream({2, 3});

    upstream.data_at(0) = 1.0f;
    upstream.data_at(1) = 2.0f;
    upstream.data_at(2) = 3.0f;
    upstream.data_at(3) = 4.0f;
    upstream.data_at(4) = 5.0f;
    upstream.data_at(5) = 6.0f;

    node.backward(upstream);

    assert(x.has_grad());

    tensor gradient = x.grad();

    // x <= 0 -> gradient 0
    // x > 0  -> upstream gradient
    assert(gradient.data_at(0) == 0.0f);
    assert(gradient.data_at(1) == 2.0f);
    assert(gradient.data_at(2) == 0.0f);
    assert(gradient.data_at(3) == 4.0f);
    assert(gradient.data_at(4) == 0.0f);
    assert(gradient.data_at(5) == 6.0f);

    std::cout << "relu_node: PASSED\n";
}

void test_chained_mul_relu_backward()
{
    // x and y are the leaf tensors.
    tensor x({2, 3}, true);
    tensor y({2, 3}, true);

    // x = [-2,  3, -4,
    //       5,  2, -1]
    x.data_at(0) = -2.0f;
    x.data_at(1) = 3.0f;
    x.data_at(2) = -4.0f;
    x.data_at(3) = 5.0f;
    x.data_at(4) = 2.0f;
    x.data_at(5) = -1.0f;

    // y = [ 2,  4,  3,
    //       2, -5,  6]
    y.data_at(0) = 2.0f;
    y.data_at(1) = 4.0f;
    y.data_at(2) = 3.0f;
    y.data_at(3) = 2.0f;
    y.data_at(4) = -5.0f;
    y.data_at(5) = 6.0f;

    // Forward:
    //
    // z = x * y
    //
    // z = [-4, 12, -12,
    //       10, -10, -6]
    tensor z = x * y;

    assert_tensor_values(
        z,
        {-4.0f,
         12.0f,
         -12.0f,
         10.0f,
         -10.0f,
         -6.0f});

    // w = relu(z)
    //
    // w = [0, 12, 0,
    //      10, 0, 0]
    tensor w = z.relu();

    assert_tensor_values(
        w,
        {0.0f,
         12.0f,
         0.0f,
         10.0f,
         0.0f,
         0.0f});

    // Start the backward pass at w.
    //
    // Since backward() starts with an upstream gradient
    // of all ones:
    //
    // dw/dz = [0, 1, 0,
    //          1, 0, 0]
    //
    assert(z.has_grad_fn());
    assert(w.has_grad_fn());
    w.backward();

    // Both x and y should have received gradients.
    assert(x.has_grad());
    assert(y.has_grad());

    // For z = x * y:
    //
    // dz/dx = y
    // dz/dy = x
    //
    // ReLU gives zero gradient wherever z <= 0.
    //
    // Therefore:
    //
    // x.grad = [0, 4, 0,
    //           2, 0, 0]
    //
    // y.grad = [0, 3, 0,
    //           5, 0, 0]

    assert_tensor_values(
        x.grad(),
        {0.0f,
         4.0f,
         0.0f,
         2.0f,
         0.0f,
         0.0f});

    assert_tensor_values(
        y.grad(),
        {0.0f,
         3.0f,
         0.0f,
         5.0f,
         0.0f,
         0.0f});

    std::cout << "chained mul -> relu backward: PASSED\n";
}

void test_gradient_accumulation()
{
    tensor x({2, 2}, true);

    tensor grad1({2, 2});
    grad1.data_at(0) = 1.0f;
    grad1.data_at(1) = 2.0f;
    grad1.data_at(2) = 3.0f;
    grad1.data_at(3) = 4.0f;

    tensor grad2({2, 2});
    grad2.data_at(0) = 10.0f;
    grad2.data_at(1) = 20.0f;
    grad2.data_at(2) = 30.0f;
    grad2.data_at(3) = 40.0f;

    x.accumulate_grad(grad1);
    x.accumulate_grad(grad2);

    assert(x.has_grad());

    assert_tensor_values(
        x.grad(),
        {11.0f, 22.0f, 33.0f, 44.0f});

    std::cout << "gradient accumulation: PASSED\n";
}


// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-4f)
{
    if (!is_close(actual, expected, tolerance))
    {
        std::cerr
            << "Expected " << expected
            << ", got " << actual
            << std::endl;

        assert(false);
    }
}

void assert_tensor_close(
    const tensor &actual,
    const tensor &expected,
    float tolerance = 1e-4f)
{
    assert(actual.shape() == expected.shape());

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (!is_close(
                actual.data_at(i),
                expected.data_at(i),
                tolerance))
        {
            std::cerr
                << "Mismatch at index " << i
                << ": expected "
                << expected.data_at(i)
                << ", got "
                << actual.data_at(i)
                << std::endl;

            assert(false);
        }
    }
}

// ------------------------------------------------------------
// Test 1: Forward matmul + basic autograd
// ------------------------------------------------------------

void test_matmul_backward_2d()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, true);

    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;

    y.data_at(0) = 5.0f;
    y.data_at(1) = 6.0f;
    y.data_at(2) = 7.0f;
    y.data_at(3) = 8.0f;

    tensor z = x.matmul(y);

    // --------------------------------------------------------
    // Forward result check
    //
    // [1 2] [5 6] = [19 22]
    // [3 4] [7 8]   [43 50]
    // --------------------------------------------------------

    assert(z.shape() == std::vector<int>({2, 2}));

    assert_close(z.data_at(0), 19.0f);
    assert_close(z.data_at(1), 22.0f);
    assert_close(z.data_at(2), 43.0f);
    assert_close(z.data_at(3), 50.0f);

    assert(z.requires_grad());
    assert(z.has_grad_fn());

    // --------------------------------------------------------
    // Backward
    // --------------------------------------------------------

    z.backward();

    assert(x.has_grad());
    assert(y.has_grad());

    tensor x_grad = x.grad();
    tensor y_grad = y.grad();

    // --------------------------------------------------------
    // Corrected Expected dx:
    //
    // dx = dz @ y^T
    //
    // [1 1] [5 7] = [11 15]
    // [1 1] [6 8]   [11 15]
    // --------------------------------------------------------

    assert(x_grad.shape() == std::vector<int>({2, 2}));

    assert_close(x_grad.data_at(0), 11.0f);
    assert_close(x_grad.data_at(1), 15.0f);
    assert_close(x_grad.data_at(2), 11.0f);
    assert_close(x_grad.data_at(3), 15.0f);

    // --------------------------------------------------------
    // Expected dy:
    //
    // dy = x^T @ dz
    //
    // [1 3] [1 1] = [4 4]
    // [2 4] [1 1]   [6 6]
    // --------------------------------------------------------

    assert(y_grad.shape() == std::vector<int>({2, 2}));

    assert_close(y_grad.data_at(0), 4.0f);
    assert_close(y_grad.data_at(1), 4.0f);
    assert_close(y_grad.data_at(2), 6.0f);
    assert_close(y_grad.data_at(3), 6.0f);

    std::cout << "test_matmul_backward_2d passed\n";
}

// ------------------------------------------------------------
// Test 2: Only x requires gradients
// ------------------------------------------------------------

void test_matmul_only_x_requires_grad()
{
    tensor x({2, 2}, true);
    tensor y({2, 2}, false);

    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;

    y.data_at(0) = 5.0f;
    y.data_at(1) = 6.0f;
    y.data_at(2) = 7.0f;
    y.data_at(3) = 8.0f;

    tensor z = x.matmul(y);

    assert(z.requires_grad());
    assert(z.has_grad_fn());

    z.backward();

    assert(x.has_grad());

    // y should not have accumulated a gradient.
    assert(!y.has_grad());

    std::cout << "test_matmul_only_x_requires_grad passed\n";
}

// ------------------------------------------------------------
// Test 3: Only y requires gradients
// ------------------------------------------------------------

void test_matmul_only_y_requires_grad()
{
    tensor x({2, 2}, false);
    tensor y({2, 2}, true);

    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;

    y.data_at(0) = 5.0f;
    y.data_at(1) = 6.0f;
    y.data_at(2) = 7.0f;
    y.data_at(3) = 8.0f;

    tensor z = x.matmul(y);

    assert(z.requires_grad());
    assert(z.has_grad_fn());

    z.backward();

    assert(!x.has_grad());
    assert(y.has_grad());

    tensor y_grad = y.grad();

    assert_close(y_grad.data_at(0), 4.0f);
    assert_close(y_grad.data_at(1), 4.0f);
    assert_close(y_grad.data_at(2), 6.0f);
    assert_close(y_grad.data_at(3), 6.0f);

    std::cout << "test_matmul_only_y_requires_grad passed\n";
}

// ------------------------------------------------------------
// Test 4: Batched matmul differentiation
// ------------------------------------------------------------

void test_matmul_backward_batched()
{
    /*
        x shape = {2, 2, 3}
        y shape = {2, 3, 2}

        Therefore:

        z shape = {2, 2, 2}
    */

    tensor x({2, 2, 3}, true);
    tensor y({2, 3, 2}, true);

    // Batch 0:
    //
    // x:
    // [1 2 3]
    // [4 5 6]
    //
    // y:
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    x.data_at(0) = 1;
    x.data_at(1) = 2;
    x.data_at(2) = 3;
    x.data_at(3) = 4;
    x.data_at(4) = 5;
    x.data_at(5) = 6;

    y.data_at(0) = 7;
    y.data_at(1) = 8;
    y.data_at(2) = 9;
    y.data_at(3) = 10;
    y.data_at(4) = 11;
    y.data_at(5) = 12;

    // Batch 1:
    //
    // x:
    // [ 7  8  9]
    // [10 11 12]
    //
    // y:
    // [13 14]
    // [15 16]
    // [17 18]

    x.data_at(6) = 7;
    x.data_at(7) = 8;
    x.data_at(8) = 9;
    x.data_at(9) = 10;
    x.data_at(10) = 11;
    x.data_at(11) = 12;

    y.data_at(6) = 13;
    y.data_at(7) = 14;
    y.data_at(8) = 15;
    y.data_at(9) = 16;
    y.data_at(10) = 17;
    y.data_at(11) = 18;

    tensor z = x.matmul(y);

    assert(z.shape() == std::vector<int>({2, 2, 2}));
    assert(z.requires_grad());
    assert(z.has_grad_fn());

    // Forward check, batch 0:
    //
    // [1 2 3] [7  8] = [58  64]
    // [4 5 6] [9 10]   [139 154]
    //
    assert_close(z.data_at(0), 58.0f);
    assert_close(z.data_at(1), 64.0f);
    assert_close(z.data_at(2), 139.0f);
    assert_close(z.data_at(3), 154.0f);

    // Batch 1:
    //
    // [7  8  9]  [13 14] = [364 388]
    // [10 11 12] [15 16]   [499 532]
    //            [17 18]
    //
    assert_close(z.data_at(4), 364.0f);
    assert_close(z.data_at(5), 388.0f);
    assert_close(z.data_at(6), 499.0f);
    assert_close(z.data_at(7), 532.0f);

    z.backward();

    tensor x_grad = x.grad();
    tensor y_grad = y.grad();

    assert(x_grad.shape() == x.shape());
    assert(y_grad.shape() == y.shape());

    // Since dz consists entirely of ones:
    //
    // dx = dz @ y^T
    //
    // Batch 0:
    //
    // [1 1] [7  9 11] = [15 19 23]
    // [1 1] [8 10 12]   [15 19 23]
    //
    assert_close(x_grad.data_at(0), 15.0f);
    assert_close(x_grad.data_at(1), 19.0f);
    assert_close(x_grad.data_at(2), 23.0f);
    assert_close(x_grad.data_at(3), 15.0f);
    assert_close(x_grad.data_at(4), 19.0f);
    assert_close(x_grad.data_at(5), 23.0f);

    // Batch 1:
    //
    // [1 1] [13 15 17] = [27 31 35]
    // [1 1] [14 16 18]   [27 31 35]
    //
    assert_close(x_grad.data_at(6), 27.0f);
    assert_close(x_grad.data_at(7), 31.0f);
    assert_close(x_grad.data_at(8), 35.0f);
    assert_close(x_grad.data_at(9), 27.0f);
    assert_close(x_grad.data_at(10), 31.0f);
    assert_close(x_grad.data_at(11), 35.0f);

    // dy = x^T @ dz
    //
    // Batch 0:
    //
    // [1 4] [1 1] = [5 5]
    // [2 5] [1 1]   [7 7]
    // [3 6]          [9 9]
    //
    assert_close(y_grad.data_at(0), 5.0f);
    assert_close(y_grad.data_at(1), 5.0f);
    assert_close(y_grad.data_at(2), 7.0f);
    assert_close(y_grad.data_at(3), 7.0f);
    assert_close(y_grad.data_at(4), 9.0f);
    assert_close(y_grad.data_at(5), 9.0f);

    // Batch 1:
    //
    // [ 7 10] [1 1] = [17 17]
    // [ 8 11] [1 1]   [19 19]
    // [ 9 12]          [21 21]
    //
    assert_close(y_grad.data_at(6), 17.0f);
    assert_close(y_grad.data_at(7), 17.0f);
    assert_close(y_grad.data_at(8), 19.0f);
    assert_close(y_grad.data_at(9), 19.0f);
    assert_close(y_grad.data_at(10), 21.0f);
    assert_close(y_grad.data_at(11), 21.0f);

    std::cout << "test_matmul_backward_batched passed\n";
}
void test_sum_node()
{
    tensor x({3}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 4.0f;
    x.data_at(2) = 6.0f;

    tensor y = x.sum();

    assert(y.size() == 1);
    assert_close(y.data_at(0), 12.0f);
    assert(y.requires_grad());
    assert(y.has_grad_fn());

    y.backward();

    tensor x_grad = x.grad();

    assert_close(x_grad.data_at(0), 1.0f);
    assert_close(x_grad.data_at(1), 1.0f);
    assert_close(x_grad.data_at(2), 1.0f);
    std::cout << "sum node test complete!\n";
}

void test_mean_backward()
{
    tensor x({4}, true);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 4.0f;
    x.data_at(2) = 6.0f;
    x.data_at(3) = 8.0f;

    tensor y = x.mean();

    assert(y.size() == 1);
    assert_close(y.data_at(0), 5.0f);

    assert(y.requires_grad());
    assert(y.has_grad_fn());

    y.backward();

    tensor x_grad = x.grad();

    assert_close(x_grad.data_at(0), 0.25f);
    assert_close(x_grad.data_at(1), 0.25f);
    assert_close(x_grad.data_at(2), 0.25f);
    assert_close(x_grad.data_at(3), 0.25f);

    std::cout << "test_mean_backward passed\n";
}

void test_numerical_gradient_checking_sum(float epsilon = 1e-3f)
{
    tensor x({2}, true);
    tensor y({2}, false);

    x.data_at(0) = 2.0f;
    x.data_at(1) = 3.0f;

    y.data_at(0) = 4.0f;
    y.data_at(1) = 5.0f;

    // Autograd
    tensor z = x * y;
    tensor loss = z.sum();

    loss.backward();

    tensor x_grad = x.grad();

    // Numerical gradient
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        float original = x.data_at(i);

        // f(x + epsilon)
        x.data_at(i) = original + epsilon;

        double loss_plus = 0.0;

        for (std::size_t j = 0; j < x.size(); ++j)
        {
            loss_plus +=
                static_cast<double>(x.data_at(j)) *
                static_cast<double>(y.data_at(j));
        }

        // f(x - epsilon)
        x.data_at(i) = original - epsilon;

        double loss_minus = 0.0;

        for (std::size_t j = 0; j < x.size(); ++j)
        {
            loss_minus +=
                static_cast<double>(x.data_at(j)) *
                static_cast<double>(y.data_at(j));
        }

        // Restore x
        x.data_at(i) = original;

        double numerical_grad =
            (loss_plus - loss_minus) /
            (2.0 * static_cast<double>(epsilon));

        std::cout
            << "x[" << i << "] "
            << "autograd = " << x_grad.data_at(i)
            << ", numerical = " << numerical_grad
            << '\n';

        assert_close(
            x_grad.data_at(i),
            static_cast<float>(numerical_grad),
            1e-3f);
    }

    std::cout << "numerical gradient checking passed!\n";
}

float calculate_matmul_sum(
    const tensor &x,
    const tensor &y)
{
    float loss = 0.0f;

    for (std::size_t i = 0; i < x.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < y.shape()[1]; ++j)
        {
            float value = 0.0f;

            for (std::size_t k = 0; k < x.shape()[1]; ++k)
            {
                value +=
                    x.data_at(i * x.shape()[1] + k) *
                    y.data_at(k * y.shape()[1] + j);
            }

            loss += value;
        }
    }

    return loss;
}

void test_numerical_gradient_checking_matmul_sum(float epsilon = 1e-3)
{
    tensor x = tensor({2, 3}, true);
    tensor y = tensor({3, 2}, true);
    x.data_at(0) = 1.0f;
    x.data_at(1) = 2.0f;
    x.data_at(2) = 3.0f;
    x.data_at(3) = 4.0f;
    x.data_at(4) = 5.0f;
    x.data_at(5) = 6.0f;

    y.data_at(0) = 7.0f;
    y.data_at(1) = 8.0f;
    y.data_at(2) = 9.0f;
    y.data_at(3) = 10.0f;
    y.data_at(4) = 11.0f;
    y.data_at(5) = 12.0f;

    tensor z = x.matmul(y);
    tensor loss = z.sum();

    loss.backward();
    tensor x_grad = x.grad();
    tensor y_grad = y.grad();

    // checking x gradients
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        float original = x.data_at(i);

        x.data_at(i) = original + epsilon;
        float loss_plus = calculate_matmul_sum(x, y);

        x.data_at(i) = original - epsilon;
        float loss_minus = calculate_matmul_sum(x, y);

        x.data_at(i) = original;

        float numerical_gradient = (loss_plus - loss_minus) / (2.0f * epsilon);

        std::cout
            << "x[" << i << "] "
            << "autograd = " << x_grad.data_at(i)
            << ", numerical = " << numerical_gradient
            << '\n';

        assert_close(x_grad.data_at(i), numerical_gradient);
    }

    // checking y gradients

    for (std::size_t i = 0; i < y.size(); ++i)
    {
        float original = y.data_at(i);

        y.data_at(i) = original + epsilon;
        float loss_plus = calculate_matmul_sum(x, y);

        y.data_at(i) = original - epsilon;
        float loss_minus = calculate_matmul_sum(x, y);
        float numerical_gradient = (loss_plus - loss_minus) / (2.0f * epsilon);

        std::cout
            << "y[" << i << "] "
            << "autograd = " << y_grad.data_at(i)
            << ", numerical = " << numerical_gradient
            << '\n';

        assert_close(y_grad.data_at(i), numerical_gradient);
    }
    std::cout
        << "test_numerical_gradient_checking_matmul_sum passed\n";
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------

int main()
{
    test_add_node();
    test_mul_node();
    test_sub_node();
    test_div_node();
    test_full_mul_backward();
    test_full_add_backward();
    test_full_div_backward();
    test_full_sub_backward();
    test_chained_mul_relu_backward();
    test_gradient_accumulation();

    std::cout << "\nAll node tests PASSED!\n";

    test_matmul_backward_2d();
    test_matmul_only_x_requires_grad();
    test_matmul_only_y_requires_grad();
    test_matmul_backward_batched();

    std::cout << "\nAll matmul differentiation tests passed!\n";

    test_sum_node();
    test_mean_backward();
    test_numerical_gradient_checking_sum();

    test_numerical_gradient_checking_matmul_sum();

    return 0;
}