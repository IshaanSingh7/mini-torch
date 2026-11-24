cmake_minimum_required(VERSION 3.20)

project(mini_torch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tensor library
add_library(mini_torch
    src/core/tensor.cpp
    src/core/broadcasting.cpp
    src/core/tensor_impl.cpp
    src/autograd/node.cpp
    src/autograd/nodes/add_node.cpp
    src/autograd/nodes/mul_node.cpp
    src/autograd/nodes/div_node.cpp
    src/autograd/nodes/sub_node.cpp
    src/autograd/nodes/relu_node.cpp
    src/autograd/nodes/matmul_node.cpp
    src/autograd/nodes/sum_node.cpp
    src/autograd/nodes/mean_node.cpp
    src/autograd/nodes/scalar_mul_node.cpp
    src/autograd/nodes/scalar_div_node.cpp
    src/nn/linear.cpp
    src/nn/sequential.cpp
    src/nn/relu.cpp
    src/nn/module.cpp
    src/nn/loss.cpp
    src/optim/optimizer.cpp
    src/optim/sgd.cpp
)

target_include_directories(mini_torch
    PUBLIC
        include
)

# Example program
add_executable(basic
    examples/basic.cpp
)

target_link_libraries(basic
    PRIVATE
        mini_torch
)

# Tests
enable_testing()

add_executable(test_sgd
    tests/optim/test_sgd.cpp
)

add_executable(test_tensor
    tests/core/test_tensor.cpp
)

add_executable(test_nodes
    tests/autograd/test_nodes.cpp
)
add_executable(test_broadcast
    tests/core/test_broadcast.cpp
)

add_executable(test_sequential
    tests/nn/test_sequential.cpp
)

add_executable(test_training_xor
    tests/integration/test_training_xor.cpp
)

add_executable(test_training_linear
    tests/integration/test_training_linear.cpp
)

add_executable(test_training_mnist
    tests/integration/test_training_mnist.cpp
)



target_link_libraries(test_sgd
    PRIVATE
        mini_torch
)

target_link_libraries(test_tensor
    PRIVATE
        mini_torch
)

target_link_libraries(test_nodes
    PRIVATE
        mini_torch
)

target_link_libraries(test_broadcast
    PRIVATE
        mini_torch
)

target_link_libraries(test_sequential
    PRIVATE
        mini_torch
)

target_link_libraries(test_training_xor
    PRIVATE
        mini_torch
)

target_link_libraries(test_training_linear
    PRIVATE
        mini_torch
)

target_link_libraries(test_training_mnist
    PRIVATE
        mini_torch
)

add_test(
    NAME sgd_test
    COMMAND test_sgd
)

add_test(
    NAME tensor_test
    COMMAND test_tensor
)

add_test(
    NAME nodes_test
    COMMAND test_nodes
)

add_test(
    NAME broadcast_test
    COMMAND test_broadcast
)

add_test(
    NAME sequential_test
    COMMAND test_sequential
)

add_test(
    NAME training_xor_test
    COMMAND test_training_xor
)

add_test(
    NAME training_linear_test
    COMMAND test_training_linear
)

add_test(
    NAME training_mnist_test
    COMMAND test_training_mnist
)