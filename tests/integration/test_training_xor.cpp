#include "tensor.h"
#include "sgd.h"
#include "sequential.h"
#include "linear.h"
#include "relu.h"
#include <memory>
#include <iostream>
#include "loss.h"
#include <random>
#include <cmath>

int main()
{
    sequential model = sequential();

    model.add(std::make_unique<linear>(2, 4));
    model.add(std::make_unique<relu>());
    model.add(std::make_unique<linear>(4, 1));

    sgd optimizer = sgd(model.parameters(), 0.01f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);

    int samples = 1000;

    // Inputs
    tensor inputs = tensor({samples, 2});
    for (std::size_t i = 0; i < samples * 2; ++i)
    {
        inputs.data_at(i) = distrib(gen);
    }

    // Targets
    tensor targets = tensor({samples, 1});
    for (std::size_t i = 0; i < samples; i++)
    {
        int a = static_cast<int>(inputs.data_at(i * 2));
        int b = static_cast<int>(inputs.data_at(i * 2 + 1));
        targets.data_at(i) = static_cast<float>(a ^ b);
    }


    int num_samples = inputs.shape()[0];
    int epochs = 1000;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {

        optimizer.zero_grad();
        tensor predictions = model.forward(inputs);
        // for(std::size_t i = 0; i<predictions.size(); ++i){
        //     if(!std::isfinite(predictions.data_at(i))){
        //         printf("Non-finite prediction at %zu, index %zu: %f\n", epoch, i, predictions.data_at(i));
        //         return 1;
        //     }
        // }

        tensor loss = mse_loss(predictions, targets);
        loss.backward();
        optimizer.step();
        if(epoch == 1){
        printf("Epoch %zu | Loss: %.6f\n", epoch, loss.data_at(0));
        }
        else if(epoch == 999){
            printf("Epoch %zu | Loss: %.6f\n", epoch, loss.data_at(0));
        }
        
    }

    // Creating Testing Data Set

    // x_test
    int num_test = samples / 4;
    tensor x_test = tensor({num_test, 2});
    for (std::size_t i = 0; i < num_test * 2; ++i)
    {
        x_test.data_at(i) = distrib(gen);
    }

    // y_test
    tensor y_test = tensor({num_test, 1});
    for (std::size_t i = 0; i < num_test; ++i)
    {
        int a = static_cast<int>(x_test.data_at(2 * i));
        int b = static_cast<int>(x_test.data_at(2 * i + 1));
        y_test.data_at(i) = static_cast<float>(a ^ b);
    }

    tensor test_pred = model.forward(x_test);
    

    float correct = 0;
    for (std::size_t i = 0; i < num_test; ++i)
    {
        int predicted = test_pred.data_at(i) >= 0.5f ? 1 : 0;
        int actual = y_test.data_at(i);

        if(predicted == actual){
            correct+=1;
        }
    }

    printf("%% correct: %f", (correct / num_test) * 100);
}