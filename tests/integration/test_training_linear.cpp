#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "mini_torch/core/tensor.h"
#include "mini_torch/nn/sequential.h"
#include "mini_torch/optim/sgd.h"
#include "mini_torch/nn/linear.h"
#include "mini_torch/nn/relu.h"
#include "mini_torch/nn/loss.h"

std::vector<std::array<float, 4>> load_data(const std::string filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Error: Could not open file" + filename);
    }

    std::string line;
    std::vector<std::array<float, 4>> content;

    std::getline(file, line);
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string value;
        std::array<float, 4> row;
        int col_idx = 0;
        while (std::getline(ss, value, ','))
        {
            if (col_idx < 4)
            {
                try
                {
                    row[col_idx] = std::stof(value);
                    col_idx++;
                }

                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Warning: invalid float found: " << value << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "Warning: number out of range: " << value << std::endl;
                }
            }
        }

        if (col_idx == 4)
        {
            content.push_back(row);
        }
        else
        {
            std::cerr << "Warning: row skipped due to incorrect column count" << std::endl;
        }
    }

    file.close();
    return content;
}

int main()
{

    std::vector<std::array<float, 4>> data = load_data("./data/linear_dataset.csv");
    int num_train = data.size() * 0.8;
    tensor x_train = tensor({static_cast<int>(num_train), 3}, true);
    tensor y_train = tensor({static_cast<int>(num_train), 1});

    for (std::size_t i = 0; i < num_train; ++i)
    {
        x_train.data_at(i * 3) = data[i][0];
        x_train.data_at(i * 3 + 1) = data[i][1];
        x_train.data_at(i * 3 + 2) = data[i][2];
        y_train.data_at(i) = data[i][3];
    }

    int num_test = static_cast<int>(data.size()) - num_train;

    tensor x_test = tensor({num_test, 3});
    tensor y_test = tensor({num_test, 1});

    for (std::size_t i = 0; i < num_test; ++i)
    {
        x_test.data_at(i * 3) = data[i + num_train][0];
        x_test.data_at(i * 3 + 1) = data[i + num_train][1];
        x_test.data_at(i * 3 + 2) = data[i + num_train][2];
        y_test.data_at(i) = data[i + num_train][3];
    }

    sequential model = sequential();
    model.add(std::make_unique<linear>(3, 5));
    model.add(std::make_unique<relu>());
    model.add(std::make_unique<linear>(5, 1));
    sgd optimizer = sgd(model.parameters(), 0.01f);

    int epochs = 5000;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // printf("Epoch: %zu", epoch);
        // printf("\n");
        optimizer.zero_grad();
        tensor predictions = model.forward(x_train);
        tensor loss = mse_loss(predictions, y_train);
        loss.backward();
        optimizer.step();
    }

    tensor test_preds = model.forward(x_test);
    
    printf("MSE on test dataset: %f", mse_loss(test_preds, y_test).data_at(0));

    return 0;
}