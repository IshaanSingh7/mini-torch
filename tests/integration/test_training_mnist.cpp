
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include "mini_torch/core/tensor.h"
#include "mini_torch/nn/sequential.h"
#include "mini_torch/nn/linear.h"
#include "mini_torch/nn/relu.h"
#include "mini_torch/optim/sgd.h"
#include "mini_torch/nn/loss.h"
#include <chrono>
#include "mini_torch/core/tensor_impl.h"
#include "mini_torch/autograd/node.h"
// Read a 32-bit big-endian integer from the IDX header
std::uint32_t read_int(std::ifstream &file)
{
    unsigned char bytes[4];

    if (!file.read(reinterpret_cast<char *>(bytes), 4))
        throw std::runtime_error("Failed to read IDX header");

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

tensor load_images(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::uint32_t magic = read_int(file);
    std::uint32_t num_images = read_int(file);
    std::uint32_t rows = read_int(file);
    std::uint32_t cols = read_int(file);

    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST image file");

    tensor images({static_cast<int>(num_images),
                   static_cast<int>(rows * cols)});

    for (std::size_t i = 0; i < images.size(); ++i)
    {
        unsigned char pixel;

        if (!file.read(reinterpret_cast<char *>(&pixel), 1))
            throw std::runtime_error("Incomplete MNIST image data");

        images.data_at(i) = static_cast<float>(pixel) / 255.0f;
    }

    return images;
}

tensor load_labels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::uint32_t magic = read_int(file);
    std::uint32_t num_labels = read_int(file);

    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label file");

    tensor labels({static_cast<int>(num_labels)});

    for (std::size_t i = 0; i < labels.size(); ++i)
    {
        unsigned char label;

        if (!file.read(reinterpret_cast<char *>(&label), 1))
            throw std::runtime_error("Incomplete MNIST label data");

        if (label > 9)
            throw std::runtime_error("Invalid MNIST label");

        labels.data_at(i) = static_cast<float>(label);
    }

    return labels;
}

int main()
{
    tensor x_train = load_images("data/mnist/train-images.idx3-ubyte");
    tensor y_train = load_labels("data/mnist/train-labels.idx1-ubyte");
    tensor y_train_one_hot = one_hot(y_train, 10);

    tensor x_test = load_images("data/mnist/t10k-images.idx3-ubyte");
    tensor y_test = load_labels("data/mnist/t10k-labels.idx1-ubyte");
    // tensor y_test_one_hot = one_hot(y_test, 10);

    if (x_train.shape()[0] != y_train.shape()[0] ||
        x_test.shape()[0] != y_test.shape()[0])
    {
        throw std::runtime_error("Image and label counts do not match");
    }

    std::cout << "MNIST loaded successfully!" << std::endl;
    fflush(stdout);
    std::cout << "Training samples: " << x_train.shape()[0] << std::endl;
    fflush(stdout);
    std::cout << "Testing samples: " << x_test.shape()[0] << std::endl;
    fflush(stdout);
    std::cout << "Features per image: " << x_train.shape()[1] << std::endl;
    fflush(stdout);

    sequential model = sequential();
    model.add(std::make_unique<linear>(784, 128));
    model.add(std::make_unique<relu>());
    model.add(std::make_unique<linear>(128, 64));
    model.add(std::make_unique<relu>());
    model.add(std::make_unique<linear>(64, 10));

    sgd optimizer = sgd(model.parameters(), 0.1);

    int num_epochs = 20;
    int batch_size = 64;
    int num_batches = (x_train.shape()[0] + batch_size - 1) / batch_size; // rounded up
    std::vector<std::size_t> indices(x_train.shape()[0]);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());

    for (std::size_t epoch = 0; epoch < num_epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), gen);

        for (std::size_t batch = 0; batch < num_batches; ++batch)
        {

            {

                std::size_t start = batch * batch_size;
                std::size_t end = std::min(start + batch_size, indices.size());

                std::vector<std::size_t> batch_indices(
                    indices.begin() + start,
                    indices.begin() + end);

                tensor x_batch = x_train.select_rows(batch_indices);
                tensor y_batch = y_train_one_hot.select_rows(batch_indices);

                optimizer.zero_grad();

                tensor predictions = model.forward(x_batch);
                tensor loss = mse_loss(predictions, y_batch);

                loss.backward();

                optimizer.step();
            }
            // if (batch % 200 == 0)
            // {
            //     std::cout << "Batch " << batch << "\n";

            //     std::cout << "Live tensors: "
            //               << tensor_impl::live_count.load() << "\n";

            //     std::cout << "Live nodes: "
            //               << node::live_count.load() << "\n\n";
            // }
        }

        std::cout << "Completed Epoch " << epoch << "\n";
    }

    // TESTING
    tensor test_preds = model.forward(x_test);

    std::vector<float> maxes;
    for (std::size_t row = 0; row < test_preds.shape()[0]; row++)
    {
        std::vector<float> preds;
        for (std::size_t col = 0; col < 10; col++)
        {
            preds.push_back(test_preds.data_at(row * 10 + col));
        }
        auto max_it = std::max_element(preds.begin(), preds.end());
        int max_index = std::distance(preds.begin(), max_it);
        maxes.push_back(max_index);
    }

    float total = 0.0f;

    for (std::size_t i = 0; i < y_test.size(); ++i)
    {
        if (maxes[i] == y_test.data_at(i))
        {
            total++;
        }
    }

    printf("Percent Correct: %f", 100 * total / y_test.size());

    return 0;
}
