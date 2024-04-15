#include "test-nnet.h"
#include "activation-func.h"
#include "loss-func.h"
#include "net.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using Vector = nnet::Net::Vector;

namespace nnet {
namespace {
std::filesystem::path cwd = std::filesystem::current_path();
std::string           pretrained = cwd.string() + "/../tests/pretrained/";
std::string           mnist_dir = cwd.string() + "/../tests/mnist/";

std::string mnist_params = pretrained + "mnist_params.txt";
std::string mnist_data_train = mnist_dir + "mnist_data.txt";
std::string mnist_ans_train = mnist_dir + "mnist_ans.txt";
std::string mnist_data_test = mnist_dir + "mnist_data_test.txt";
std::string mnist_ans_test = mnist_dir + "mnist_ans_test.txt";
}  // namespace

std::vector<Vector> ReadData(std::string file_name) {
    std::fstream file(file_name);
    size_t       size = 0;
    size_t       n = 0;
    file >> size >> n;
    std::vector<Vector> data;
    data.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        data.emplace_back(n);
        for (size_t j = 0; j < n; ++j) {
            file >> data.back()(j);
        }
    }
    return data;
}

std::vector<Vector> NumToInd(const std::vector<Vector>& ans, int max_num) {
    std::vector<Vector> ret;
    ret.reserve(ans.size());
    for (int i = 0; i < ans.size(); ++i) {
        ret.emplace_back(Vector::Zero(max_num + 1));
        ret.back()[ans[i](0)] = 1;
    }
    return ret;
}

void RunTests() {
    TestMnist();
    std::cout << "Tests finished\n";
}

void TestMnist() {
    std::cout << "____TEST_MNIST_BEGIN______\n";
    Net net = TrainMnist();
    std::cout << "____TEST_MNIST_END________\n";
}

Net TrainMnist() {

    std::fstream inp(mnist_params);
    Net          net;
    if (inp.is_open()) {
        inp >> net;
        // return net;
    }

    std::vector<Vector> x = ReadData(mnist_data_train);
    std::vector<Vector> ans = ReadData(mnist_ans_train);
    std::vector<Vector> y = NumToInd(ans, 9);

    //Net net({784, 400, 120, 10}, {nnet::ReLu(), nnet::ReLu(), nnet::SoftMax()});
    net.Train(x, y, nnet::MSE(), 200);

    std::ofstream file("mnist_params.txt");
    file << net;
    return net;
}
}  // namespace nnet
