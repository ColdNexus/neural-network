#include "test-nnet.h"
#include "activation-func.h"
#include "loss-func.h"
#include "net.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>

using Vector = nnet::Net::Vector;
using TimePoint = std::chrono::system_clock::time_point;

namespace nnet {
namespace {

auto Now() {
    return std::chrono::high_resolution_clock::now();
}

std::filesystem::path cwd = std::filesystem::current_path();
std::string           pretrained = cwd.string() + "/../tests/pretrained/";
std::string           mnist_dir = cwd.string() + "/../tests/mnist/";
std::string           data_dir = cwd.string() + "/../tests/train_data/";

std::string mnist_params = pretrained + "mnist_params.txt";
std::string mnist_data_train = mnist_dir + "mnist_data.txt";
std::string mnist_ans_train = mnist_dir + "mnist_ans.txt";
std::string mnist_data_test = mnist_dir + "mnist_data_test.txt";
std::string mnist_ans_test = mnist_dir + "mnist_ans_test.txt";

std::string linear_train = data_dir + "train_linear";
std::string linear_ans = data_dir + "ans_linear";
std::string linear_test = data_dir + "test_linear";
std::string linear_test_ans = data_dir + "test_ans_linear";

class Printer {
public:
    Printer(std::string name) : name_(std::move(name)), time_(Now()) {
        std::cout << "____" << name_ << "___STARTED______\n";
    };

    ~Printer() {
        std::cout << "____" << name_ << "___FINISHED_____\n";
        std::cout << "TIME: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(Now() - time_) << '\n';
        std::cout << '\n';
    }

private:
    std::string name_;
    TimePoint   time_;
};
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
    std::cout << "Tests started\n";
    TestLinear();
    // TestMnist();
    std::cout << "Tests finished\n";
}

void TestLinear() {
    Net                 net = TrainLinear();
    std::vector<Vector> x = ReadData(linear_test);
    std::vector<Vector> ans = ReadData(linear_test_ans);

    LossFunction loss = MSE();
    double       acc = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        // std::cout << net.Predict(x[i]) << ' ' << ans[i] << '\n';
        acc += loss.Dist(net.Predict(x[i]), ans[i]);
    }
    std::cout << "TestLinear accuracy: " << acc / x.size() << '\n';
    std::cout << '\n';
}

Net TrainLinear() {
    std::vector<Vector> x = ReadData(linear_train);
    std::vector<Vector> ans = ReadData(linear_ans);
    Net                 net({5, 5, 1}, {ReLu(), Id()});
    int                 epochs = 200;
    {
        Printer p("TrainLinear 5 -> 5 -> 1, {ReLu, Id}, " + std::to_string(epochs) + " epochs");
        net.Train(x, ans, MSE(), epochs, 1e-8, false);
    }
    return net;
}

void TestMnist() {
    Net net = TrainMnist();
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

    // Net net({784, 400, 120, 10}, {nnet::ReLu(), nnet::ReLu(), nnet::SoftMax()});
    net.Train(x, y, nnet::MSE(), 200);

    std::ofstream file("mnist_params.txt");
    file << net;
    return net;
}
}  // namespace nnet
