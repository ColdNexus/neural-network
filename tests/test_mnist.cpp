#include <fstream>
#include <filesystem>
#include <iostream>
#include "activation-func.h"
#include "loss-func.h"
#include "net.h"

using Vector = nnet::Net::Vector;

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

int main() {
    auto data = ReadData("mnist_data.txt");
    auto ans = ReadData("mnist_ans.txt");

    int inp_size = data.back().rows();
    int outp_size = ans.back().rows();

    nnet::Net net({inp_size, inp_size * 2, inp_size / 2, outp_size},
                  {nnet::ReLu(), nnet::ReLu(), nnet::Id()});
    net.Train(data, ans, nnet::MSE(), 12);

    std::ofstream params("params.txt");
    params << net;

    auto data_test = ReadData("mnist_data_test.txt");
    auto ans_test = ReadData("mnist_ans_test.txt");

    std::cout << "________________\n";

    for (int i = 0; i < data.size(); ++i) {
        std::cout << "prediction: " << net.Predict(data[i]) << " correct: " << ans[i] << '\n';
    }
}
