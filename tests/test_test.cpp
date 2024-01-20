#include "activation-func.h"
#include "layer.h"
#include "loss-func.h"
#include "net.h"

#include <cassert>
#include <iostream>

int main() {
    nnet::LossFunction::Vector v1{{2, 2, -5}};
    nnet::LossFunction::Vector v2{{2, 2, -2}};
    std::cout << nnet::MSE.Dist(v1, v2) << '\n';

    std::cout << nnet::ReLu.Apply(10) << '\n';
    std::cout << nnet::ReLu.ApplyDerivative(10) << '\n';
    std::cout << nnet::ReLu.Apply(v1) << '\n';
    std::cout << nnet::ReLu.ApplyDerivative(v2) << '\n';

    nnet::Net net({3, 10, 2}, {nnet::ReLu}, nnet::MSE);
    std::cout << net.Predict(v2) << '\n';
    return 0;
}