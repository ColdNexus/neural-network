#pragma once

#include <initializer_list>
#include "activation-func.h"
#include "layer.h"
#include "loss-func.h"

namespace nnet {

class Net {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Net(std::initializer_list<unsigned>, std::initializer_list<ActivationFunction>,
        LossFunction loss);

    void Train();
    Vector Predict(const Vector& v);

private:
    std::vector<Layer> layers_;
    std::vector<ActivationFunction> activ_funcs_;
    LossFunction loss_;
};

}  // namespace nnet
