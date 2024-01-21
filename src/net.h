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
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

    Net(std::initializer_list<unsigned>, std::initializer_list<ActivationFunction>,
        LossFunction loss);

    void Train(const std::vector<Vector>& data, const std::vector<Vector>& ans, int epochs);
    Vector Predict(const Vector& v);

private:
    std::vector<Layer> layers_;
    LossFunction loss_;

    static constexpr Scalar kStep = 0.0005;
};

}  // namespace nnet
