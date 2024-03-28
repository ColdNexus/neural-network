#pragma once

#include "activation-func.h"
#include "layer.h"
#include "loss-func.h"

#include <initializer_list>

namespace nnet {

class Net {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Index = Eigen::Index;

    Net(std::initializer_list<Index>, std::initializer_list<ActivationFunction>);

    void Train(const std::vector<Vector>& data, const std::vector<Vector>& ans, LossFunction loss,
               int epochs);

    Vector Predict(const Vector& v) const;

private:
    static constexpr Scalar kDefaultStep = 0.0005;

private:
    std::vector<Layer> layers_;
};

}  // namespace nnet
