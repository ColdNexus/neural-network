#pragma once

#include <functional>
#include <Eigen/Dense>

namespace nnet {

class LossFunction {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using DistSignature = Scalar(const Vector&, const Vector&);
    using GradientSignature = VectorT(const Vector&, const Vector&);

    LossFunction(std::function<DistSignature> dist, std::function<GradientSignature> gradient);

    Scalar Dist(const Vector& actual, const Vector& ideal);
    VectorT Gradient(const Vector& actual, const Vector& ideal);

private:
    std::function<DistSignature> dist_;
    std::function<GradientSignature> gradient_;
};

extern LossFunction MSE;  // NOLINT

}  // namespace nnet
