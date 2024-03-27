#pragma once

#include <Eigen/Dense>
#include <functional>

namespace nnet {
class ActivationFunction {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Signature0 = Scalar(Scalar);
    using Signature1 = Matrix(const Vector&);
    using Function0 = std::function<Signature0>;
    using Function1 = std::function<Signature1>;

    ActivationFunction(Function0 function, Function1 derivative);

    Vector Apply0(const Vector& v);
    Matrix Derivative(const Vector& v);

private:
    Function0 func0_;
    Function1 func1_;
};

extern ActivationFunction ReLu;  // NOLINT
extern ActivationFunction Id;    // NOLINT
}  // namespace nnet
