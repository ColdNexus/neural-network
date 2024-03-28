#pragma once

#include <Eigen/Dense>

#include <functional>

namespace nnet {
class ActivationFunction {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Signature0 = Vector(const Vector&);
    using Signature1 = Matrix(const Vector&);
    using Function0 = std::function<Signature0>;
    using Function1 = std::function<Signature1>;

    ActivationFunction(Function0 function, Function1 derivative);

    Vector Apply0(const Vector& v) const;
    Matrix Derivative(const Vector& v) const;

    static ActivationFunction ReLu();
    static ActivationFunction Id();

private:
    Function0 func0_;
    Function1 func1_;
};

constexpr auto ReLu = ActivationFunction::ReLu;  // NOLINT
constexpr auto Id = ActivationFunction::Id;      // NOLINT

}  // namespace nnet
