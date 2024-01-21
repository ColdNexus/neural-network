#pragma once

#include <Eigen/Dense>
#include <functional>

namespace nnet {
class ActivationFunction {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ScalarSignature = Scalar(Scalar);
    using VectorSignature = Vector(const Vector&);

    ActivationFunction(std::function<ScalarSignature> function,
                       std::function<ScalarSignature> derivative,
                       std::function<VectorSignature> vector_function,
                       std::function<VectorSignature> vector_derivative);

    Scalar Apply(Scalar x);
    Scalar ApplyDerivative(Scalar x);

    Vector Apply(const Vector& v);
    Vector ApplyDerivative(const Vector& v);

private:
    std::function<ScalarSignature> function_;
    std::function<ScalarSignature> derivative_;
    std::function<VectorSignature> vector_function_;
    std::function<VectorSignature> vector_derivative_;
};

extern ActivationFunction ReLu;  // NOLINT
extern ActivationFunction Id;    // NOLINT
}  // namespace nnet