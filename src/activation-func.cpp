#include "activation-func.h"
#include "loss-func.h"

#include <algorithm>

namespace nnet {
ActivationFunction::ActivationFunction(Function0 function, Function1 derivative)
    : func0_(std::move(function)), func1_(std::move(derivative)) {
    assert(func1_(Vector::Zero(4)).rows() == func1_(Vector::Zero(4)).cols() &&
           "derivative should be a square matrix");
    assert(func0_(Vector::Zero(4)).rows() == 4 &&
           "activation function shouldn't change dimensions");
};

ActivationFunction::Vector ActivationFunction::Apply0(const Vector &v) const {
    return func0_(v);
}
ActivationFunction::Matrix ActivationFunction::Derivative(const Vector &v) const {
    return func1_(v);
}

namespace {
ActivationFunction::Vector ReLuApply(const ActivationFunction::Vector &v) {
    return v.unaryExpr([](ActivationFunction::Scalar x) { return std::max(0.0, x); });
}

ActivationFunction::Matrix ReLuDerivative(const ActivationFunction::Vector &x) {
    return x
        .unaryExpr([](ActivationFunction::Scalar x) -> ActivationFunction::Scalar { return x > 0; })
        .asDiagonal();
}

ActivationFunction::Vector IdApply(const ActivationFunction::Vector &v) {
    return v;
}

ActivationFunction::Matrix IdDerivative(const ActivationFunction::Vector &x) {
    return ActivationFunction::Matrix::Identity(x.rows(), x.rows());
}

}  // namespace

ActivationFunction ActivationFunction::ReLu() {
    return ActivationFunction(ReLuApply, ReLuDerivative);
};
ActivationFunction ActivationFunction::Id() {
    return ActivationFunction(IdApply, IdDerivative);
}

}  // namespace nnet
