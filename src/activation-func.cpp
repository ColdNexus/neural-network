#include "activation-func.h"
#include "loss-func.h"
#include <algorithm>

namespace nnet {
ActivationFunction::ActivationFunction(Function0 function, Function1 derivative)
    : func0_(std::move(function)), func1_(std::move(derivative)){};

ActivationFunction::Vector ActivationFunction::Apply0(const Vector& v) {
    return v.unaryExpr(func0_);
}
ActivationFunction::Matrix ActivationFunction::Derivative(const Vector& v) {
    return func1_(v);
}

namespace {
ActivationFunction::Scalar ReLuApply(ActivationFunction::Scalar x) {
    return std::max<ActivationFunction::Scalar>(0, x);
}
ActivationFunction::Matrix ReLuDerivative(const ActivationFunction::Vector& x) {
    return x
        .unaryExpr([](ActivationFunction::Scalar x) -> ActivationFunction::Scalar { return x > 0; })
        .asDiagonal();
}

ActivationFunction::Scalar IdApply(ActivationFunction::Scalar x) {
    return x;
}
ActivationFunction::Matrix IdDerivative(const ActivationFunction::Vector& x) {
    return ActivationFunction::Matrix::Identity(x.rows(), x.rows());
}

}  // namespace

ActivationFunction ReLu{ReLuApply, ReLuDerivative};
ActivationFunction Id{IdApply, IdDerivative};
}  // namespace nnet
