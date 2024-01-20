#include "activation-func.h"
#include <algorithm>
#include "loss-func.h"

namespace nnet {
ActivationFunction::ActivationFunction(std::function<ScalarSignature> function,
                                       std::function<ScalarSignature> derivative,
                                       std::function<VectorSignature> vector_function,
                                       std::function<VectorSignature> vector_derivative)
    : function_(function),
      derivative_(derivative),
      vector_function_(vector_function),
      vector_derivative_(vector_derivative){};

ActivationFunction::Scalar ActivationFunction::Apply(Scalar x) {
    return function_(x);
}
ActivationFunction::Scalar ActivationFunction::ApplyDerivative(Scalar x) {
    return derivative_(x);
}

ActivationFunction::Vector ActivationFunction::Apply(const Vector& v) {
    return vector_function_(v);
}
ActivationFunction::Vector ActivationFunction::ApplyDerivative(const Vector& v) {
    return vector_derivative_(v);
}

namespace {
ActivationFunction::Scalar ReLuApply(ActivationFunction::Scalar x) {
    return std::max<ActivationFunction::Scalar>(0, x);
}
ActivationFunction::Scalar ReLuApplyDerivative(ActivationFunction::Scalar x) {
    return x > 0;
}
ActivationFunction::Vector ReLuVectorApply(const ActivationFunction::Vector& v) {
    ActivationFunction::Vector ret(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        ret[i] = ReLuApply(v[i]);
    }
    return ret;
}
ActivationFunction::Vector ReLuVectorApplyDerivative(const ActivationFunction::Vector& v) {
    ActivationFunction::Vector ret(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        ret[i] = ReLuApplyDerivative(v[i]);
    }
    return ret;
}
}  // namespace

ActivationFunction ReLu{ReLuApply, ReLuApplyDerivative, ReLuVectorApply, ReLuVectorApplyDerivative};
}  // namespace nnet