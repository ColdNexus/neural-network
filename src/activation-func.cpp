#include "activation-func.h"
#include "loss-func.h"

#include <algorithm>

namespace nnet {
ActivationFunction::ActivationFunction(Function0 function, Function1 derivative, Names name)
    : func0_(std::move(function)), func1_(std::move(derivative)), name_(name) {
    assert(func1_(Vector::Zero(4)).rows() == func1_(Vector::Zero(4)).cols() &&
           "derivative should be a square matrix");
    assert(func0_(Vector::Zero(4)).rows() == 4 &&
           "activation function shouldn't change dimensions");
};

ActivationFunction::Vector ActivationFunction::Apply0(const Vector &v) const {
    assert(IsInitialized() && "Activation function is NOT initialized");
    return func0_(v);
}
ActivationFunction::Matrix ActivationFunction::Derivative(const Vector &v) const {
    assert(IsInitialized() && "Activation function is NOT initialized");
    return func1_(v);
}

bool ActivationFunction::IsInitialized() const {
    return static_cast<bool>(func0_);
}

static ActivationFunction NameToAF(ActivationFunction::Names name) {
    switch (name) {
        case ActivationFunction::Names::ReLu:
            return nnet::ReLu();
        case ActivationFunction::Names::Id:
            return nnet::Id();
    }
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

template <typename Enumeration>
typename std::underlying_type<Enumeration>::type AsInteger(Enumeration const value) {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

}  // namespace

ActivationFunction ActivationFunction::ReLu() {
    return ActivationFunction(ReLuApply, ReLuDerivative, Names::ReLu);
};
ActivationFunction ActivationFunction::Id() {
    return ActivationFunction(IdApply, IdDerivative, Names::Id);
}

bool ActivationFunction::operator==(const ActivationFunction &af) const {
    return name_ == af.name_;
}

std::ostream &operator<<(std::ostream &stream, const ActivationFunction &af) {
    stream << AsInteger(af.name_);
    return stream;
}

std::istream &operator>>(std::istream &stream, ActivationFunction &af) {
    int x = 0;
    stream >> x;
    af = NameToAF(ActivationFunction::Names{x});
    return stream;
}

}  // namespace nnet
