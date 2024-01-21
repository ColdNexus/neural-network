#include "layer.h"

#include <cassert>
#include <iostream>
#include "activation-func.h"

namespace nnet {
Layer::Layer(unsigned rows, unsigned cols, ActivationFunction sigma) : sigma_(std::move(sigma)) {
    a_ = distribution.template generate<Matrix>(rows, cols, generator);
    b_ = distribution.template generate<Vector>(rows, 1, generator);
}

Layer::Vector Layer::Calculate(const Vector& x) {
    assert(x.rows() == a_.cols() && "wrong vector size");
    return sigma_.Apply(a_ * x + b_);
}

unsigned Layer::MatrixRows() {
    return a_.rows();
}
unsigned Layer::MatrixCols() {
    return a_.cols();
}
unsigned Layer::VecSize() {
    return b_.size();
}

Layer::Matrix Layer::EvaluateMatrixModification(const VectorT& u, const Vector& x) {
    return (x * u * sigma_.ApplyDerivative(a_ * x + b_).asDiagonal()).transpose();
}

Layer::Vector Layer::EvaluateVectorModification(const VectorT& u, const Vector& x) {
    return sigma_.ApplyDerivative(a_ * x + b_).asDiagonal() * u.transpose();
}
Layer::VectorT Layer::Propagate(const VectorT& u, const Vector& x) {
    return u * sigma_.ApplyDerivative(a_ * x + b_).asDiagonal() * a_;
}

void Layer::UpdateMatrix(const Matrix& m) {
    a_ += m;
}
void Layer::UpdateVector(const Vector& v) {
    b_ += v;
}
}  // namespace nnet