#include "layer.h"
#include "activation-func.h"

#include <cassert>
#include <iostream>
#include <random>

namespace nnet {

Layer::Layer(Index rows, Index cols, ActivationFunction sigma)
    : a_(Layer::Rand().RandomMatrix(rows, cols)),
      b_(Layer::Rand().RandomVector(rows)),
      sigma_(std::move(sigma)) {
    assert(rows >= 0 && "rows should be non negative");
    assert(cols >= 0 && "cols should be non negative");
}

Layer::Vector Layer::Calculate(const Vector& x) const {
    assert(x.rows() == a_.cols() && "wrong vector size");
    return sigma_.Apply0(a_ * x + b_);
}

Layer::Index Layer::InSize() const {
    return a_.cols();
}
Layer::Index Layer::OutSize() const {
    return a_.rows();
}

Layer::Matrix Layer::GetDa(const VectorT& u, const Vector& x) const {
    assert(x.cols() == u.rows() && "cannot multiply, wrong size");
    assert(u.cols() == sigma_.Derivative(a_ * x + b_).rows() && "cannot multiply, wrong size");
    return (x * u * sigma_.Derivative(a_ * x + b_)).transpose();
}

Layer::Vector Layer::GetDb(const VectorT& u, const Vector& x) const {
    assert(sigma_.Derivative(a_ * x + b_).cols() == u.cols() && "cannot multiply, wrong size");
    return sigma_.Derivative(a_ * x + b_) * u.transpose();
}
Layer::VectorT Layer::Propagate(const VectorT& u, const Vector& x) const {
    assert(u.cols() == sigma_.Derivative(a_ * x + b_).rows() && "cannot multiply, wrong size");
    assert(u.cols() == a_.rows() && "cannot multiply, wrong size");
    return u * sigma_.Derivative(a_ * x + b_) * a_;
}

void Layer::UpdateA(const Matrix& m, Scalar rate) {
    assert(a_.rows() == m.rows() && "cannot update, wrong dimensions");
    assert(a_.cols() == m.cols() && "cannot update, wrong dimensions");
    a_ += rate * m;
}
void Layer::UpdateB(const Vector& v, Scalar rate) {
    assert(b_.rows() == v.rows() && "cannot update, wrong dimensions");
    b_ += rate * v;
}

}  // namespace nnet
