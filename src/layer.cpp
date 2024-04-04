#include "layer.h"
#include "activation-func.h"

#include <cassert>
#include <iostream>
#include <random>

namespace nnet {

Layer::Layer(Index input_size, Index output_size, ActivationFunction sigma)
    : a_(Layer::Rand().RandomMatrix(output_size, input_size)),
      b_(Layer::Rand().RandomVector(output_size)),
      sigma_(std::move(sigma)) {
    assert(output_size >= 0 && "output should be non negative");
    assert(input_size >= 0 && "input should be non negative");
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
