#include "layer.h"
#include "activation-func.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <ostream>
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
    assert(IsInitialized() && "layer is NOT initialized");
    assert(x.rows() == a_.cols() && "wrong vector size");
    return sigma_.Apply0(a_ * x + b_);
}

Layer::Index Layer::InSize() const {
    assert(IsInitialized() && "layer is NOT initialized");
    return a_.cols();
}
Layer::Index Layer::OutSize() const {
    assert(IsInitialized() && "layer is NOT initialized");
    return a_.rows();
}

Layer::Matrix Layer::GetDa(const VectorT& u, const Vector& x) const {
    assert(IsInitialized() && "layer is NOT initialized");
    assert(x.cols() == u.rows() && "cannot multiply, wrong size");
    assert(u.cols() == sigma_.Derivative(a_ * x + b_).rows() && "cannot multiply, wrong size");
    return (x * u * sigma_.Derivative(a_ * x + b_)).transpose();
}

Layer::Vector Layer::GetDb(const VectorT& u, const Vector& x) const {
    assert(IsInitialized() && "layer is NOT initialized");
    assert(sigma_.Derivative(a_ * x + b_).cols() == u.cols() && "cannot multiply, wrong size");
    return (u * sigma_.Derivative(a_ * x + b_)).transpose();
}

Layer::VectorT Layer::Propagate(const VectorT& u, const Vector& x) const {
    assert(IsInitialized() && "layer is NOT initialized");
    assert(u.cols() == sigma_.Derivative(a_ * x + b_).rows() && "cannot multiply, wrong size");
    assert(u.cols() == a_.rows() && "cannot multiply, wrong size");
    return u * sigma_.Derivative(a_ * x + b_) * a_;
}

void Layer::UpdateA(const Matrix& m, Scalar rate) {
    assert(IsInitialized() && "layer is NOT initialized");
    assert(a_.rows() == m.rows() && "cannot update, wrong dimensions");
    assert(a_.cols() == m.cols() && "cannot update, wrong dimensions");
    a_ += rate * m;
}
void Layer::UpdateB(const Vector& v, Scalar rate) {
    assert(IsInitialized() && "layer is NOT initialized");
    assert(b_.rows() == v.rows() && "cannot update, wrong dimensions");
    b_ += rate * v;
}

bool Layer::IsInitialized() const {
    return sigma_.IsInitialized();
}

std::ostream& operator<<(std::ostream& stream, const Layer& layer) {
    stream << std::setprecision(std::numeric_limits<Layer::Scalar>::digits10 + 2);
    stream << layer.InSize() << ' ' << layer.OutSize() << '\n';
    stream << layer.a_ << '\n';
    stream << layer.b_ << '\n';
    stream << layer.sigma_;
    return stream;
}

std::istream& operator>>(std::istream& stream, Layer& layer) {
    Layer::Index in_size, out_size;
    stream >> in_size >> out_size;
    layer.a_ = Layer::Matrix(out_size, in_size);
    for (Layer::Index i = 0; i < out_size; ++i) {
        for (Layer::Index j = 0; j < in_size; ++j) {
            stream >> layer.a_(i, j);
        }
    }
    layer.b_ = Layer::Vector(out_size);
    for (Layer::Index i = 0; i < out_size; ++i) {
        stream >> layer.b_(i);
    }
    stream >> layer.sigma_;
    return stream;
}

}  // namespace nnet
