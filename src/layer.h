#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace nnet {
class Layer {
public:
    using Scalar = double;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Generator = Eigen::Rand::P8_mt19937_64;
    using Distribution = Eigen::Rand::NormalGen<Scalar>;

    Layer(unsigned rows, unsigned cols);

    Vector Calculate(const Vector& x);
    auto EvaluateModification();
    auto Propagate();
    auto UpdateParameters();

private:
    Matrix a_;
    Vector b_;

    static constexpr int kSeed = 42;
    inline static Generator generator{kSeed};
    inline static Distribution distribution{0, 1};
};

}  // namespace nnet