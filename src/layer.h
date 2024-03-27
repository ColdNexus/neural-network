#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include "activation-func.h"

namespace nnet {
class Layer {
public:
    using Scalar = double;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Generator = Eigen::Rand::P8_mt19937_64;
    using Distribution = Eigen::Rand::NormalGen<Scalar>;

    Layer(unsigned rows, unsigned cols, ActivationFunction sigma);

    Vector Calculate(const Vector& x);
    Matrix EvaluateMatrixModification(const VectorT& u, const Vector& x);
    Vector EvaluateVectorModification(const VectorT& u, const Vector& x);
    VectorT Propagate(const VectorT& u, const Vector& x);

    void UpdateMatrix(const Matrix& m);
    void UpdateVector(const Vector& v);

    unsigned MatrixRows();
    unsigned MatrixCols();
    unsigned VecSize();

private:
    Matrix a_;
    Vector b_;
    ActivationFunction sigma_;

    static constexpr int kSeed = 42;
    inline static Generator generator{kSeed};
    inline static Distribution distribution{0, 1};
};

}  // namespace nnet
