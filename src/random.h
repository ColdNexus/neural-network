#pragma once

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace nnet {
class Random {
public:
    using Scalar = double;
    using Generator = Eigen::Rand::P8_mt19937_64;
    using Distribution = Eigen::Rand::NormalGen<Scalar>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index = Eigen::Index;

    Matrix RandomMatrix(Index rows, Index cols);
    Vector RandomVector(Index size);

private:
    static constexpr int       kSeed = 42;
    inline static Generator    generator{kSeed};
    inline static Distribution distribution{0, 1};
};
}  // namespace nnet
