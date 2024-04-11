#pragma once

#include "activation-func.h"
#include "random.h"

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

namespace nnet {
class Layer {
public:
    using Scalar = double;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Index = Eigen::Index;


    explicit Layer() = default;
    Layer(Index input_size, Index ouput_size, ActivationFunction sigma);

    Vector  Calculate(const Vector& x) const;
    Matrix  GetDa(const VectorT& u, const Vector& x) const;
    Vector  GetDb(const VectorT& u, const Vector& x) const;
    VectorT Propagate(const VectorT& u, const Vector& x) const;

    void UpdateA(const Matrix& m, Scalar rate);
    void UpdateB(const Vector& v, Scalar rate);

    Index InSize() const;
    Index OutSize() const;

    bool IsInitialized() const;

    friend std::ostream& operator<<(std::ostream& stream, const Layer& layer);
    friend std::istream& operator>>(std::istream& stream, Layer& layer);

private:
    static Random& Rand() {
        static Random r;
        return r;
    }

private:
    Matrix             a_;
    Vector             b_;
    ActivationFunction sigma_;
};

}  // namespace nnet
