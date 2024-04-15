#pragma once

#include <Eigen/Dense>

#include <functional>
#include <ostream>
#include <unordered_map>
#include <string>

namespace nnet {
class ActivationFunction {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Signature0 = Vector(const Vector&);
    using Signature1 = Matrix(const Vector&);
    using Function0 = std::function<Signature0>;
    using Function1 = std::function<Signature1>;

    enum class Names { ReLu, Id, SoftMax };

    ActivationFunction() = default;
    ActivationFunction(Function0 function, Function1 derivative, Names name);

    Vector Apply0(const Vector& v) const;
    Matrix Derivative(const Vector& v) const;

    bool IsInitialized() const;

    static ActivationFunction ReLu();
    static ActivationFunction Id();
    static ActivationFunction SoftMax();

    static ActivationFunction NameToAF(Names name);

    bool                 operator==(const ActivationFunction& af) const;
    friend std::ostream& operator<<(std::ostream& stream, const ActivationFunction& af);
    friend std::istream& operator>>(std::istream& stream, ActivationFunction& af);

private:
    Function0 func0_;
    Function1 func1_;

public:
    Names name_;
};

constexpr auto ReLu = ActivationFunction::ReLu;        // NOLINT
constexpr auto Id = ActivationFunction::Id;            // NOLINT
constexpr auto SoftMax = ActivationFunction::SoftMax;  // NOLINT

}  // namespace nnet
