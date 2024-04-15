#pragma once

#include "activation-func.h"
#include "layer.h"
#include "loss-func.h"

#include <initializer_list>

namespace nnet {

class Net {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Index = Eigen::Index;

    Net() = default;
    Net(std::initializer_list<Index>, std::initializer_list<ActivationFunction>);

    void Train(const std::vector<Vector>& data, const std::vector<Vector>& ans, LossFunction loss,
               int epochs, double tol = 1e-8, bool debug = false);

    Vector Predict(const Vector& v) const;

    friend std::ostream& operator<<(std::ostream& stream, const Net& net);
    friend std::istream& operator>>(std::istream& stream, Net& net);

private:
    struct Adam {
        Scalar alpha = 0.001;
        Scalar beta1 = 0.9;
        Scalar beta2 = 0.999;
        Scalar eps = 1e-8;

        std::vector<Matrix> m_a;
        std::vector<Vector> m_b;
        std::vector<Matrix> v_a;
        std::vector<Vector> v_b;
    };

    Adam InitAdam();
    void Forward(const Vector& inp, std::vector<Vector>* outputs);
    void Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                  Adam* optimizer);
    void Update(const Adam& optimizer, long long time);

private:
    std::vector<Layer> layers_;
};

}  // namespace nnet
