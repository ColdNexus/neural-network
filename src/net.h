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

    struct TrainData {
        Vector data;
        Vector ans;
    };

    Net() = default;
    Net(std::initializer_list<Index>, std::initializer_list<ActivationFunction>);

    void TrainVanilla(const std::vector<TrainData>& data, LossFunction loss, int epochs,
                      double tol = 1e-8);
    void TrainSGD(const std::vector<TrainData>& data, LossFunction loss, int epochs,
                  int batch_size = 8, double tol = 1e-8);
    void TrainAdam(std::vector<TrainData>& data, LossFunction loss, int epochs, int batch_size = 8,
                   double tol = 1e-8);

    Vector Predict(const Vector& v) const;

    friend std::ostream& operator<<(std::ostream& stream, const Net& net);
    friend std::istream& operator>>(std::istream& stream, Net& net);

private:
    void Forward(const Vector& inp, std::vector<Vector>* outputs);

private:
    std::vector<Layer> layers_;
};

}  // namespace nnet
