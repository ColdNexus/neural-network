#pragma once

#include "loss-func.h"
#include "layer.h"

#include <Eigen/Dense>

namespace nnet {
class IOptimizer {
public:
    using Scalar = double;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
    using Index = Eigen::Index;

    virtual void Backward(const std::vector<Vector> outputs, const Vector& ans,
                          const LossFunction& loss, int batch_size, const std::vector<Layer>&) = 0;
    virtual void UpdateParams(std::vector<Layer>* layers) = 0;
    virtual void ZeroGrads() = 0;

private:
};

class Adam : IOptimizer {
public:
    Adam(const std::vector<Layer>& layers);
    void Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                  int batch_size, const std::vector<Layer>&) override;
    void UpdateParams(std::vector<Layer>* layers) override;
    void ZeroGrads() override;

private:
    Scalar alpha_ = 0.001;
    Scalar beta1_ = 0.9;
    Scalar beta2_ = 0.999;
    Scalar eps_ = 1e-8;
    Scalar time_ = 1;

    std::vector<Matrix> m_a_;
    std::vector<Vector> m_b_;
    std::vector<Matrix> v_a_;
    std::vector<Vector> v_b_;
};

class SGD : IOptimizer {
public:
    SGD(const std::vector<Layer>& layers);
    void Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                  int batch_size, const std::vector<Layer>&) override;
    void UpdateParams(std::vector<Layer>* layers) override;
    void ZeroGrads() override;

private:
    Scalar rate_ = 0.05;

    std::vector<Matrix> das_;
    std::vector<Vector> dbs_;
};

}  // namespace nnet
