#include "loss-func.h"

namespace nnet {

LossFunction::LossFunction(std::function<DistSignature>     dist,
                           std::function<GradientSignature> gradient)
    : dist_(std::move(dist)), gradient_(std::move(gradient)){};

LossFunction::Scalar LossFunction::Dist(const Vector& actual, const Vector& ideal) const {
    assert(actual.rows() == ideal.rows() && "size of rows are incorrect");
    assert(actual.cols() == ideal.cols() && actual.cols() == 1 && "size of cols are incorrect");
    return dist_(actual, ideal);
}

LossFunction::VectorT LossFunction::Gradient(const Vector& actual, const Vector& ideal) const {
    assert(actual.rows() == ideal.rows() && "size of rows are incorrect");
    assert(actual.cols() == ideal.cols() && actual.cols() == 1 && "size of cols are incorrect");
    return gradient_(actual, ideal);
}

namespace {
LossFunction::Scalar MSEDist(const LossFunction::Vector& actual,
                             const LossFunction::Vector& ideal) {
    return (actual - ideal).squaredNorm();
};

LossFunction::VectorT MSEGradient(const LossFunction::Vector& actual,
                                  const LossFunction::Vector& ideal) {
    return 2 * (actual - ideal).transpose();
}
}  // namespace

LossFunction LossFunction::MSE() {
    return LossFunction(MSEDist, MSEGradient);
}

}  // namespace nnet
