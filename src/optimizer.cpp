#include "optimizer.h"

namespace nnet {
Adam::Adam(const std::vector<Layer>& layers) {
    m_a_.reserve(layers.size());
    v_a_.reserve(layers.size());
    m_b_.reserve(layers.size());
    v_b_.reserve(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
        m_a_.emplace_back(Matrix::Zero(layers[i].OutSize(), layers[i].InSize()));
        m_b_.emplace_back(Vector::Zero(layers[i].OutSize()));
        v_a_.emplace_back(Matrix::Zero(layers[i].OutSize(), layers[i].InSize()));
        v_b_.emplace_back(Vector::Zero(layers[i].OutSize()));
    }
}

void Adam::Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                    int batch_size, const std::vector<Layer>& layers) {
    VectorT u;
    u = loss.Gradient(outputs.back(), ans);

    Matrix da;
    Vector db;
    for (size_t j = 0; j < layers.size(); ++j) {
        size_t k = layers.size() - j - 1;

        da = layers[k].GetDa(u, outputs[k]);
        db = layers[k].GetDb(u, outputs[k]);

        m_a_[k] += (beta1_ * m_a_[k] + (1 - beta1_) * da) / batch_size;
        v_a_[k] += (beta2_ * v_a_[k] + (1 - beta2_) * da.cwiseProduct(da)) / batch_size;

        m_b_[k] += (beta1_ * m_b_[k] + (1 - beta1_) * db) / batch_size;
        v_b_[k] += (beta2_ * v_b_[k] + (1 - beta2_) * db.cwiseProduct(db)) / batch_size;

        u = layers[k].Propagate(u, outputs[k]);
    }
}

void Adam::UpdateParams(std::vector<Layer>* layers) {
    Matrix m_hat_a;
    Matrix v_hat_a;
    Vector m_hat_b;
    Vector v_hat_b;
    for (size_t k = 0; k < layers->size(); ++k) {
        m_hat_a = m_a_[k] / (1 - std::pow(beta1_, time_));
        v_hat_a = (v_a_[k] / (1 - std::pow(beta2_, time_))).cwiseSqrt();
        v_hat_a.array() += eps_;
        (*layers)[k].UpdateA(m_hat_a.cwiseProduct(v_hat_a.cwiseInverse()), -alpha_);

        m_hat_b = m_b_[k] / (1 - std::pow(beta1_, time_));
        v_hat_b = (v_b_[k] / (1 - std::pow(beta2_, time_))).cwiseSqrt();
        v_hat_b.array() += eps_;
        (*layers)[k].UpdateB(m_hat_b.cwiseProduct(v_hat_b.cwiseInverse()), -alpha_);
    }
}

void Adam::ZeroGrads() {
    ++time_;
    for (size_t i = 0; i < m_a_.size(); ++i) {
        m_a_[i].setZero();
        m_b_[i].setZero();
        v_a_[i].setZero();
        v_b_[i].setZero();
    }
}

SGD::SGD(const std::vector<Layer>& layers) {
    das_.reserve(layers.size());
    dbs_.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        das_.emplace_back(Matrix::Zero(layers[i].OutSize(), layers[i].InSize()));
        dbs_.emplace_back(Vector::Zero(layers[i].OutSize()));
    }
}

void SGD::Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                   int batch_size, const std::vector<Layer>& layers) {
    VectorT u;
    u = loss.Gradient(outputs.back(), ans);
    for (size_t j = 0; j < layers.size(); ++j) {
        size_t k = layers.size() - j - 1;

        das_[k] += layers[k].GetDa(u, outputs[k]) / batch_size;
        dbs_[k] += layers[k].GetDb(u, outputs[k]) / batch_size;

        u = layers[k].Propagate(u, outputs[k]);
    }
}

void SGD::UpdateParams(std::vector<Layer>* layers) {
    for (size_t i = 0; i < layers->size(); ++i) {
        (*layers)[i].UpdateA(das_[i], -rate_);
        (*layers)[i].UpdateB(dbs_[i], -rate_);
    }
}

void SGD::ZeroGrads() {
    for (size_t i = 0; i < das_.size(); ++i) {
        das_[i].setZero();
        dbs_[i].setZero();
    }
}

}  // namespace nnet
