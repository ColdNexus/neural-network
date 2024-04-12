#include "net.h"
#include "layer.h"

#include <iostream>

namespace nnet {
Net::Net(std::initializer_list<Index>              layers_sizes,
         std::initializer_list<ActivationFunction> activ_funcs) {

    assert((layers_sizes.size() == activ_funcs.size() + 1) && "n layers => n activ funcs");
    assert((layers_sizes.begin() != layers_sizes.end()) && "list is empty");

    layers_.reserve(layers_sizes.size() - 1);

    auto layer_iter = layers_sizes.begin();
    auto func_iter = activ_funcs.begin();

    unsigned size = *layer_iter;

    for (++layer_iter; layer_iter != layers_sizes.end(); ++layer_iter, ++func_iter) {
        layers_.emplace_back(size, *layer_iter, *func_iter);
        size = *layer_iter;
    }
};

Net::Vector Net::Predict(const Vector& v) const {
    Vector temp = v;
    for (size_t i = 0; i < layers_.size(); ++i) {
        temp = layers_[i].Calculate(temp);
    }
    return temp;
}

void Net::Train(const std::vector<Vector>& data, const std::vector<Vector>& ans, LossFunction loss,
                int epochs) {

    Scalar alpha = 0.001;
    Scalar beta1 = 0.9;
    Scalar beta2 = 0.999;
    Scalar eps = 1e-8;

    std::vector<Matrix> a_modifications;
    std::vector<Vector> b_modifications;

    std::vector<Matrix> m_a;
    std::vector<Vector> m_b;
    std::vector<Matrix> m_at;
    std::vector<Vector> m_bt;

    std::vector<Matrix> v_a;
    std::vector<Vector> v_b;
    std::vector<Matrix> v_at;
    std::vector<Vector> v_bt;

    a_modifications.reserve(layers_.size());
    m_a.reserve(layers_.size());
    v_a.reserve(layers_.size());
    m_at.reserve(layers_.size());
    v_at.reserve(layers_.size());

    b_modifications.reserve(layers_.size());
    m_b.reserve(layers_.size());
    v_b.reserve(layers_.size());
    m_bt.reserve(layers_.size());
    v_bt.reserve(layers_.size());

    for (size_t i = 0; i < layers_.size(); ++i) {
        a_modifications.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        b_modifications.emplace_back(Vector::Zero(layers_[i].OutSize()));

        m_a.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        m_b.emplace_back(Vector::Zero(layers_[i].OutSize()));
        v_a.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        v_b.emplace_back(Vector::Zero(layers_[i].OutSize()));
        m_at.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        m_bt.emplace_back(Vector::Zero(layers_[i].OutSize()));
        v_at.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        v_bt.emplace_back(Vector::Zero(layers_[i].OutSize()));
    }

    std::vector<Vector> x(layers_.size());
    Vector              tmp;
    Vector              res;
    VectorT             u;

    Scalar dist = 0;

    for (int t = 1; t <= epochs; ++t) {
        for (size_t i = 0; i < data.size(); ++i) {
            tmp = data[i];
            for (size_t j = 0; j < layers_.size(); ++j) {
                x[j] = tmp;
                tmp = layers_[j].Calculate(tmp);
            }
            dist += loss.Dist(tmp, ans[i]);
            
            if (i % 100 == 0) {
                std::cout << loss.Dist(tmp, ans[i]) << '\n';
            }

            u = loss.Gradient(tmp, ans[i]);

            for (size_t j = 0; j < layers_.size(); ++j) {
                size_t k = layers_.size() - j - 1;
                a_modifications[k] += layers_[k].GetDa(u, x[k]);
                b_modifications[k] += layers_[k].GetDb(u, x[k]);
                u = layers_[k].Propagate(u, x[k]);
            }

            for (size_t k = 0; k < layers_.size(); ++k) {
                m_a[k] = beta1 * m_a[k] + (1 - beta1) * a_modifications[k];
                v_a[k] = beta2 * v_a[k] +
                         (1 - beta2) * a_modifications[k].cwiseProduct(a_modifications[k]);

                m_b[k] = beta1 * m_b[k] + (1 - beta1) * b_modifications[k];
                v_b[k] = beta2 * v_b[k] +
                         (1 - beta2) * b_modifications[k].cwiseProduct(b_modifications[k]);

                m_at[k] = m_a[k] / (1 - std::pow(beta1, t));
                v_at[k] = (v_a[k] / (1 - std::pow(beta2, t))).cwiseSqrt();
                v_at[k].array() += eps;

                m_bt[k] = m_b[k] / (1 - std::pow(beta1, t));
                v_bt[k] = (v_b[k] / (1 - std::pow(beta2, t))).cwiseSqrt();
                v_bt[k].array() += eps;
            }
            for (size_t i = 0; i < layers_.size(); ++i) {
                layers_[i].UpdateA(m_at[i].cwiseProduct(v_at[i].cwiseInverse()), -alpha);
                layers_[i].UpdateB(m_bt[i].cwiseProduct(v_bt[i].cwiseInverse()), -alpha);
                a_modifications[i].setZero();
                b_modifications[i].setZero();
            }
        }
        std::cout << "epoch " << t << '\n';
        std::cout << "dist " << dist / data.size() << '\n';
    }
}

std::ostream& operator<<(std::ostream& stream, const Net& net) {
    stream << net.layers_.size() << '\n';
    for (size_t i = 0; i < net.layers_.size(); ++i) {
        stream << net.layers_[i] << '\n';
    }
    return stream;
}
std::istream& operator>>(std::istream& stream, Net& net) {
    size_t size;
    stream >> size;
    net.layers_.resize(size);
    for (size_t i = 0; i < size; ++i) {
        stream >> net.layers_[i];
    }
    return stream;
}

}  // namespace nnet
