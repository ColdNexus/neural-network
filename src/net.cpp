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
    assert(!layers_.empty() && "Net in NOT initialized");
    Vector temp = v;
    for (size_t i = 0; i < layers_.size(); ++i) {
        temp = layers_[i].Calculate(temp);
    }
    return temp;
}

void Net::Train(const std::vector<Vector>& data, const std::vector<Vector>& ans, LossFunction loss,
                int epochs, double tol, bool debug) {

    assert(!layers_.empty() && "Net in NOT initialized");
    assert(epochs >= 1 && "epochs should be a positive integer");

    Adam                optimizer = InitAdam();
    std::vector<Vector> outputs(layers_.size() + 1);
    long long           time = 0;
    Scalar              dist = 0;

    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            ++time;
            Forward(data[i], &outputs);
            dist += loss.Dist(outputs.back(), ans[i]);
            Backward(outputs, ans[i], loss, &optimizer);
            Update(optimizer, time);

            // if (debug && i % (data.size() / 10) == 0) {
            //     std::cout << 10 * i / (data.size() / 10) << " happened of epoch\n";
            // }
        }
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / data.size() << '\n';
        }
        if (dist / data.size() < tol) {
            if (debug) {
                std::cout << "converged\n";
            }
            return;
        }
        dist = 0;
    }
}

Net::Adam Net::InitAdam() {

    Adam ret;
    ret.m_a.reserve(layers_.size());
    ret.v_a.reserve(layers_.size());
    ret.m_b.reserve(layers_.size());
    ret.v_b.reserve(layers_.size());

    for (size_t i = 0; i < layers_.size(); ++i) {
        ret.m_a.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        ret.m_b.emplace_back(Vector::Zero(layers_[i].OutSize()));
        ret.v_a.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        ret.v_b.emplace_back(Vector::Zero(layers_[i].OutSize()));
    }
    return ret;
}

void Net::Forward(const Vector& inp, std::vector<Vector>* x) {
    (*x)[0] = inp;
    for (size_t j = 0; j < layers_.size(); ++j) {
        (*x)[j + 1] = layers_[j].Calculate((*x)[j]);
    }
}

void Net::Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                   Adam* opt) {

    VectorT u;
    u = loss.Gradient(outputs.back(), ans);

    Matrix da;
    Vector db;
    for (size_t j = 0; j < layers_.size(); ++j) {
        size_t k = layers_.size() - j - 1;

        da = layers_[k].GetDa(u, outputs[k]);
        db = layers_[k].GetDb(u, outputs[k]);

        (*opt).m_a[k] = (*opt).beta1 * (*opt).m_a[k] + (1 - (*opt).beta1) * da;
        (*opt).v_a[k] = (*opt).beta2 * (*opt).v_a[k] + (1 - (*opt).beta2) * da.cwiseProduct(da);

        u = layers_[k].Propagate(u, outputs[k]);
    }
}

void Net::Update(const Adam& opt, long long t) {
    Matrix m_hat_a;
    Matrix v_hat_a;
    Vector m_hat_b;
    Vector v_hat_b;
    for (size_t k = 0; k < layers_.size(); ++k) {
        m_hat_a = opt.m_a[k] / (1 - std::pow(opt.beta1, t));
        v_hat_a = (opt.v_a[k] / (1 - std::pow(opt.beta2, t))).cwiseSqrt();
        v_hat_a.array() += opt.eps;
        layers_[k].UpdateA(m_hat_a.cwiseProduct(v_hat_a.cwiseInverse()), -opt.alpha);

        m_hat_b = opt.m_b[k] / (1 - std::pow(opt.beta1, t));
        v_hat_b = (opt.v_b[k] / (1 - std::pow(opt.beta2, t))).cwiseSqrt();
        v_hat_b.array() += opt.eps;
        layers_[k].UpdateB(m_hat_b.cwiseProduct(v_hat_b.cwiseInverse()), -opt.alpha);
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
