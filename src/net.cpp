#include "net.h"
#include "random.h"
#include "layer.h"
#include "random.h"

#include <algorithm>
#include <iostream>

namespace nnet {
bool debug = true;

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
                int epochs) {

    Scalar alpha = 0.001;
    Scalar beta1 = 0.9;
    Scalar beta2 = 0.999;
    Scalar eps = 1e-8;

    Scalar dist = 0;

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

    for (int t = 1; t <= epochs; ++t) {
        for (size_t i = 0; i < data.size(); ++i) {
            tmp = data[i];
            for (size_t j = 0; j < layers_.size(); ++j) {
                x[j] = tmp;
                tmp = layers_[j].Calculate(tmp);
            }
            u = loss.Gradient(tmp, ans[i]);
            dist += loss.Dist(tmp, ans[i]);

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
        std::cout << "epoch " << t << " dist " << dist / data.size() << '\n';
        dist = 0;
    }
}

void Net::TrainVanilla(const std::vector<Vector>& data, const std::vector<Vector>& ans,
                       LossFunction loss, int epochs, double tol) {
    assert(!layers_.empty() && "Net in NOT initialized");
    assert(epochs >= 1 && "epochs should be a positive integer");
    Scalar dist = 0;

    auto [das, dbs] = InitGrads();
    std::vector<Vector> outputs(layers_.size() + 1);
    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            Forward(data[i], &outputs);
            dist += loss.Dist(outputs.back(), ans[i]);
            UpdateGrads(outputs, ans[i], loss, data.size(), &das, &dbs);
        }
        UpdateParams(das, dbs);
        ZeroGrads(&das, &dbs);
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / data.size() << '\n';
        }
        dist = 0;
    }
}

void Net::TrainSGD(const std::vector<TrainData>& train_data, LossFunction loss, int epochs,
                   int batch_size, double tol) {
    assert(!layers_.empty() && "Net in NOT initialized");
    assert(epochs >= 1 && "epochs should be a positive integer");

    Scalar dist = 0;
    Scalar prev = 0;

    auto [das, dbs] = InitGrads();
    std::vector<Vector> outputs(layers_.size() + 1);
    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        // random.Shuffle(&train_data);
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            for (size_t j = i; j < train_data.size() && j < i + batch_size; ++j) {
                Forward(train_data[j].data, &outputs);
                dist += loss.Dist(outputs.back(), train_data[j].ans);
                UpdateGrads(outputs, train_data[j].ans, loss, batch_size, &das, &dbs);
            }
            UpdateParams(das, dbs);
            ZeroGrads(&das, &dbs);
        }
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / train_data.size() << '\n';
        }
        if (std::abs(dist - prev) < tol) {
            if (debug) {
                std::cout << "converged\n";
            }
            return;
        }
        prev = dist;
        dist = 0;
    }
}

void Net::TrainAdam(std::vector<TrainData>& train_data, LossFunction loss, int epochs,
                    int batch_size, double tol) {

    assert(!layers_.empty() && "Net in NOT initialized");
    assert(epochs >= 1 && "epochs should be a positive integer");

    Adam                optimizer = InitAdam();
    std::vector<Vector> outputs(layers_.size() + 1);
    Scalar              dist = 0;

    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            for (size_t j = i; j < train_data.size() && j < i + batch_size; ++j) {
                Forward(train_data[j].data, &outputs);
                dist += loss.Dist(outputs.back(), train_data[j].ans);
                Backward(outputs, train_data[j].ans, loss, batch_size, &optimizer);
            }
            Update(optimizer, cur_epoch);
            ZeroAdam(&optimizer);
        }
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / train_data.size() << '\n';
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
void Net::ZeroAdam(Adam* optimizer) {
    for (size_t i = 0; i < optimizer->m_a.size(); ++i) {
        optimizer->m_a[i].setZero();
        optimizer->m_b[i].setZero();
        optimizer->v_a[i].setZero();
        optimizer->v_b[i].setZero();
    }
}

void Net::Forward(const Vector& inp, std::vector<Vector>* x) {
    (*x)[0] = inp;
    for (size_t j = 0; j < layers_.size(); ++j) {
        (*x)[j + 1] = layers_[j].Calculate((*x)[j]);
    }
}

void Net::Backward(const std::vector<Vector> outputs, const Vector& ans, const LossFunction& loss,
                   int batch_size, Adam* opt) {

    VectorT u;
    u = loss.Gradient(outputs.back(), ans);

    Matrix da;
    Vector db;
    for (size_t j = 0; j < layers_.size(); ++j) {
        size_t k = layers_.size() - j - 1;

        da = layers_[k].GetDa(u, outputs[k]);
        db = layers_[k].GetDb(u, outputs[k]);

        opt->m_a[k] += (opt->beta1 * opt->m_a[k] + (1 - opt->beta1) * da) / batch_size;
        opt->v_a[k] +=
            (opt->beta2 * opt->v_a[k] + (1 - opt->beta2) * da.cwiseProduct(da)) / batch_size;

        opt->m_b[k] += (opt->beta1 * opt->m_b[k] + (1 - opt->beta1) * db) / batch_size;
        opt->v_b[k] +=
            (opt->beta2 * opt->v_b[k] + (1 - opt->beta2) * db.cwiseProduct(db)) / batch_size;

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

std::pair<std::vector<Net::Matrix>, std::vector<Net::Vector>> Net::InitGrads() {
    std::vector<Matrix> das;
    das.reserve(layers_.size());
    std::vector<Vector> dbs;
    dbs.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        das.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        dbs.emplace_back(Vector::Zero(layers_[i].OutSize()));
    }
    return {das, dbs};
}
void Net::ZeroGrads(std::vector<Matrix>* das, std::vector<Vector>* dbs) {
    for (size_t i = 0; i < layers_.size(); ++i) {
        (*das)[i].setZero();
        (*dbs)[i].setZero();
    }
}

void Net::UpdateGrads(const std::vector<Vector> outputs, const Vector& ans,
                      const LossFunction& loss, int batch_size, std::vector<Matrix>* das,
                      std::vector<Vector>* dbs) {
    VectorT u;
    u = loss.Gradient(outputs.back(), ans);
    for (size_t j = 0; j < layers_.size(); ++j) {
        size_t k = layers_.size() - j - 1;

        (*das)[k] += layers_[k].GetDa(u, outputs[k]) / batch_size;
        (*dbs)[k] += layers_[k].GetDb(u, outputs[k]) / batch_size;

        u = layers_[k].Propagate(u, outputs[k]);
    }
}
void Net::UpdateParams(const std::vector<Matrix>& das, const std::vector<Vector>& dbs) {
    double mu = 0.005;
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].UpdateA(das[i], -mu);
        layers_[i].UpdateB(dbs[i], -mu);
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
