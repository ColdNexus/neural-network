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
        layers_.emplace_back(*layer_iter, size, *func_iter);
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
    Scalar              step = kDefaultStep;
    std::vector<Matrix> a_modifications;
    std::vector<Vector> b_modifications;
    a_modifications.reserve(layers_.size());
    b_modifications.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        a_modifications.emplace_back(Matrix::Zero(layers_[i].OutSize(), layers_[i].InSize()));
        b_modifications.emplace_back(Vector::Zero(layers_[i].OutSize()));
    }

    std::vector<Vector> x(layers_.size());
    Vector              tmp;
    Vector              res;
    VectorT             u;

    for (int tt = 0; tt < epochs; ++tt) {
        for (size_t i = 0; i < data.size(); ++i) {

            tmp = data[i];
            for (size_t j = 0; j < layers_.size(); ++j) {
                x[j] = tmp;
                tmp = layers_[j].Calculate(tmp);
            }
            u = loss.Gradient(tmp, ans[i]);

            for (size_t j = 0; j < layers_.size(); ++j) {
                size_t k = layers_.size() - j - 1;
                a_modifications[k] += layers_[k].GetDa(u, x[k]);
                b_modifications[k] += layers_[k].GetDb(u, x[k]);
                u = layers_[k].Propagate(u, x[k]);
            }
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].UpdateA(a_modifications[i], -step);
            layers_[i].UpdateB(b_modifications[i], -step);
            a_modifications[i].setZero();
            b_modifications[i].setZero();
        }
        step /= 2;
        std::cout << "epoch " << tt << '\n';
    }
}

}  // namespace nnet
