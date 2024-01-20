#include "net.h"

namespace nnet {
Net::Net(std::initializer_list<unsigned> layers_sizes,
         std::initializer_list<ActivationFunction> activ_funcs, LossFunction loss)
    : loss_(loss) {

    assert((layers_sizes.size() == activ_funcs.size() + 2) && "n layers => n-1 activ funcs");

    layers_.reserve(layers_sizes.size() - 1);
    activ_funcs_.reserve(activ_funcs.size());

    auto iter = layers_sizes.begin();
    unsigned size = *iter;

    for (++iter; iter != layers_sizes.end(); ++iter) {
        layers_.emplace_back(*iter, size);
        size = *iter;
    }

    for (auto func : activ_funcs) {
        activ_funcs_.push_back(func);
    }
};

Net::Vector Net::Predict(const Vector &v) {
    Vector temp = v;
    for (size_t i = 0; i < layers_.size() - 1; ++i) {
        temp = layers_[i].Calculate(temp);
        temp = activ_funcs_[i].Apply(temp);
    }
    return layers_.back().Calculate(temp);
}

}  // namespace nnet