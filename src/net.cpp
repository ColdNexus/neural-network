#include "net.h"
#include "random.h"
#include "layer.h"
#include "optimizer.h"

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

void Net::TrainVanilla(const std::vector<TrainData>& train_data, LossFunction loss, int epochs,
                       double tol) {
    assert(!layers_.empty() && "Net in NOT initialized");
    assert(epochs >= 1 && "epochs should be a positive integer");
    Scalar dist = 0;
    int    batch_size = train_data.size();

    SGD                 optimizer(layers_);
    std::vector<Vector> outputs(layers_.size() + 1);
    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < train_data.size(); ++i) {
            Forward(train_data[i].data, &outputs);
            dist += loss.Dist(outputs.back(), train_data[i].ans);
            optimizer.Backward(outputs, train_data[i].ans, loss, batch_size, layers_);
        }
        optimizer.UpdateParams(&layers_);
        optimizer.ZeroGrads();
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / train_data.size() << '\n';
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

    SGD                 optimizer(layers_);
    std::vector<Vector> outputs(layers_.size() + 1);
    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            for (size_t j = i; j < train_data.size() && j < i + batch_size; ++j) {
                Forward(train_data[j].data, &outputs);
                dist += loss.Dist(outputs.back(), train_data[j].ans);
                optimizer.Backward(outputs, train_data[j].ans, loss, batch_size, layers_);
            }
            optimizer.UpdateParams(&layers_);
            optimizer.ZeroGrads();
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

    Adam                optimizer(layers_);
    std::vector<Vector> outputs(layers_.size() + 1);
    Scalar              dist = 0;

    for (int cur_epoch = 1; cur_epoch <= epochs; ++cur_epoch) {
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            for (size_t j = i; j < train_data.size() && j < i + batch_size; ++j) {
                Forward(train_data[j].data, &outputs);
                dist += loss.Dist(outputs.back(), train_data[j].ans);
                optimizer.Backward(outputs, train_data[j].ans, loss, batch_size, layers_);
            }
            optimizer.UpdateParams(&layers_);
            optimizer.ZeroGrads();
        }
        if (debug) {
            std::cout << cur_epoch << " epoch ended\n";
            std::cout << "average dist " << dist / train_data.size() << '\n';
        }
        dist = 0;
    }
}

void Net::Forward(const Vector& inp, std::vector<Vector>* x) {
    (*x)[0] = inp;
    for (size_t j = 0; j < layers_.size(); ++j) {
        (*x)[j + 1] = layers_[j].Calculate((*x)[j]);
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
