#include "layer.h"

#include <cassert>
#include <iostream>

namespace nnet {
    Layer::Layer(unsigned rows, unsigned cols){
        a_ = distribution.template generate<Matrix>(rows, cols, generator);
        b_ = distribution.template generate<Vector>(rows, 1, generator);
    }

    Layer::Vector Layer::Calculate(const Vector& x){
        assert(x.rows() == a_.cols() && "wrong vector size");
        return a_*x + b_;
    }
}  // namespace nnet