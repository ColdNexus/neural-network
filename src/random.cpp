#include "random.h"

namespace nnet {
Random::Matrix Random::RandomMatrix(Index rows, Index cols) {
    return distribution.generate<Matrix>(rows, cols, generator);
}

Random::Vector Random::RandomVector(Index size) {
    return distribution.generate<Vector>(size, 1, generator);
}
}  // namespace nnet
