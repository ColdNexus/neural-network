#include <gtest/gtest.h>
#include "activation-func.h"
#include "net.h"
#include "usings.h"

TEST(ActivationFunction, ReLu) {
    Vector v{{1, 2, -1, -2, 5}};
    Vector ans0{{1, 2, 0, 0, 5}};
    EXPECT_TRUE(nnet::ReLu().Apply0(v) == ans0);

    Matrix ans1(5, 5);
    ans1 << 1, 0, 0, 0, 0,  // NOLINT
        0, 1, 0, 0, 0,      // NOLINT
        0, 0, 0, 0, 0,      // NOLINT
        0, 0, 0, 0, 0,      // NOLINT
        0, 0, 0, 0, 1;      // NOLINT
    EXPECT_TRUE(nnet::ReLu().Derivative(v) == ans1);
}
