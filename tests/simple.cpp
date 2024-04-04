#include <gtest/gtest.h>
#include <math.h>
#include "activation-func.h"
#include "layer.h"
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

TEST(Layer, Basic) {
    int         input = 5;
    int         output = 3;
    nnet::Layer layer{input, output, nnet::ReLu()};

    EXPECT_EQ(input, layer.InSize());
    EXPECT_EQ(output, layer.OutSize());

    Vector v(5);
    v << 1, 2, 3, 4, 5;

    EXPECT_EQ(layer.Calculate(v), layer.Calculate(v));
}

TEST(LossFunction, MSE) {
    Vector v(5);
    v << 1, 2, -3, 4, -5;
    Vector w(5);
    w << 0, 0, -2, 4, -6;
    int second_norm = 7;
    VectorT gradient(5);
    gradient << 2, 4, -2, 0, 2;

    EXPECT_EQ(nnet::MSE().Dist(v, w), second_norm);
    
    EXPECT_EQ(nnet::MSE().Gradient(v, w).rows(), gradient.rows());
    EXPECT_EQ(nnet::MSE().Gradient(v, w).cols(), gradient.cols());
    EXPECT_EQ(nnet::MSE().Gradient(v, w), gradient);
}
