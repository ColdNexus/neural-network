#include <gtest/gtest.h>
#include <math.h>
#include <fstream>
#include <sstream>

// lord forgive me for what i am going to do
#define private public  // for testing purposes

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
    int     second_norm = 7;
    VectorT gradient(5);
    gradient << 2, 4, -2, 0, 2;

    EXPECT_EQ(nnet::MSE().Dist(v, w), second_norm);

    EXPECT_EQ(nnet::MSE().Gradient(v, w).rows(), gradient.rows());
    EXPECT_EQ(nnet::MSE().Gradient(v, w).cols(), gradient.cols());
    EXPECT_EQ(nnet::MSE().Gradient(v, w), gradient);
}

TEST(IO, ActivationFunction) {
    std::stringstream stream;
    stream << nnet::ReLu() << ' ' << nnet::Id();
    nnet::ActivationFunction sigma1, sigma2;
    stream >> sigma1 >> sigma2;

    EXPECT_EQ(nnet::ReLu(), sigma1);
    EXPECT_EQ(nnet::Id(), sigma2);
}

void ExpectLayerEq(const nnet::Layer& l, const nnet::Layer& r) {
    EXPECT_EQ(l.InSize(), r.InSize());
    EXPECT_EQ(l.OutSize(), r.OutSize());
    EXPECT_EQ(l.a_, r.a_);
    EXPECT_EQ(l.b_, r.b_);
    EXPECT_EQ(l.sigma_, r.sigma_);
}

TEST(IO, Layer) {
    std::stringstream stream;
    nnet::Layer       layer1(3, 5, nnet::ReLu());
    nnet::Layer       layer2(5, 2, nnet::Id());
    stream << layer1 << ' ' << layer2;

    nnet::Layer read_layer1;
    nnet::Layer read_layer2;
    stream >> read_layer1 >> read_layer2;

    ExpectLayerEq(layer1, read_layer1);
    ExpectLayerEq(layer2, read_layer2);
}

TEST(IO, Net) {
    std::stringstream stream;
    nnet::Net         net({10, 2, 4, 1}, {nnet::ReLu(), nnet::ReLu(), nnet::Id()});
    stream << net;

    nnet::Net read_net;
    stream >> read_net;

    EXPECT_EQ(net.layers_.size(), read_net.layers_.size());
    for (size_t i = 0; i < net.layers_.size(); ++i) {
        ExpectLayerEq(net.layers_[i], read_net.layers_[i]);
    }
}
