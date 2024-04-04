#include "activation-func.h"
#include "layer.h"
#include "loss-func.h"
#include "net.h"
#include "usings.h"

#include <cassert>
#include <iostream>

void Hello() {
    std::cout << "Hello!\n";
}

int main() {
    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    // std::cout << m/m << '\n';

    // auto x = m.cwiseProduct(1/m);
    std::cout << m.cwiseInverse().cwiseSqrt() << '\n';
    m.array() += 1;
    std::cout << m << '\n';

    Vector v(3);
    v << 1, 2, 3;
    std::cout << v << '\n';

    auto y = v.cwiseProduct(v);
    std::cout << v << '\n';
    std::cout << y << '\n';

    std::vector<nnet::Net::Vector> data;
    std::vector<nnet::Net::Vector> ans;
    for (int i = 0; i < 100; ++i) {
        nnet::Net::Vector datatmp(2);
        datatmp[0] = static_cast<int>(rand()) % 10;
        datatmp[1] = static_cast<int>(rand()) % 10;

        // std::cout << datatmp << "\n";
        data.push_back(datatmp);

        nnet::Net::Vector anstmp(1);
        anstmp[0] = datatmp[0] + datatmp[1];
        // std::cout << anstmp << "\n\n";
        ans.push_back(anstmp);
    }

    nnet::Net net({2, 1}, {nnet::Id()});
    net.Train(data, ans, nnet::MSE(), 400);

    nnet::Net::Vector test1{{2, 2}};
    nnet::Net::Vector test2{{3, 3}};
    nnet::Net::Vector test3{{4, 2}};
    nnet::Net::Vector test4{{1, 5}};
    nnet::Net::Vector test5{{9, 1}};
    nnet::Net::Vector test6{{3, 2}};
    nnet::Net::Vector test7{{0, 3}};

    std::cout << net.Predict(test1) << '\n';
    std::cout << net.Predict(test2) << '\n';
    std::cout << net.Predict(test3) << '\n';
    std::cout << net.Predict(test4) << '\n';
    std::cout << net.Predict(test5) << '\n';
    std::cout << net.Predict(test6) << '\n';
    std::cout << net.Predict(test7) << '\n';
    return 0;
}
