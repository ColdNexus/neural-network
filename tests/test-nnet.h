#pragma once

#include "net.h"

namespace nnet {
    void RunTests();
    void TestMnist();
    Net TrainMnist();
    std::vector<Net::Vector> ReadData(std::string file_name);
    std::vector<Net::Vector> NumToInd(std::vector<Net::Vector> ans);
}
