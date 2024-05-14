#pragma once

#include "net.h"

namespace nnet {
void RunTests();

void TestLinear();
Net  TrainLinear();
void TestMnist();
Net  TrainMnist();
void TestXor();
Net  TrainXor();

std::vector<Net::Vector> ReadData(std::string file_name);
std::vector<Net::Vector> NumToInd(std::vector<Net::Vector> ans);
}  // namespace nnet
