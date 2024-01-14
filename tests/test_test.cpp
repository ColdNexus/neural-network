#include "program.h"

#include <cassert>
#include <iostream>

int main() {
    assert(nnet::AlwaysZero() == 0);
    
    std::cout << "TESTS PASSED\n" << nnet::AlwaysZero() << '\n';
    return 0;
}