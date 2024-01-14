#include "../src/program.h"

#include <cassert>
#include <iostream>

int main() {
    assert(neural_network::AlwaysZero() == 1);
    
    std::cout << "TESTS PASSED\n" << neural_network::AlwaysZero() << '\n';
    return 0;
}