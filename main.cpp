#include "except.h"
#include "test-nnet.h"

int main() {
    try {
        nnet::RunTests();
    } catch (...) {
        except::React();
    }
}
