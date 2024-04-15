#include "except.h"

#include <exception>
#include <iostream>

namespace except {
void React() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    } catch (...) {
        std::cerr << "something went wrong\n";
    }
}
}  // namespace except
