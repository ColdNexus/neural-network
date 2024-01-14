# Require CMake >= 3.18.
cmake_minimum_required(VERSION 3.18)

# Project name.
project(neural-network)

# Public options for users to control.
option(NN_ENABLE_TESTS "Enable testing of the neural network" OFF)
option(NN_ENABLE_BENCHMARKS "Enable benchmarking of the neural network" OFF)

# Import CMake modules.
include(cmake/Ccache.cmake)
include(cmake/CompileOptions.cmake)
include(cmake/Helpers.cmake)
include(cmake/ProjectSettings.cmake)

# Create library target.
add_library(neural-network
  src/program.cpp
)
set_compile_options(neural-network)

# Build tests if needed.
if (${NN_ENABLE_TESTS})
  add_subdirectory(tests)
endif()

# Build benchmarks if needed.
if (${NN_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()
