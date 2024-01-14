# Require CMake >= 3.18.
cmake_minimum_required(VERSION 3.18)

# Project name.
project(nnet)

# Public options for users to control.
option(NNET_ENABLE_TESTS "Enable testing of the neural network" OFF)
option(NNET_ENABLE_BENCHMARKS "Enable benchmarking of the neural network" OFF)

# Import CMake modules.
include(cmake/Ccache.cmake)
include(cmake/CompileOptions.cmake)
include(cmake/Helpers.cmake)
include(cmake/ProjectSettings.cmake)

# Create library target.
add_library(nnet
  src/program.cpp
)
set_compile_options(nnet)

# Namespace library for users.
add_library(nnet::nnet ALIAS nnet)

# Set search paths.
target_include_directories(nnet PUBLIC src)

# Build tests if needed.
if (${NNET_ENABLE_TESTS})
  add_subdirectory(tests)
endif()

# Build benchmarks if needed.
if (${NNET_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

include_directories(eigen)
include_directories(EigenRand)