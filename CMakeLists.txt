# Require CMake >= 3.18.
cmake_minimum_required(VERSION 3.18)

# Project name.
project(nnet)

# Public options for users to control.
option(NN_ENABLE_TESTS "Enable testing of the neural network" OFF)

# Import CMake modules.
include(cmake/Ccache.cmake)
include(cmake/CompileOptions.cmake)
include(cmake/Helpers.cmake)
include(cmake/ProjectSettings.cmake)

# Create library target.
add_library(nnet
  src/net.cpp
  src/layer.cpp
  src/loss-func.cpp
  src/activation-func.cpp
  src/random.cpp
  src/optimizer.cpp
)

set_compile_options(nnet)

# Namespace library for users.
add_library(nnet::nnet ALIAS nnet)

add_subdirectory(googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Set search paths.
target_include_directories(nnet PUBLIC src)
target_include_directories(nnet PUBLIC tests)
target_include_directories(nnet PUBLIC eigen)
target_include_directories(nnet PUBLIC EigenRand)
target_include_directories(nnet PUBLIC googletest)

set(CMAKE_CXX_FLAGS "-O2")

add_executable(run
  main.cpp
  tests/except.cpp
  tests/test-nnet.cpp
)
target_link_libraries(run nnet)

# Build tests if needed.
if (${NNET_ENABLE_TESTS})
  add_subdirectory(tests)
endif()
