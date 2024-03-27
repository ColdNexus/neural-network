# Require CMake >= 3.18.
cmake_minimum_required(VERSION 3.18)

# Project name.
project(nnet)

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
)

set_compile_options(nnet)

# Namespace library for users.
add_library(nnet::nnet ALIAS nnet)

add_subdirectory(googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Set search paths.
target_include_directories(nnet PUBLIC src)
target_include_directories(nnet PUBLIC eigen)
target_include_directories(nnet PUBLIC EigenRand)
target_include_directories(nnet PUBLIC googletest)

add_executable(runTests 
  tests/test_googletest.cpp
)

target_link_libraries(runTests gtest gtest_main)
target_link_libraries(runTests nnet)
