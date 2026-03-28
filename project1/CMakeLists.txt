cmake_minimum_required(VERSION 3.16)
project(tensor_phase1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(include)

# Student implementation files (they create these)
add_library(tensor_lib
    src/storage.cpp
    src/tensor.cpp
)

# Test binary
add_executable(tests tests/test_tensor.cpp)
target_link_libraries(tests tensor_lib)
