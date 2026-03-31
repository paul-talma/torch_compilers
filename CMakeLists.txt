cmake_minimum_required(VERSION 3.16)
project(ml_systems CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -g)

# Phase 1: tensor + storage

add_library(tensor_lib
project1_tensor/src/storage.cpp
project1_tensor/src/tensor.cpp
)
target_include_directories(tensor_lib PUBLIC project1_tensor/include)

# Phase 2: ops

add_library(ops_lib
project2_ops/src/ops.cpp
)
target_include_directories(ops_lib PUBLIC project2_ops/include)
target_link_libraries(ops_lib tensor_lib)

# Phase 3: autodiff

add_library(autodiff_lib
project3_autodiff/src/graph.cpp
project3_autodiff/src/autograd.cpp
)
target_include_directories(autodiff_lib PUBLIC project3_autodiff/include)
target_link_libraries(autodiff_lib ops_lib tensor_lib)

# Phase 1 tests

add_executable(test_phase1 project1_tensor/tests/test_tensor.cpp)
target_link_libraries(test_phase1 tensor_lib)

# Phase 2+3 tests

add_executable(test_phase23 tests/test_ops_autodiff.cpp)
target_link_libraries(test_phase23 autodiff_lib ops_lib tensor_lib)
