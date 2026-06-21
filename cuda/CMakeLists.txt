cmake_minimum_required(VERSION 3.24)

# Standalone CUDA river-subgame solver engine.
# Intentionally Qt-free: it consumes a serialized subgame produced by the
# CPU (MinGW+Qt) exporter and produces a strategy file for comparison.
project(TexasSolverCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3070 = Ampere, compute capability 8.6
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(river_gpu
  src/main.cpp
  src/subgame.cpp
  src/validate.cpp
  src/compare_golden.cpp
  src/dump_strategy.cpp
  src/cuda_kernels.cu
  src/cfr_solver.cu
)

# repo include/ provides the header-only json.hpp (Qt-free); the CUDA toolkit
# include dir lets the host C++ TUs see <cuda_runtime.h> (cfr_solver.h uses
# cudaStream_t / cudaGraphExec_t for graph capture).
target_include_directories(river_gpu PRIVATE src ${CMAKE_SOURCE_DIR}/../include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(river_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
