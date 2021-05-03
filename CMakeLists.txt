cmake_minimum_required(VERSION 3.13)

project(CUTLASS-ISSUE CUDA CXX C)

include(cmake/Dependencies.cmake)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

add_subdirectory(src)
