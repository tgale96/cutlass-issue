# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# compile CUDA with /usr/local/cuda/bin/nvcc
CUDA_DEFINES = 

CUDA_INCLUDES = -isystem=/mount/cutlass-issue/third_party/cutlass/include

CUDA_FLAGS =  -gencode arch=compute_80,code=sm_80 --ptxas-options=-v -O3 -DNDEBUG -std=c++14
