cmake_minimum_required(VERSION 3.12)
project(simson_integration_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -lgomp")

find_package(OpenMP)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(particles STATIC
        integration.cu
        integration.h
        )

target_compile_features(particles PUBLIC cxx_std_11)
set_target_properties( particles
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

set(CUDA_PROPAGATE_HOST_FLAGS ON)
# можно добавить флаги nvcc для компиляции
# (https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-specifying-compilation-phase)
#set(CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35;")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -Xcompiler -fopenmp" )
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-Xcompiler -fopenmp,-lgomp")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler="-fopenmp -fPIC")
# inline setup
#set(CUDASRCS
#        main.cu
#        )
#
#CUDA_ADD_EXECUTABLE(cuda_matrix ${CUDASRCS})
add_executable(simson_integration_cuda main.cpp)

set_property(TARGET simson_integration_cuda
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
message("CUDA flags:" ${CMAKE_CUDA_FLAGS})
target_link_libraries(simson_integration_cuda particles)