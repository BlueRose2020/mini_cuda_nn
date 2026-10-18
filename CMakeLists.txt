cmake_minimum_required(VERSION 3.18)
project(mini_cuda_nn LANGUAGES CXX CUDA)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

find_package(CUDAToolkit REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(mcn_core
    src/core/tensor.cpp
    src/core/tensor.cu
)

target_include_directories(
    mcn_core PUBLIC include
)
target_link_libraries(
    mcn_core PUBLIC
    CUDA::cudart
)

add_subdirectory(./tests)