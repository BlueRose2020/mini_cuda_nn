cmake_minimum_required(VERSION 3.18)
project(mini_cuda_nn VERSION 0.1.0 LANGUAGES CXX CUDA)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_library(mcn_core
    src/core/tensor.cpp
    src/core/tensor.cu
    src/autograd/backward_nodes.cpp
)

target_include_directories(
    mcn_core PUBLIC include
)
target_link_libraries(
    mcn_core PUBLIC
    CUDA::cudart
)
# 让使用方自动获得 C++20（只影响 INTERFACE，不给自身的 CUDA 编译加重约束）
target_compile_features(mcn_core INTERFACE cxx_std_20)

enable_testing()
add_subdirectory(./tests)

# 开发期试验场：examples/ 下每个 .cpp/.cu 编成一个可执行文件到 build/bin/。
# 关掉用 -DMCN_BUILD_EXAMPLES=OFF。
option(MCN_BUILD_EXAMPLES "编译 examples/ 下的试验程序" ON)
if(MCN_BUILD_EXAMPLES)
    add_subdirectory(./examples)
endif()