#pragma once
#include "mininn.hpp"
namespace mininn { extern std::vector<std::function<void()>> tape;
#ifdef MINI_CUDA
extern "C" void* mininn_cuda_alloc(size_t); extern "C" void mininn_cuda_free(void*); extern "C" void mininn_cuda_upload(void*,const float*,size_t); extern "C" void mininn_cuda_download(float*,const void*,size_t); extern "C" void mininn_cuda_matmul(const float*,const float*,float*,int,int,int); extern "C" void mininn_cuda_sgd(float*,const float*,float,int); extern "C" bool mininn_cuda_available();
#endif
}
