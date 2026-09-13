#include "internal.hpp"
namespace mininn { bool cuda_available(){
#ifdef MINI_CUDA
return mininn_cuda_available();
#else
return false;
#endif
} }
