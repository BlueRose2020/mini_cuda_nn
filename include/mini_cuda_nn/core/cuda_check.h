#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                                        \
    do                                                                                          \
    {                                                                                           \
        cudaError_t error = call;                                                               \
        if (error != cudaSuccess)                                                               \
        {                                                                                       \
            printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            abort();                                                                            \
        }                                                                                       \
    } while (0)
