#include "mini_cuda_nn/detail/sgd_kernel.h"

#include "mini_cuda_nn/core/cuda_check.h"
#include "mini_cuda_nn/detail/dtype_dispatch.h"

#include <algorithm>
#include <type_traits>

namespace
{
    constexpr unsigned kBlock = 256;
    constexpr unsigned kMaxBlocks = 4096;

    inline unsigned grid_for(size_t n)
    {
        const size_t blocks = (n + kBlock - 1) / kBlock;
        return static_cast<unsigned>(std::min<size_t>(blocks, kMaxBlocks));
    }

    // ---- 内核 ----
    template <typename T>
    __global__ void sgd_update_kernel(T *p, const T *g, T lr, T wd, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            p[i] -= lr * (g[i] + wd * p[i]);
    }

    template <typename T>
    __global__ void sgd_momentum_update_kernel(T *p, const T *g, T *buf,
                                               T lr, T mom, T wd, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        {
            const T b = mom * buf[i] + g[i] + wd * p[i];
            buf[i] = b;
            p[i] -= lr * b;
        }
    }

    // ---- 实现 ----
    template <typename T>
    void sgd_update_impl(void *param, const void *grad, double lr, double weight_decay, size_t n)
    {
        sgd_update_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(param),
                                                      static_cast<const T *>(grad),
                                                      static_cast<T>(lr),
                                                      static_cast<T>(weight_decay), n);
    }

    template <typename T>
    void sgd_momentum_update_impl(void *param, const void *grad, void *buf,
                                  double lr, double momentum, double weight_decay, size_t n)
    {
        sgd_momentum_update_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(param),
                                                               static_cast<const T *>(grad),
                                                               static_cast<T *>(buf),
                                                               static_cast<T>(lr),
                                                               static_cast<T>(momentum),
                                                               static_cast<T>(weight_decay), n);
    }
} // namespace

namespace mcn::detail
{
    void launch_sgd_update(void *param, const void *grad, double lr, double weight_decay,
                           size_t n, DataType dtype)
    {
        if (param == nullptr || grad == nullptr || n == 0)
            return;

        dispatch_dtype(dtype, [&]<typename T>()
                       {
                           if constexpr (!std::is_same_v<T, bool>)
                               sgd_update_impl<T>(param, grad, lr, weight_decay, n);
                       });
        CUDA_CHECK(cudaGetLastError());
    }

    void launch_sgd_momentum_update(void *param, const void *grad, void *momentum,
                                    double lr, double momentum_coef, double weight_decay,
                                    size_t n, DataType dtype)
    {
        if (param == nullptr || grad == nullptr || momentum == nullptr || n == 0)
            return;

        dispatch_dtype(dtype, [&]<typename T>()
                       {
                           if constexpr (!std::is_same_v<T, bool>)
                               sgd_momentum_update_impl<T>(param, grad, momentum, lr,
                                                           momentum_coef, weight_decay, n);
                       });
        CUDA_CHECK(cudaGetLastError());
    }
} // namespace mcn::detail
