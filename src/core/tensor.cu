#include "mini_cuda_nn/core/cuda_check.h"
#include "mini_cuda_nn/core/dtype_dispatch.h"
#include "mini_cuda_nn/core/kernel_launch.h"

#include <algorithm>
#include <cstdint>
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
    __global__ void fill_kernel(T *dst, T value, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = value;
    }

    template <typename T>
    __global__ void copy_kernel(T *dst, const T *src, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = src[i];
    }

    template <typename T>
    __global__ void add_kernel(T *dst, const T *a, const T *b, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = a[i] + b[i];
    }

    template <typename T>
    __global__ void sub_kernel(T *dst, const T *a, const T *b, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = a[i] - b[i];
    }

    template <typename T>
    __global__ void mul_kernel(T *dst, const T *a, const T *b, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = a[i] * b[i];
    }

    // ---- 矩阵乘法：共享内存分块 ----
    // 每个 block 算 out 的一块 kTile x kTile。沿 K 方向逐块载入 a/b 的 tile 到
    // shared memory，块内做 kTile 次乘加，因此每个元素只从 global memory 读
    // (K/kTile) 次而不是 K 次。
    constexpr unsigned kTile = 16;

    template <typename T>
    __global__ void matmul_kernel(T *out, const T *a, const T *b,
                                  size_t M, size_t K, size_t N)
    {
        __shared__ T tile_a[kTile][kTile];
        __shared__ T tile_b[kTile][kTile];

        const size_t row = blockIdx.y * kTile + threadIdx.y;
        const size_t col = blockIdx.x * kTile + threadIdx.x;
        const size_t n_tiles = (K + kTile - 1) / kTile;

        T acc = T(0);
        for (size_t t = 0; t < n_tiles; ++t)
        {
            const size_t k0 = t * kTile;

            const size_t a_k = k0 + threadIdx.x;
            tile_a[threadIdx.y][threadIdx.x] =
                (row < M && a_k < K) ? a[row * K + a_k] : T(0);

            const size_t b_k = k0 + threadIdx.y;
            tile_b[threadIdx.y][threadIdx.x] =
                (b_k < K && col < N) ? b[b_k * N + col] : T(0);

            __syncthreads();

            for (unsigned k = 0; k < kTile; ++k)
                acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];

            __syncthreads();
        }

        if (row < M && col < N)
            out[row * N + col] = acc;
    }

    // ---- 实现 ----
    template <typename T>
    void fill_impl(void *dst, double value, size_t n)
    {
        fill_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst), static_cast<T>(value), n);
    }

    template <typename T>
    void copy_impl(void *dst, const void *src, size_t n)
    {
        copy_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                                static_cast<const T *>(src), n);
    }

    template <typename T>
    void add_impl(void *dst, const void *a, const void *b, size_t n)
    {
        add_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                               static_cast<const T *>(a),
                                               static_cast<const T *>(b), n);
    }

    template <typename T>
    void sub_impl(void *dst, const void *a, const void *b, size_t n)
    {
        sub_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                               static_cast<const T *>(a),
                                               static_cast<const T *>(b), n);
    }

    template <typename T>
    void mul_impl(void *dst, const void *a, const void *b, size_t n)
    {
        mul_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                               static_cast<const T *>(a),
                                               static_cast<const T *>(b), n);
    }

    template <typename T>
    void matmul_impl(void *out, const void *a, const void *b,
                     size_t M, size_t K, size_t N)
    {
        const dim3 block(kTile, kTile);
        const dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
                        static_cast<unsigned>((M + kTile - 1) / kTile));
        matmul_kernel<T><<<grid, block>>>(static_cast<T *>(out),
                                          static_cast<const T *>(a),
                                          static_cast<const T *>(b), M, K, N);
    }
}

namespace mcn::detail
{
    template <typename Fn>
    void launch_checked(DataType dtype, size_t n, bool params_ok, Fn &&fn)
    {
        if (!params_ok || n == 0)
            return;

        dispatch_dtype(dtype, [&]<typename T>()
                       { fn.template operator()<T>(); });
        CUDA_CHECK(cudaGetLastError());
    }

    void launch_fill(void *dst, double value, size_t n, DataType dtype)
    {
        launch_checked(dtype, n, dst != nullptr, [&]<typename T>()
                       { fill_impl<T>(dst, value, n); });
    }

    void launch_copy(void *dst, const void *src, size_t n, DataType dtype)
    {
        launch_checked(dtype, n, dst != nullptr && src != nullptr, [&]<typename T>()
                       { copy_impl<T>(dst, src, n); });
    }

    void launch_binary(BinaryOp op, void *dst, const void *a, const void *b,
                       size_t n, DataType dtype)
    {
        if (dst == nullptr || a == nullptr || b == nullptr || n == 0)
            return;

        switch (op)
        {
        case BinaryOp::Add:
            dispatch_dtype(dtype, [&]<typename T>()
                           { add_impl<T>(dst, a, b, n); });
            break;
        case BinaryOp::Sub:
            dispatch_dtype(dtype, [&]<typename T>()
                           { sub_impl<T>(dst, a, b, n); });
            break;
        case BinaryOp::Mul:
            dispatch_dtype(dtype, [&]<typename T>()
                           { mul_impl<T>(dst, a, b, n); });
            break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void launch_matmul(void *out, const void *a, const void *b,
                       size_t M, size_t K, size_t N, DataType dtype)
    {
        if (out == nullptr || a == nullptr || b == nullptr ||
            M == 0 || K == 0 || N == 0)
            return;

        // matmul 只支持浮点；bool 分支不实例化
        dispatch_dtype(dtype, [&]<typename T>()
                       {
                           if constexpr (!std::is_same_v<T, bool>)
                               matmul_impl<T>(out, a, b, M, K, N); });
        CUDA_CHECK(cudaGetLastError());
    }
}
