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

    template <typename T>
    __global__ void neg_kernel(T *dst, const T *src, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = -src[i];
    }

    template <typename T>
    __global__ void scale_kernel(T *dst, const T *src, T factor, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
            dst[i] = src[i] * factor;
    }

    __device__ __forceinline__ void bcast_offsets(
        const mcn::detail::BcastArgs &args, size_t i, size_t &ao, size_t &bo)
    {
        size_t rem = i;
        ao = 0;
        bo = 0;
        for (int d = args.ndim - 1; d >= 0; --d)
        {
            const size_t dim = args.out_shape[d];
            const size_t idx = rem % dim;
            rem /= dim;
            ao += idx * args.a_stride[d];
            bo += idx * args.b_stride[d];
        }
    }

    template <typename T>
    __global__ void add_strided_kernel(T *dst, const T *a, const T *b,
                                       mcn::detail::BcastArgs args, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        {
            size_t ao, bo;
            bcast_offsets(args, i, ao, bo);
            dst[i] = a[ao] + b[bo];
        }
    }

    template <typename T>
    __global__ void sub_strided_kernel(T *dst, const T *a, const T *b,
                                       mcn::detail::BcastArgs args, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        {
            size_t ao, bo;
            bcast_offsets(args, i, ao, bo);
            dst[i] = a[ao] - b[bo];
        }
    }

    template <typename T>
    __global__ void mul_strided_kernel(T *dst, const T *a, const T *b,
                                       mcn::detail::BcastArgs args, size_t n)
    {
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        {
            size_t ao, bo;
            bcast_offsets(args, i, ao, bo);
            dst[i] = a[ao] * b[bo];
        }
    }

    // 梯度归约
    template <typename T>
    __global__ void sum_to_kernel(T *dst, const T *src, mcn::detail::SumToArgs args)
    {
        constexpr int kMax = mcn::detail::kMaxNdim;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        for (size_t o = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; o < args.n_out; o += stride)
        {
            // 输出元素的多维索引 -> src 的基础偏移
            size_t rem = o, base = 0;
            for (int d = args.ndim - 1; d >= 0; --d)
            {
                const size_t idx = rem % args.out_shape[d];
                rem /= args.out_shape[d];
                base += (args.src_shape[d] == 1 ? 0 : idx) * args.src_stride[d];
            }

            // 收集需要遍历的广播维
            size_t bn[kMax] = {};
            size_t bs[kMax] = {};
            int nb = 0;
            for (int d = 0; d < args.ndim; ++d)
            {
                if (args.out_shape[d] == 1 && args.src_shape[d] > 1)
                {
                    bn[nb] = args.src_shape[d];
                    bs[nb] = args.src_stride[d];
                    ++nb;
                }
            }

            T acc = T(0);
            if (nb == 0)
            {
                acc = src[base];
            }
            else
            {
                size_t counter[kMax] = {};
                while (true)
                {
                    size_t off = base;
                    for (int k = 0; k < nb; ++k)
                        off += counter[k] * bs[k];
                    acc += src[off];

                    // 里程表进位
                    int k = nb - 1;
                    while (k >= 0 && ++counter[k] >= bn[k])
                    {
                        counter[k] = 0;
                        --k;
                    }
                    if (k < 0)
                        break;
                }
            }
            dst[o] = acc;
        }
    }

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
    void neg_impl(void *dst, const void *src, size_t n)
    {
        neg_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                               static_cast<const T *>(src), n);
    }

    template <typename T>
    void scale_impl(void *dst, const void *src, double factor, size_t n)
    {
        scale_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                                 static_cast<const T *>(src),
                                                 static_cast<T>(factor), n);
    }

    template <typename T>
    void add_strided_impl(void *dst, const void *a, const void *b,
                          const mcn::detail::BcastArgs &args, size_t n)
    {
        add_strided_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                                       static_cast<const T *>(a),
                                                       static_cast<const T *>(b),
                                                       args, n);
    }

    template <typename T>
    void sub_strided_impl(void *dst, const void *a, const void *b,
                          const mcn::detail::BcastArgs &args, size_t n)
    {
        sub_strided_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                                       static_cast<const T *>(a),
                                                       static_cast<const T *>(b),
                                                       args, n);
    }

    template <typename T>
    void mul_strided_impl(void *dst, const void *a, const void *b,
                          const mcn::detail::BcastArgs &args, size_t n)
    {
        mul_strided_kernel<T><<<grid_for(n), kBlock>>>(static_cast<T *>(dst),
                                                       static_cast<const T *>(a),
                                                       static_cast<const T *>(b),
                                                       args, n);
    }

    template <typename T>
    void sum_to_impl(void *dst, const void *src, const mcn::detail::SumToArgs &args)
    {
        sum_to_kernel<T><<<grid_for(args.n_out), kBlock>>>(static_cast<T *>(dst),
                                                           static_cast<const T *>(src),
                                                           args);
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

    void launch_unary(UnaryOp op, void *dst, const void *src, size_t n, DataType dtype)
    {
        switch (op)
        {
        case UnaryOp::Neg:
            launch_checked(dtype, n, dst != nullptr && src != nullptr, [&]<typename T>()
                           { neg_impl<T>(dst, src, n); });
            break;
        }
    }

    void launch_scale(void *dst, const void *src, double factor, size_t n, DataType dtype)
    {
        launch_checked(dtype, n, dst != nullptr && src != nullptr, [&]<typename T>()
                       { scale_impl<T>(dst, src, factor, n); });
    }

    void launch_binary_strided(BinaryOp op, void *dst, const void *a, const void *b,
                               const BcastArgs &args, size_t n, DataType dtype)
    {
        if (dst == nullptr || a == nullptr || b == nullptr || n == 0)
            return;

        // op 的分支放在 host：device 侧每个 kernel 只有单一表达式（同 launch_binary）
        switch (op)
        {
        case BinaryOp::Add:
            dispatch_dtype(dtype, [&]<typename T>()
                           { add_strided_impl<T>(dst, a, b, args, n); });
            break;
        case BinaryOp::Sub:
            dispatch_dtype(dtype, [&]<typename T>()
                           { sub_strided_impl<T>(dst, a, b, args, n); });
            break;
        case BinaryOp::Mul:
            dispatch_dtype(dtype, [&]<typename T>()
                           { mul_strided_impl<T>(dst, a, b, args, n); });
            break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void launch_sum_to(void *dst, const void *src, const SumToArgs &args, DataType dtype)
    {
        if (dst == nullptr || src == nullptr || args.n_out == 0)
            return;

        dispatch_dtype(dtype, [&]<typename T>()
                       { sum_to_impl<T>(dst, src, args); });
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
