#pragma once
#include <cstddef>

#include "dtype.h"

// 仅宿主端可见的启动函数声明。
// 实现在 tensor.cu
namespace mcn::detail
{
    enum class BinaryOp : int
    {
        Add = 0,
        Sub = 1,
        Mul = 2
    };

    void launch_fill(void *dst, double value, size_t n, DataType dtype);

    void launch_copy(void *dst, const void *src, size_t n, DataType dtype);

    void launch_binary(BinaryOp op, void *dst, const void *a, const void *b,
                       size_t n, DataType dtype);

    // 行优先矩阵乘法：out[M,N] = a[M,K] @ b[K,N]
    // 三个指针都必须是连续内存，且 dtype 一致。
    void launch_matmul(void *out, const void *a, const void *b,
                       size_t M, size_t K, size_t N, DataType dtype);
}
