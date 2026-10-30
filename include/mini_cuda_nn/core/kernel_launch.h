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

    enum class UnaryOp : int
    {
        Neg = 0
    };

    inline constexpr int kMaxNdim = 8;

    struct BcastArgs
    {
        int ndim = 0;
        size_t out_shape[kMaxNdim] = {};
        size_t a_stride[kMaxNdim] = {};
        size_t b_stride[kMaxNdim] = {};
    };

    struct SumToArgs
    {
        int ndim = 0;
        size_t out_shape[kMaxNdim] = {};
        size_t src_shape[kMaxNdim] = {};
        size_t src_stride[kMaxNdim] = {};
        size_t n_out = 0;
    };

    // 按任意 stride 把源拷成连续布局（用于 clone 非连续视图）
    struct StridedArgs
    {
        int ndim = 0;
        size_t shape[kMaxNdim] = {};
        size_t stride[kMaxNdim] = {};
    };

    void launch_fill(void *dst, double value, size_t n, DataType dtype);

    void launch_copy(void *dst, const void *src, size_t n, DataType dtype);

    // dst[i] = src[stride_offset(i)]，把非连续视图物化成连续布局
    void launch_copy_strided(void *dst, const void *src, const StridedArgs &args,
                             size_t n, DataType dtype);

    void launch_binary(BinaryOp op, void *dst, const void *a, const void *b,
                       size_t n, DataType dtype);

    // 广播版逐元素二元（两个输入可任意非连续/广播）
    void launch_binary_strided(BinaryOp op, void *dst, const void *a, const void *b,
                               const BcastArgs &args, size_t n, DataType dtype);

    // 把 src 的梯度累加归约到 out_shape（out 必须已清零）
    void launch_sum_to(void *dst, const void *src, const SumToArgs &args, DataType dtype);

    void launch_unary(UnaryOp op, void *dst, const void *src, size_t n, DataType dtype);

    // dst = max(src, 0)
    void launch_relu(void *dst, const void *src, size_t n, DataType dtype);

    // dst = input > 0 ? grad : 0
    void launch_relu_backward(void *dst, const void *grad, const void *input,
                              size_t n, DataType dtype);

    void launch_scale(void *dst, const void *src, double factor, size_t n, DataType dtype);

    void launch_matmul(void *out, const void *a, const void *b,
                       size_t M, size_t K, size_t N, DataType dtype);
}
