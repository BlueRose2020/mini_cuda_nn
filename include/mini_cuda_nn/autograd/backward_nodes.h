#pragma once

#include <vector>

#include "mini_cuda_nn/autograd/grad_fn.h"
#include "mini_cuda_nn/core/kernel_launch.h"

namespace mcn::detail
{
    // y = a <op> b（逐元素），out 已算好，这里只挂图
    void attach_binary(const Tensor &lhs, const Tensor &rhs, BinaryOp op, Tensor &out);

    // y = a @ b，out 已算好，这里只挂图
    void attach_matmul(const Tensor &lhs, const Tensor &rhs, Tensor &out);

    // y = -x，out 已算好，这里只挂图
    void attach_neg(const Tensor &in, Tensor &out);

    // y = x * factor，out 已算好，这里只挂图
    void attach_scalar_mul(const Tensor &in, double factor, Tensor &out);

    // y = sum(x)（全归约或沿某一维），out 已算好，这里只挂图。
    // keepdim_shape：被归约的维置 1 后的形状，反向用它把 g 放回正确维度。
    // mean 由 sum + mul_scalar 组合而成，不需要单独的节点。
    void attach_sum(const Tensor &in, std::vector<int> keepdim_shape, Tensor &out);
} // namespace mcn::detail
