#pragma once

#include <cstddef>

#include "mini_cuda_nn/core/dtype.h"

// 实现细节，不属于公开 API：只有 sgd.cpp / sgd.cu 需要。
// 宿主端可见的启动函数声明，实现在 src/optim/sgd.cu。

namespace mcn::detail
{
    // p -= lr * (g + weight_decay * p)
    void launch_sgd_update(void *param, const void *grad,
                           double lr, double weight_decay,
                           size_t n, DataType dtype);

    // buf = momentum * buf + g + weight_decay * p;  p -= lr * buf
    void launch_sgd_momentum_update(void *param, const void *grad, void *momentum,
                                    double lr, double momentum_coef, double weight_decay,
                                    size_t n, DataType dtype);
} // namespace mcn::detail
