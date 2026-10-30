#include "mini_cuda_nn/nn/loss.h"

namespace mcn::nn
{
    Tensor MSELoss::forward(const Tensor &pred, const Tensor &target)
    {
        // 不比形状：mse 的梯度必须一一对应，广播会把"谁减谁"变得含糊
        MCN_CHECK(pred.shape() == target.shape(),
                  "mse_loss: pred {} and target {} must have the same shape",
                  format_shape(pred.shape()), format_shape(target.shape()));
        MCN_CHECK(pred.device() == target.device(),
                  "mse_loss: pred and target are on different devices");
        MCN_CHECK(pred.dtype() == target.dtype(),
                  "mse_loss: pred dtype {} != target dtype {}",
                  dtype_to_string(pred.dtype()), dtype_to_string(target.dtype()));
        MCN_CHECK(is_floating_point(pred.dtype()),
                  "mse_loss: requires a floating point dtype, got {}",
                  dtype_to_string(pred.dtype()));

        // target 只当数据：detach 掉，避免把 target 那边的 require_grad 牵连进来
        const Tensor diff = pred - target.detach();
        const Tensor sq = diff * diff;

        switch (reduction_)
        {
        case Reduction::Sum:
            return sq.sum();
        case Reduction::Mean:
            return sq.mean();
        case Reduction::None:
            break;
        }
        return sq;
    }
} // namespace mcn::nn
