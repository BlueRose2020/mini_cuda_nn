#pragma once

#include "mini_cuda_nn/core/tensor.h"

namespace mcn::nn
{
    // 标量归约方式，跟 PyTorch 的 reduction 同名
    enum class Reduction
    {
        None, // 逐元素返回，形状不变
        Sum,  // 全部加起来
        Mean  // 平均值（默认）
    };

    // 损失函数基类。
    // 跟 Module 的区别：既要 pred 也要 target，产出的是标量（或逐元素 loss），
    // 所以单独一个基类，接口只有 forward(pred, target)。
    class Loss
    {
    public:
        virtual ~Loss() = default;
        Loss(const Loss &) = delete;
        Loss &operator=(const Loss &) = delete;

        Tensor operator()(const Tensor &pred, const Tensor &target)
        {
            return forward(pred, target);
        }

        virtual Tensor forward(const Tensor &pred, const Tensor &target) = 0;
        virtual const char *name() const = 0;

        Reduction reduction() const { return reduction_; }

    protected:
        explicit Loss(Reduction reduction) : reduction_(reduction) {}
        Reduction reduction_;
    };

    // 均方误差：mean((pred - target)^2)
    //
    // 反向由已有算子组合出来（sub / mul / mean），没有专门的 kernel：
    //   d loss / d pred = 2 * (pred - target) / n
    class MSELoss : public Loss
    {
    public:
        explicit MSELoss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}

        const char *name() const override { return "mse_loss"; }
        Tensor forward(const Tensor &pred, const Tensor &target) override;
    };
} // namespace mcn::nn
