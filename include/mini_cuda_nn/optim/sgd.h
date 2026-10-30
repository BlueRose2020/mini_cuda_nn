#pragma once

#include <cstddef>
#include <vector>

#include "mini_cuda_nn/optim/optimizer.h"

namespace mcn::optim
{
    // 随机梯度下降：
    //
    //   g <- g + weight_decay * p          （weight_decay != 0 时）
    //   b <- momentum * b + g              （momentum != 0 时）
    //   p <- p - lr * b
    //
    // 两个系数都为 0 时退化成最朴素的 p -= lr * g，走单独的 kernel，不做多余运算。
    class SGD : public Optimizer
    {
    public:
        explicit SGD(std::vector<Tensor *> params, double lr,
                     double momentum = 0.0, double weight_decay = 0.0);

        const char *name() const override { return "sgd"; }

        double momentum() const { return momentum_; }
        double weight_decay() const { return weight_decay_; }

    protected:
        void update_one(size_t index, Tensor &param, const Tensor &grad) override;

        // 参数表变化时补齐动量缓冲
        void on_params_added(size_t first, size_t count) override;

    private:
        double momentum_;
        double weight_decay_;
        // 一阶动量缓冲，与 params() 一一对应；momentum_ == 0 时为空
        std::vector<Tensor> momentum_buf_;
    };
} // namespace mcn::optim
