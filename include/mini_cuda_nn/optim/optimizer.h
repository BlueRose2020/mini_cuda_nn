#pragma once

#include <cstddef>
#include <vector>

#include "mini_cuda_nn/core/tensor.h"

namespace mcn::optim
{
    // 优化器基类。
    //
    // 基类只负责所有优化器都一样的部分：持有参数表、校验、遍历、跳过无梯度参数。
    // "怎么更新一个参数"由子类的 update_one() 决定（SGD / Adam / ... 各自实现）。
    //
    // step() 刻意不是虚函数（NVI）：子类改不了遍历顺序和校验。
    // 需要"每步只做一次"的准备工作（如 Adam 的 bias correction），重写 on_step_begin()。
    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;

        // 优化器持有参数的地址，语义上不可复制
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;

        // 遍历参数表，对每个有梯度的参数调用 update_one()
        void step();

        // 清空所有参数的梯度。
        // 每轮迭代前必须调用，或者用 Tensor::zero_grad() 手动清；否则梯度会一直累加。
        void zero_grad();

        // 子类名，只用于报错与日志
        virtual const char *name() const = 0;

        double lr() const { return lr_; }
        void set_lr(double lr) { lr_ = lr; }

        // 已经执行过多少步（Adam 的 bias correction 依赖它）
        size_t step_count() const { return step_count_; }

        // 存的是指针而不是张量对象。
        // Tensor 的拷贝是浅拷贝，存对象确实也能改到数据，但 vector 扩容时对象会搬家，
        // 外部持有的引用随即失效；再加上优化器只该看到调用方真正的那个叶子张量，
        // 所以这里统一用指针。
        const std::vector<Tensor *> &params() const { return params_; }

        // 追加一组参数（支持分批加入，例如先建编码器再建解码头）
        void add_param_group(const std::vector<Tensor *> &group);

    protected:
        Optimizer(std::vector<Tensor *> params, double lr);

        // 就地更新第 index 个参数：p = f(p, g)。
        // 契约（基类已保证，子类不用重复检查）：
        //   - g 非空、连续、形状/设备/dtype 与 p 完全一致
        //   - p 连续、浮点、叶子（不参与计算图）
        virtual void update_one(size_t index, Tensor &param, const Tensor &grad) = 0;

        // 每次 step() 开始时调用一次，在遍历参数之前
        virtual void on_step_begin() {}

        // 参数表发生变化后调用，子类在这里为 [first, first+count) 准备每参数状态
        // （动量缓冲、二阶矩……）。
        // 注意：基类构造期间不会调用它（此时子类尚未构造完，虚派发不可靠），
        // 构造时请由子类自己初始化好状态。
        virtual void on_params_added(size_t first, size_t count) {}

        double lr_;

    private:
        void check_param(const Tensor &param, size_t index) const;

        std::vector<Tensor *> params_;
        size_t step_count_ = 0;
    };
} // namespace mcn::optim
