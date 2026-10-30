#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "mini_cuda_nn/core/tensor.h"

namespace mcn::nn
{
    // 网络层基类，思路跟 optim::Optimizer 一致：
    // 基类负责所有层都一样的事（持有参数、清梯度、换设备、按名字对齐复制），
    // 子类只写 forward() 和自己的参数列表。
    class Module
    {
    public:
        virtual ~Module() = default;
        Module() = default;

        // 参数以 Tensor 成员的形式存在子类里。Tensor 的拷贝是浅拷贝，
        // 一旦允许拷贝 Module，两个"层"就会共享同一块权重、训练时互相踩，
        // 而且不会报错 —— 直接禁掉。要独立副本请新建一个层再 copy_parameters()。
        Module(const Module &) = delete;
        Module &operator=(const Module &) = delete;
        Module(Module &&) = default;
        Module &operator=(Module &&) = default;

        // 前向：x -> y。子类实现 forward，外部统一用 operator() 调用
        Tensor operator()(const Tensor &x) { return forward(x); }
        virtual Tensor forward(const Tensor &x) = 0;

        // 参数的指针视图（不拥有内存），直接交给优化器
        virtual std::vector<Tensor *> parameters() = 0;

        // 带名字的参数，用于 copy_parameters() 与调试打印
        virtual std::vector<std::pair<std::string, Tensor *>> named_parameters() = 0;

        virtual const char *name() const = 0;

        // 清空所有参数的梯度（等价于对每个参数 zero_grad()）
        void zero_grad();

        // 就地迁移设备。改的是 Tensor 对象内部的存储指针，
        // 所以优化器已经持有的 Tensor* 依然有效。
        void to(Device device);

        size_t num_parameters();

        // 调试用：形如 "linear: weight[8, 3] bias[8]" 的一行摘要
        std::string describe();
    };

    // 按名字把 src 的参数数值复制进 dst（形状/设备/dtype 必须一致）。
    // DQN 的 target network 靠它同步：直接 `dst = src` 只会共享权重。
    void copy_parameters(Module &dst, Module &src);
} // namespace mcn::nn
