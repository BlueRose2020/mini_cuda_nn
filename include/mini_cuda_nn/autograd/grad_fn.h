#pragma once

#include <memory>
#include <vector>
#include "mini_cuda_nn/core/tensor.h"

namespace mcn
{
    class GradFn;

    class Edge
    {
    public:
        std::shared_ptr<GradFn> fn;
        const Tensor *sink = nullptr;
        // 前向输入的形状。反向时梯度会先归约到这个形状（处理广播）
        std::vector<size_t> shape;
        std::shared_ptr<Tensor> saved;
    };

    // 反向节点基类。每个前向算子对应一个派生类。
    class GradFn
    {
    public:
        virtual ~GradFn() = default;
        virtual const char *name() const = 0;
        virtual std::vector<Tensor> backward(const Tensor &grad_out) = 0;
        std::vector<Edge> edges;
    };
} // namespace mcn
