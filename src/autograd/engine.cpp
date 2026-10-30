// 自动微分引擎：拓扑排序 + 梯度累加。
//
// 单独成一个 TU，是为了让 autograd/ 目录名副其实 —— 反向节点
// （backward_nodes.cpp）和驱动它们的引擎住在一起，core/ 只管张量本身。
//
// 这里是 Tensor::backward() 的实现：它是 Tensor 的成员（需要访问 grad_fn_），
// 但逻辑完全属于自动微分，不涉及任何存储/视图操作。
#include "mini_cuda_nn/core/tensor.h"

#include "mini_cuda_nn/autograd/grad_mode.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

namespace mcn
{
    void Tensor::backward()
    {
        MCN_CHECK(grad_fn_ != nullptr,
                  "backward(): this tensor is not part of a graph "
                  "(missing require_grad on inputs, or produced by an op without a backward)");
        MCN_CHECK(is_floating_point(dtype_), "backward(): requires a floating point tensor, got {}",
                  dtype_to_string(dtype_));

        // 1. 拓扑排序：DFS 后序，再反转 => 输出节点在前，叶子在后
        std::vector<GradFn *> order;
        std::unordered_map<GradFn *, bool> visited;
        std::function<void(GradFn *)> visit = [&](GradFn *n)
        {
            if (visited[n])
                return;
            visited[n] = true;
            for (const auto &e : n->edges)
                if (e.fn)
                    visit(e.fn.get());
            order.push_back(n);
        };
        visit(grad_fn_.get());
        std::reverse(order.begin(), order.end());

        // 2. 播种：对输出整体求导，等价于 grad_output = 全 1
        std::unordered_map<GradFn *, Tensor> pending;
        pending.emplace(grad_fn_.get(), ones(shape_, device_, dtype_));

        // 3. 按拓扑序反传（反向公式里的算子不应再建图）
        detail::NoGradGuard guard;
        for (GradFn *node : order)
        {
            auto it = pending.find(node);
            if (it == pending.end())
                continue;

            const Tensor gout = it->second;
            std::vector<Tensor> grads = node->backward(gout);
            MCN_CHECK(grads.size() == node->edges.size(),
                      "{}: returned {} grads for {} edges",
                      node->name(), grads.size(), node->edges.size());

            for (size_t i = 0; i < grads.size(); ++i)
            {
                const Edge &e = node->edges[i];
                if (!e.sink && !e.fn)
                    continue;
                if (grads[i].data_ptr() == nullptr)
                    continue;

                // 前向可能做过广播：把梯度归约回该输入原本的形状
                Tensor gi = grads[i];
                if (gi.shape() != e.shape)
                    gi = gi.sum_to(e.shape);

                if (e.fn)
                {
                    auto jt = pending.find(e.fn.get());
                    if (jt == pending.end())
                        pending.emplace(e.fn.get(), gi.contiguous().clone());
                    else
                        jt->second.add_(gi.contiguous());
                }
                else
                {
                    e.sink->accumulate_grad(gi);
                }
            }
        }
    }
} // namespace mcn
