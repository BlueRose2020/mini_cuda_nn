#include "mini_cuda_nn/optim/optimizer.h"

#include "mini_cuda_nn/autograd/grad_mode.h"

#include <utility>

namespace mcn::optim
{
    Optimizer::Optimizer(std::vector<Tensor *> params, double lr)
        : lr_(lr)
    {
        MCN_CHECK(lr >= 0.0, "Optimizer: lr must be non-negative, got {}", lr);
        for (size_t i = 0; i < params.size(); ++i)
        {
            MCN_CHECK(params[i] != nullptr, "Optimizer: null param at index {}", i);
            check_param(*params[i], i);
        }
        params_ = std::move(params);
    }

    void Optimizer::check_param(const Tensor &param, size_t index) const
    {
        // 没有 require_grad 的参数永远拿不到梯度，step() 会静默跳过它 —— 这是最难查的一类 bug
        MCN_CHECK(param.require_grad(),
                  "optimizer param #{} does not require grad, so it would never be updated; "
                  "create it with require_grad=true, or leave it out of the param list",
                  index);
        MCN_CHECK(is_floating_point(param.dtype()),
                  "optimizer param #{} must be floating point, got {}",
                  index, dtype_to_string(param.dtype()));
        // 更新 kernel 按线性下标访问，参数必须是连续布局
        MCN_CHECK(param.is_contiguous(),
                  "optimizer param #{} is not contiguous; materialize it with clone() first",
                  index);
    }

    void Optimizer::add_param_group(const std::vector<Tensor *> &group)
    {
        const size_t first = params_.size();
        for (size_t i = 0; i < group.size(); ++i)
        {
            MCN_CHECK(group[i] != nullptr, "add_param_group: null param at index {}", i);
            check_param(*group[i], first + i);
        }
        params_.insert(params_.end(), group.begin(), group.end());

        // 让子类把动量/二阶矩之类的每参数状态补齐，否则 update_one 会越界
        on_params_added(first, group.size());
    }

    void Optimizer::step()
    {
        // 更新参数不是可微操作。即使调用方忘了 NoGradGuard，也不该在这里建图。
        detail::NoGradGuard guard;

        on_step_begin();

        for (size_t i = 0; i < params_.size(); ++i)
        {
            Tensor &p = *params_[i];
            const Tensor *g = p.grad();

            // 本轮没有参与前向（或只对部分参数调用了 backward）：跳过
            if (g == nullptr || g->data_ptr() == nullptr)
                continue;

            // accumulate_grad 只接受同形状/同设备/同 dtype 的梯度，
            // 这里再确认一次，防止有人手动塞过梯度
            MCN_CHECK(g->shape() == p.shape(), "{}: grad shape {} does not match param {}",
                      name(), format_shape(g->shape()), format_shape(p.shape()));
            MCN_CHECK(g->device() == p.device(), "{}: grad device mismatch on param #{}", name(), i);
            MCN_CHECK(g->dtype() == p.dtype(), "{}: grad dtype mismatch on param #{}", name(), i);

            update_one(i, p, *g);
        }

        ++step_count_;
    }

    void Optimizer::zero_grad()
    {
        for (Tensor *p : params_)
            p->zero_grad();
    }
} // namespace mcn::optim
