#include "mini_cuda_nn/nn/module.h"

#include "mini_cuda_nn/autograd/grad_mode.h"

#include <string>

namespace mcn::nn
{
    void Module::zero_grad()
    {
        for (Tensor *p : parameters())
            p->zero_grad();
    }

    void Module::to(Device device)
    {
        // to_ 原地替换 data_，Tensor 对象本身的地址不变，优化器持有的指针仍然有效
        for (Tensor *p : parameters())
            p->to_(device);
    }

    size_t Module::num_parameters()
    {
        size_t n = 0;
        for (const Tensor *p : parameters())
            n += p->numel();
        return n;
    }

    std::string Module::describe()
    {
        std::string s = name();
        for (const auto &[n, p] : named_parameters())
        {
            s += " ";
            s += n;
            s += format_shape(p->shape());
        }
        return s;
    }

    void copy_parameters(Module &dst, Module &src)
    {
        auto dst_params = dst.named_parameters();
        auto src_params = src.named_parameters();

        MCN_CHECK(dst_params.size() == src_params.size(),
                  "copy_parameters: {} has {} tensors but {} has {}",
                  dst.name(), dst_params.size(), src.name(), src_params.size());

        // 按名字对齐，不依赖容器顺序 —— Sequential 顺序被人调过也不会静默错位
        for (auto &[dst_name, dst_p] : dst_params)
        {
            const Tensor *found = nullptr;
            for (auto &[src_name, src_p] : src_params)
            {
                if (src_name == dst_name)
                {
                    found = src_p;
                    break;
                }
            }

            MCN_CHECK(found != nullptr, "copy_parameters: {} not found in {}",
                      dst_name, src.name());
            MCN_CHECK(found->shape() == dst_p->shape(),
                      "copy_parameters: {} shape {} != {}",
                      dst_name, format_shape(dst_p->shape()), format_shape(found->shape()));
            MCN_CHECK(found->device() == dst_p->device() && found->dtype() == dst_p->dtype(),
                      "copy_parameters: {} device/dtype mismatch", dst_name);

            // 数值复制：写进已有存储，不动 grad（权重同步不是可微操作）
            detail::NoGradGuard guard;
            dst_p->copy_(found->contiguous());
        }
    }
} // namespace mcn::nn
