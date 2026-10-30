#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "mini_cuda_nn/nn/module.h"

namespace mcn::nn
{
    // 固定权重初始化的随机种子（影响所有后续构造的 Linear）。
    // 默认用 random_device 播种，训练结果不可复现；RL 这类对种子敏感的场合先调它。
    void manual_seed(unsigned seed);

    // 全连接层：y = x @ W^T + b
    //   x: [B, in_features]
    //   W: [out_features, in_features]（与 PyTorch 一致，方便对照权重）
    //   b: [out_features]，加到每一批的每一行（走广播）
    //
    // 注意这里用的是 weight.t()，而转置视图是带 grad_fn 的，
    // 所以梯度能正确回到 W 本身（见 Tensor::transpose）。
    class Linear : public Module
    {
    public:
        Linear(size_t in_features, size_t out_features,
               bool require_grad = true,
               Device device = Device::CPU,
               DataType dtype = DataType::FLOAT32);

        Tensor forward(const Tensor &x) override;

        std::vector<Tensor *> parameters() override { return {&weight_, &bias_}; }

        std::vector<std::pair<std::string, Tensor *>> named_parameters() override
        {
            return {{"weight", &weight_}, {"bias", &bias_}};
        }

        const char *name() const override { return "linear"; }

        Tensor &weight() { return weight_; }
        Tensor &bias() { return bias_; }
        size_t in_features() const { return in_features_; }
        size_t out_features() const { return out_features_; }

    private:
        size_t in_features_;
        size_t out_features_;
        Tensor weight_; // [out, in]
        Tensor bias_;   // [out]
    };
} // namespace mcn::nn
