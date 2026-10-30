#pragma once

#include <string>
#include <utility>
#include <vector>

#include "mini_cuda_nn/nn/module.h"

namespace mcn::nn
{
    // 激活层：没有参数，只是把 Tensor 的算子包成 Module，
    // 这样它能进 Sequential、也能被统一遍历。
    class ReLU : public Module
    {
    public:
        Tensor forward(const Tensor &x) override { return x.relu(); }

        std::vector<Tensor *> parameters() override { return {}; }
        std::vector<std::pair<std::string, Tensor *>> named_parameters() override { return {}; }
        const char *name() const override { return "relu"; }
    };
} // namespace mcn::nn
