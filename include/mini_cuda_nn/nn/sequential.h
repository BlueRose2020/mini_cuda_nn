#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mini_cuda_nn/nn/module.h"

namespace mcn::nn
{
    // 按顺序串联若干层：y = Ln(...L1(x))。
    // 层以 unique_ptr 持有，所以 Sequential 本身不可拷贝（Module 已经禁掉了）。
    class Sequential : public Module
    {
    public:
        // seq.add<Linear>(4, 8).add<ReLU>().add<Linear>(8, 1);
        template <typename M, typename... Args>
        Sequential &add(Args &&...args)
        {
            layers_.push_back(std::make_unique<M>(std::forward<Args>(args)...));
            return *this;
        }

        Tensor forward(const Tensor &x) override
        {
            Tensor y = x;
            for (auto &layer : layers_)
                y = layer->forward(y);
            return y;
        }

        std::vector<Tensor *> parameters() override
        {
            std::vector<Tensor *> out;
            for (auto &layer : layers_)
            {
                auto sub = layer->parameters();
                out.insert(out.end(), sub.begin(), sub.end());
            }
            return out;
        }

        // 名字带序号前缀（"0.weight"），copy_parameters 靠它对齐
        std::vector<std::pair<std::string, Tensor *>> named_parameters() override
        {
            std::vector<std::pair<std::string, Tensor *>> out;
            for (size_t i = 0; i < layers_.size(); ++i)
                for (auto &[n, p] : layers_[i]->named_parameters())
                    out.emplace_back(std::to_string(i) + "." + n, p);
            return out;
        }

        const char *name() const override { return "sequential"; }

        size_t size() const { return layers_.size(); }
        Module &operator[](size_t i) { return *layers_[i]; }

    private:
        std::vector<std::unique_ptr<Module>> layers_;
    };
} // namespace mcn::nn
