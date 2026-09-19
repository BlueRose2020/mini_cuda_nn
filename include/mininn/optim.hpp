#pragma once
#include "mininn/tensor.hpp"
#include <unordered_map>
#include <vector>
namespace mininn { void sgd_step(const std::vector<TensorPtr>& parameters, float learning_rate); void hybrid_sgd_step(const std::vector<TensorPtr>& parameters, float learning_rate); }
namespace mininn {
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<TensorPtr>& parameters) = 0;
    virtual void zero_grad(const std::vector<TensorPtr>& parameters);
};
class SGD final : public Optimizer {
public:
    explicit SGD(float learning_rate, float momentum=0.0f);
    void step(const std::vector<TensorPtr>& parameters) override;
private:
    float lr_, momentum_;
    std::unordered_map<Tensor*, std::vector<float>> velocity_;
};
class Adam final : public Optimizer {
public:
    explicit Adam(float learning_rate=1e-3f, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f);
    void step(const std::vector<TensorPtr>& parameters) override;
private:
    float lr_, beta1_, beta2_, eps_;
    std::size_t step_{0};
    struct State { std::vector<float> m, v; };
    std::unordered_map<Tensor*, State> state_;
};
}
