#pragma once
#include "mininn/tensor.hpp"
#include <vector>
namespace mininn {
class Module {
public:
    virtual ~Module() = default;
    virtual TensorPtr forward(TensorPtr input) = 0;
    virtual std::vector<TensorPtr> parameters() = 0;
    void zero_grad();
};
class Linear final : public Module {
public:
    Linear(std::size_t in_features, std::size_t out_features, Device device=Device::CPU);
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
private:
    TensorPtr weight_, bias_;
};
class ReLU final : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override { return {}; }
};
class Sequential final : public Module {
public:
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
private:
    std::vector<std::shared_ptr<Module>> modules_;
};
}
