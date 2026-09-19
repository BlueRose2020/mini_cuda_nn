#include "mininn/nn.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
namespace mininn {
void Module::zero_grad() { for (auto& p : parameters()) p->zero_grad(); }
Linear::Linear(size_t in_features, size_t out_features, Device device)
    : weight_(tensor({in_features, out_features}, device, true)), bias_(zeros({out_features}, device, true)) {
    std::mt19937 gen(42u + static_cast<unsigned>(in_features * 31 + out_features));
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(in_features)));
    for (auto& x : weight_->data) x = dist(gen);
    weight_->to_device();
}
TensorPtr Linear::forward(TensorPtr input) { return add_bias(matmul(input, weight_), bias_); }
std::vector<TensorPtr> Linear::parameters() { return {weight_, bias_}; }
TensorPtr ReLU::forward(TensorPtr input) { return relu(input); }
Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) : modules_(modules) {}
TensorPtr Sequential::forward(TensorPtr input) { for (auto& module : modules_) input = module->forward(input); return input; }
std::vector<TensorPtr> Sequential::parameters() { std::vector<TensorPtr> out; for (auto& m : modules_) { auto ps=m->parameters(); out.insert(out.end(), ps.begin(), ps.end()); } return out; }
}
