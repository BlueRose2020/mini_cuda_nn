#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
namespace mininn {
enum class Device { CPU, CUDA };
struct Tensor {
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<std::size_t> shape;
    Device device{Device::CPU};
    bool requires_grad{false};
    void* device_data{nullptr};
    void* device_grad{nullptr};
    Tensor(std::vector<std::size_t> shape, Device device=Device::CPU, bool requires_grad=false);
    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    std::size_t numel() const;
    void zero_grad();
};
using TensorPtr = std::shared_ptr<Tensor>;
TensorPtr tensor(std::vector<std::size_t> shape, Device device=Device::CPU, bool requires_grad=false);
TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr matmul(TensorPtr a, TensorPtr b);
TensorPtr mse_loss(TensorPtr prediction, TensorPtr target);
void backward(TensorPtr loss);
}
