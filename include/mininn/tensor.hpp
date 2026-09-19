#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <initializer_list>
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
    void fill(float value);
    void to_host();
    void to_device();
    float item() const;
};
using TensorPtr = std::shared_ptr<Tensor>;
TensorPtr tensor(std::vector<std::size_t> shape, Device device=Device::CPU, bool requires_grad=false);
TensorPtr tensor(std::vector<std::size_t> shape, const std::vector<float>& values, Device device=Device::CPU, bool requires_grad=false);
TensorPtr zeros(std::vector<std::size_t> shape, Device device=Device::CPU, bool requires_grad=false);
TensorPtr ones(std::vector<std::size_t> shape, Device device=Device::CPU, bool requires_grad=false);
TensorPtr scalar(float value, Device device=Device::CPU, bool requires_grad=false);
TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr add_bias(TensorPtr input, TensorPtr bias);
TensorPtr matmul(TensorPtr a, TensorPtr b);
TensorPtr relu(TensorPtr input);
TensorPtr mse_loss(TensorPtr prediction, TensorPtr target);
void backward(TensorPtr loss);
}
