#include "internal.hpp"
#include <stdexcept>
#include <algorithm>
namespace mininn
{
    Tensor::Tensor(std::vector<size_t> s, Device d, bool r) : shape(std::move(s)), device(d), requires_grad(r)
    {
        size_t n = 1;
        for (size_t x : shape)
            n *= x;
        data.assign(n, 0);
        if (r)
            grad.assign(n, 0);
        if (d == Device::CUDA)
        {
#ifdef MINI_CUDA
            device_data = mininn_cuda_alloc(n);
            if (r)
                device_grad = mininn_cuda_alloc(n);
            std::vector<float> z(n, 0);
            mininn_cuda_upload(device_data, z.data(), n);
            if (r)
                mininn_cuda_upload(device_grad, z.data(), n);
#else
            throw std::runtime_error("CUDA disabled; configure with -DMINI_CUDA=ON");
#endif
        }
    }
    Tensor::~Tensor()
    {
#ifdef MINI_CUDA
        if (device_data)
            mininn_cuda_free(device_data);
        if (device_grad)
            mininn_cuda_free(device_grad);
#endif
    }
    size_t Tensor::numel() const { return data.size(); }
    void Tensor::fill(float value)
    {
        std::fill(data.begin(), data.end(), value);
        to_device();
    }
    void Tensor::to_host()
    {
#ifdef MINI_CUDA
        if (device == Device::CUDA && device_data) mininn_cuda_download(data.data(), device_data, numel());
#endif
    }
    void Tensor::to_device()
    {
#ifdef MINI_CUDA
        if (device == Device::CUDA && device_data) mininn_cuda_upload(device_data, data.data(), numel());
#endif
    }
    float Tensor::item() const
    {
        if (numel() != 1) throw std::runtime_error("item() requires a scalar tensor");
        const_cast<Tensor*>(this)->to_host();
        return data[0];
    }
    void Tensor::zero_grad()
    {
        std::fill(grad.begin(), grad.end(), 0.0f);
#ifdef MINI_CUDA
        if (device == Device::CUDA && device_grad)
        {
            std::vector<float> z(numel(), 0);
            mininn_cuda_upload(device_grad, z.data(), numel());
        }
#endif
    }
    TensorPtr tensor(std::vector<size_t> s, Device d, bool r) { return std::make_shared<Tensor>(std::move(s), d, r); }
    TensorPtr tensor(std::vector<size_t> s, const std::vector<float>& values, Device d, bool r)
    {
        auto out = tensor(std::move(s), d, r);
        if (values.size() != out->numel()) throw std::runtime_error("tensor value count mismatch");
        out->data = values;
        out->to_device();
        return out;
    }
    TensorPtr zeros(std::vector<size_t> s, Device d, bool r) { return tensor(std::move(s), d, r); }
    TensorPtr ones(std::vector<size_t> s, Device d, bool r) { auto out = tensor(std::move(s), d, r); out->fill(1.0f); return out; }
    TensorPtr scalar(float value, Device d, bool r) { return tensor({1}, {value}, d, r); }
}
