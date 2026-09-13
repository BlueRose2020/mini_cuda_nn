#include "internal.hpp"
#include <stdexcept>
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
    void Tensor::zero_grad()
    {
        std::fill(grad.begin(), grad.end(), 0);
#ifdef MINI_CUDA
        if (device == Device::CUDA && device_grad)
        {
            std::vector<float> z(numel(), 0);
            mininn_cuda_upload(device_grad, z.data(), numel());
        }
#endif
    }
    TensorPtr tensor(std::vector<size_t> s, Device d, bool r) { return std::make_shared<Tensor>(std::move(s), d, r); }
}
