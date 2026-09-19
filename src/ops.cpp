#include "internal.hpp"
#include <stdexcept>
namespace mininn
{
    std::vector<std::function<void()>> tape;
    TensorPtr add(TensorPtr a, TensorPtr b)
    {
        if (a->shape != b->shape || a->device != b->device)
            throw std::runtime_error("add shape/device mismatch");
        auto o = tensor(a->shape, a->device, a->requires_grad || b->requires_grad);
        if (a->device == Device::CPU)
        {
            for (size_t i = 0; i < o->numel(); i++)
                o->data[i] = a->data[i] + b->data[i];
        }
#ifdef MINI_CUDA
        else
        {
            mininn_cuda_upload(a->device_data, a->data.data(), a->numel());
            mininn_cuda_upload(b->device_data, b->data.data(), b->numel());
            std::vector<float> x(a->numel()), y(b->numel());
            mininn_cuda_download(x.data(), a->device_data, a->numel());
            mininn_cuda_download(y.data(), b->device_data, b->numel());
            for (size_t i = 0; i < o->numel(); i++)
                o->data[i] = x[i] + y[i];
            mininn_cuda_upload(o->device_data, o->data.data(), o->numel());
        }
#endif
        if (o->requires_grad) tape.push_back([o,a,b](){ for(size_t i=0;i<o->numel();++i) { if(a->requires_grad) a->grad[i]+=o->grad[i]; if(b->requires_grad) b->grad[i]+=o->grad[i]; } });
        return o;
    }
    TensorPtr add_bias(TensorPtr input, TensorPtr bias)
    {
        if (input->shape.size() != 2 || bias->shape.size() != 1 || input->shape[1] != bias->shape[0] || input->device != bias->device)
            throw std::runtime_error("add_bias shape/device mismatch");
        auto o = tensor(input->shape, input->device, input->requires_grad || bias->requires_grad);
        input->to_host(); bias->to_host();
        for (size_t i=0; i<input->shape[0]; ++i) for (size_t j=0; j<input->shape[1]; ++j) o->data[i*input->shape[1]+j] = input->data[i*input->shape[1]+j] + bias->data[j];
        o->to_device();
        if (o->requires_grad) tape.push_back([o,input,bias](){ for(size_t i=0;i<input->shape[0];++i) for(size_t j=0;j<input->shape[1];++j) { auto g=o->grad[i*input->shape[1]+j]; if(input->requires_grad) input->grad[i*input->shape[1]+j]+=g; if(bias->requires_grad) bias->grad[j]+=g; } });
        return o;
    }
    TensorPtr relu(TensorPtr input)
    {
        input->to_host(); auto o = tensor(input->shape, input->device, input->requires_grad);
        for(size_t i=0;i<input->numel();++i) o->data[i] = input->data[i] > 0 ? input->data[i] : 0;
        o->to_device();
        if(o->requires_grad) tape.push_back([o,input](){ for(size_t i=0;i<input->numel();++i) if(input->requires_grad && input->data[i]>0) input->grad[i]+=o->grad[i]; });
        return o;
    }
    TensorPtr matmul(TensorPtr a, TensorPtr b)
    {
        if (a->shape.size() != 2 || b->shape.size() != 2 || a->shape[1] != b->shape[0] || a->device != b->device)
            throw std::runtime_error("matmul shape/device mismatch");
        size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
        auto o = tensor({m, n}, a->device, a->requires_grad || b->requires_grad);
#ifdef MINI_CUDA
        if (a->device == Device::CUDA)
        {
            mininn_cuda_upload(a->device_data, a->data.data(), a->numel());
            mininn_cuda_upload(b->device_data, b->data.data(), b->numel());
            mininn_cuda_matmul((const float *)a->device_data, (const float *)b->device_data, (float *)o->device_data, (int)m, (int)n, (int)k);
            mininn_cuda_download(o->data.data(), o->device_data, o->numel());
        }
        else
#endif
            for (size_t i = 0; i < m; i++)
                for (size_t j = 0; j < n; j++)
                    for (size_t q = 0; q < k; q++)
                        o->data[i * n + j] += a->data[i * k + q] * b->data[q * n + j];
        if (o->requires_grad)
            tape.push_back([o, a, b, m, k, n]()
                           {for(size_t i=0;i<m;i++)for(size_t j=0;j<n;j++)for(size_t q=0;q<k;q++){float g=o->grad[i*n+j];if(a->requires_grad)a->grad[i*k+q]+=g*b->data[q*n+j];if(b->requires_grad)b->grad[q*n+j]+=g*a->data[i*k+q];} });
        return o;
    }
    TensorPtr mse_loss(TensorPtr p, TensorPtr t)
    {
        if (p->shape != t->shape || p->device != t->device) throw std::runtime_error("mse_loss shape/device mismatch");
        p->to_host(); t->to_host();
        if (p->device == Device::CUDA)
        {
#ifdef MINI_CUDA
            mininn_cuda_download(p->data.data(), p->device_data, p->numel());
#endif
        }
        auto o = tensor({1}, p->device, p->requires_grad);
        float s = 0;
        for (size_t i = 0; i < p->numel(); i++)
        {
            float d = p->data[i] - t->data[i];
            s += d * d;
        }
        o->data[0] = s / p->numel();
        o->to_device();
        if (o->requires_grad)
            tape.push_back([o, p, t]()
                           {for(size_t i=0;i<p->numel();i++)p->grad[i]+=o->grad[0]*2*(p->data[i]-t->data[i])/p->numel(); });
        return o;
    }
}

