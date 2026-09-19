#include "internal.hpp"
namespace mininn
{
    void sgd_step(const std::vector<TensorPtr> &ps, float lr)
    {
        for (auto &p : ps)
        {
            p->to_host();
            for (size_t i = 0; i < p->numel(); i++)
                p->data[i] -= lr * p->grad[i];
            p->to_device();
        }
    }
    void hybrid_sgd_step(const std::vector<TensorPtr> &ps, float lr)
    {
        for (auto &p : ps)
        {
#ifdef MINI_CUDA
            if (p->device == Device::CUDA)
            {
                mininn_cuda_upload(p->device_grad, p->grad.data(), p->numel());
                mininn_cuda_sgd((float *)p->device_data, (const float *)p->device_grad, lr, (int)p->numel());
                mininn_cuda_download(p->data.data(), p->device_data, p->numel());
                continue;
            }
#endif
            sgd_step({p}, lr);
        }
    }
}

