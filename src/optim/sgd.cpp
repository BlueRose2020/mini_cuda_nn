#include "mini_cuda_nn/optim/sgd.h"

#include "mini_cuda_nn/detail/dtype_dispatch.h"
#include "mini_cuda_nn/detail/sgd_kernel.h"

#include <type_traits>
#include <utility>

namespace
{
    // p -= lr * (g + weight_decay * p)
    template <typename T>
    void host_sgd_update(T *p, const T *g, double lr, double weight_decay, size_t n)
    {
        const T lr_t = static_cast<T>(lr);
        const T wd_t = static_cast<T>(weight_decay);
        for (size_t i = 0; i < n; ++i)
            p[i] -= lr_t * (g[i] + wd_t * p[i]);
    }

    // buf = momentum * buf + g + weight_decay * p;  p -= lr * buf
    template <typename T>
    void host_sgd_momentum_update(T *p, const T *g, T *buf,
                                  double lr, double momentum, double weight_decay, size_t n)
    {
        const T lr_t = static_cast<T>(lr);
        const T mom_t = static_cast<T>(momentum);
        const T wd_t = static_cast<T>(weight_decay);
        for (size_t i = 0; i < n; ++i)
        {
            const T b = mom_t * buf[i] + g[i] + wd_t * p[i];
            buf[i] = b;
            p[i] -= lr_t * b;
        }
    }
} // namespace

namespace mcn::optim
{
    SGD::SGD(std::vector<Tensor *> params, double lr, double momentum, double weight_decay)
        : Optimizer(std::move(params), lr),
          momentum_(momentum),
          weight_decay_(weight_decay)
    {
        MCN_CHECK(momentum >= 0.0 && momentum < 1.0,
                  "SGD: momentum must be in [0, 1), got {}", momentum);
        MCN_CHECK(weight_decay >= 0.0,
                  "SGD: weight_decay must be non-negative, got {}", weight_decay);

        // 构造期不能依赖虚派发，直接初始化一次
        on_params_added(0, this->params().size());
    }

    void SGD::on_params_added(size_t first, size_t count)
    {
        if (momentum_ == 0.0 || count == 0)
            return;

        // 动量缓冲与 params() 一一对应，设备/dtype/形状都跟着参数走
        momentum_buf_.resize(this->params().size());
        for (size_t i = first; i < this->params().size(); ++i)
        {
            const Tensor *p = this->params()[i];
            momentum_buf_[i] = Tensor::zeros(p->shape(), p->device(), p->dtype());
        }
    }

    void SGD::update_one(size_t index, Tensor &param, const Tensor &grad)
    {
        const size_t n = param.numel();
        if (n == 0)
            return;

        void *p = param.data_ptr();
        const void *g = grad.data_ptr();
        void *buf = momentum_buf_.empty() ? nullptr : momentum_buf_[index].data_ptr();

        if (param.device() != Device::CPU)
        {
            if (buf == nullptr)
                detail::launch_sgd_update(p, g, lr_, weight_decay_, n, param.dtype());
            else
                detail::launch_sgd_momentum_update(p, g, buf, lr_, momentum_, weight_decay_,
                                                   n, param.dtype());
            return;
        }

        detail::dispatch_dtype(param.dtype(), [&]<typename T>()
                               {
                                   if constexpr (!std::is_same_v<T, bool>)
                                   {
                                       if (buf == nullptr)
                                           host_sgd_update<T>(static_cast<T *>(p),
                                                              static_cast<const T *>(g),
                                                              lr_, weight_decay_, n);
                                       else
                                           host_sgd_momentum_update<T>(static_cast<T *>(p),
                                                                       static_cast<const T *>(g),
                                                                       static_cast<T *>(buf),
                                                                       lr_, momentum_, weight_decay_, n);
                                   }
                               });
    }
} // namespace mcn::optim
