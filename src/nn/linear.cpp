#include "mini_cuda_nn/nn/linear.h"

#include "mini_cuda_nn/autograd/grad_mode.h"
#include "mini_cuda_nn/detail/dtype_dispatch.h"

#include <cmath>
#include <random>
#include <type_traits>

namespace
{
    // 初始化用的引擎。thread_local + 可重置，方便复现训练。
    std::mt19937 &init_rng()
    {
        static thread_local std::mt19937 gen{std::random_device{}()};
        return gen;
    }

    // PyTorch 的默认初始化：U(-1/sqrt(in), 1/sqrt(in))
    template <typename T>
    void uniform_init(mcn::Tensor &t, double bound)
    {
        std::uniform_real_distribution<double> dist(-bound, bound);

        T *p = t.data_ptr<T>();
        for (size_t i = 0; i < t.numel(); ++i)
            p[i] = static_cast<T>(dist(init_rng()));
    }
} // namespace

namespace mcn::nn
{
    void manual_seed(unsigned seed)
    {
        init_rng().seed(seed);
    }

    Linear::Linear(size_t in_features, size_t out_features,
                   bool require_grad, Device device, DataType dtype)
        : in_features_(in_features),
          out_features_(out_features)
    {
        MCN_CHECK(in_features > 0 && out_features > 0,
                  "Linear: in_features and out_features must be positive, got {} -> {}",
                  in_features, out_features);
        MCN_CHECK(is_floating_point(dtype),
                  "Linear: requires a floating point dtype, got {}", dtype_to_string(dtype));

        weight_ = Tensor::empty({out_features, in_features}, Device::CPU, dtype, require_grad);
        bias_ = Tensor::empty({out_features}, Device::CPU, dtype, require_grad);

        // 初始化一律在 CPU 上做（主机端只能直接写 CPU 内存），再搬到目标设备
        const double bound = 1.0 / std::sqrt(static_cast<double>(in_features));
        detail::dispatch_dtype(dtype, [&]<typename T>()
                               {
                                   if constexpr (!std::is_same_v<T, bool>)
                                   {
                                       uniform_init<T>(weight_, bound);
                                       uniform_init<T>(bias_, bound);
                                   }
                               });

        if (device != Device::CPU)
        {
            weight_.to_(device);
            bias_.to_(device);
        }
    }

    Tensor Linear::forward(const Tensor &x)
    {
        MCN_CHECK(x.ndim() == 2,
                  "Linear: expected a 2-D input [B, in_features], got {}", format_shape(x.shape()));
        MCN_CHECK(x.shape()[1] == in_features_,
                  "Linear: expected {} input features, got {}",
                  in_features_, x.shape()[1]);
        MCN_CHECK(x.device() == weight_.device(),
                  "Linear: input and weight are on different devices "
                  "(call Module::to() on the whole network, not just the input)");
        MCN_CHECK(x.dtype() == weight_.dtype(),
                  "Linear: input dtype {} != weight dtype {}",
                  dtype_to_string(x.dtype()), dtype_to_string(weight_.dtype()));

        return x.matmul(weight_.t()) + bias_;
    }
} // namespace mcn::nn
