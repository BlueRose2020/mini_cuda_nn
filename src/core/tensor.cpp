#include "mini_cuda_nn/core/tensor.h"
#include "mini_cuda_nn/core/cuda_check.h"
#include "mini_cuda_nn/core/dtype_dispatch.h"
#include "mini_cuda_nn/core/kernel_launch.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <type_traits>

namespace
{
    std::shared_ptr<void> make_host_ptr(size_t nbytes)
    {
        void *p = std::malloc(nbytes);
        MCN_CHECK(p != nullptr, "Failed to allocate host memory");
        return std::shared_ptr<void>(p, [](void *ptr)
                                     { std::free(ptr); });
    }

    std::shared_ptr<void> make_device_ptr(size_t nbytes)
    {
        void *p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, nbytes));
        return std::shared_ptr<void>(p, [](void *ptr)
                                     { CUDA_CHECK(cudaFree(ptr)); });
    }

    [[nodiscard("buffer must be stored, otherwise it is freed immediately")]]
    std::shared_ptr<void> make_buffer(size_t nbytes, mcn::Device device)
    {
        if (nbytes == 0)
            return nullptr;
        return device == mcn::Device::CPU ? make_host_ptr(nbytes) : make_device_ptr(nbytes);
    }

    std::vector<size_t> resolve_shape(const std::vector<int> &req, size_t numel, const char *who)
    {
        std::vector<size_t> out(req.size());
        size_t infer = SIZE_MAX;
        size_t known = 1;
        for (size_t i = 0; i < req.size(); ++i)
        {
            if (req[i] == -1)
            {
                MCN_CHECK(infer == SIZE_MAX, "{}: only one -1 is allowed", who);
                infer = i;
            }
            else
            {
                MCN_CHECK(req[i] >= 0, "{}: shape dim {} is negative", who, req[i]);
                out[i] = static_cast<size_t>(req[i]);
                known *= out[i];
            }
        }

        if (infer != SIZE_MAX)
        {
            MCN_CHECK(known != 0 && numel % known == 0,
                      "{}: cannot infer -1, numel={} is not divisible by {}", who, numel, known);
            out[infer] = numel / known;
        }
        else
        {
            MCN_CHECK(known == numel,
                      "{}: numel mismatch, requested {} but tensor has {}", who,
                      mcn::format_shape(out), numel);
        }
        return out;
    }

    // ---- 主机端逐元素算子 ----
    template <typename T>
    void host_fill(void *dst, double value, size_t n)
    {
        std::fill_n(static_cast<T *>(dst), n, static_cast<T>(value));
    }

    template <typename T>
    void host_add(void *dst, const void *src, size_t n)
    {
        T *d = static_cast<T *>(dst);
        const T *s = static_cast<const T *>(src);
        if constexpr (std::is_same_v<T, bool>)
        {
            for (size_t i = 0; i < n; ++i)
                d[i] = d[i] || s[i];
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
                d[i] += s[i];
        }
    }

    template <typename T>
    void host_binary(mcn::detail::BinaryOp op, void *dst, const void *a, const void *b, size_t n)
    {
        T *d = static_cast<T *>(dst);
        const T *x = static_cast<const T *>(a);
        const T *y = static_cast<const T *>(b);
        switch (op)
        {
        case mcn::detail::BinaryOp::Add:
            for (size_t i = 0; i < n; ++i)
                d[i] = x[i] + y[i];
            break;
        case mcn::detail::BinaryOp::Sub:
            for (size_t i = 0; i < n; ++i)
                d[i] = x[i] - y[i];
            break;
        case mcn::detail::BinaryOp::Mul:
            for (size_t i = 0; i < n; ++i)
                d[i] = x[i] * y[i];
            break;
        }
    }

    // 行优先矩阵乘法 out[M,N] = a[M,K] @ b[K,N]
    // 用 i-k-j 循环顺序：内层遍历 j 时 b 的行和 out 的行都是连续访问，
    // 比 i-j-k（内层每次跳 K 个元素）对缓存友好得多。
    template <typename T>
    void host_matmul(void *out, const void *a, const void *b, size_t M, size_t K, size_t N)
    {
        T *o = static_cast<T *>(out);
        const T *x = static_cast<const T *>(a);
        const T *y = static_cast<const T *>(b);

        std::fill_n(o, M * N, T(0));
        for (size_t i = 0; i < M; ++i)
        {
            T *orow = o + i * N;
            for (size_t k = 0; k < K; ++k)
            {
                const T aik = x[i * K + k];
                const T *yrow = y + k * N;
                for (size_t j = 0; j < N; ++j)
                    orow[j] += aik * yrow[j];
            }
        }
    }
}

namespace mcn
{
    Tensor::Tensor(const std::vector<size_t> &shape, const void *data, Device device,
                   DataType dtype, bool require_grad)
        : require_grad_(require_grad),
          size_(1),
          shape_(shape),
          device_(device),
          dtype_(dtype)
    {
        calculate_size();
        allocate_memory();
        copy_memory(data, Device::CPU);
    }

    Tensor::Tensor(const Tensor &other)
        : require_grad_(other.require_grad_),
          grad_(nullptr),
          grad_fn_(nullptr),
          offset_(0),
          size_(other.size_),
          shape_(other.shape_),
          device_(other.device_),
          dtype_(other.dtype_)
    {
        allocate_memory();

        if (other.size_ == 0)
            return;

        if (other.is_contiguous())
        {
            copy_memory(other.byte_ptr(0), other.device_);
        }
        else
        {
            MCN_CHECK(other.device_ == Device::CPU,
                      "copying a non-contiguous CUDA view is not supported yet");
            const size_t esz = itemsize(dtype_);
            const char *src = static_cast<const char *>(other.byte_ptr(0)); // 已包含 other.offset_
            char *dst = static_cast<char *>(data_.get());
            for (size_t i = 0; i < size_; ++i)
                std::memcpy(dst + i * esz, src + other.view_offset(i) * esz, esz);
        }
    }

    Tensor::~Tensor() = default;

    Tensor Tensor::empty(const std::vector<size_t> &shape, Device device, DataType dtype, bool require_grad)
    {
        Tensor result;
        result.require_grad_ = require_grad;
        result.shape_ = shape;
        result.device_ = device;
        result.dtype_ = dtype;
        result.calculate_size();
        result.allocate_memory();
        return result;
    }

    Tensor Tensor::zeros(const std::vector<size_t> &shape, Device device, DataType dtype, bool require_grad)
    {
        Tensor t = empty(shape, device, dtype, require_grad);
        t.fill_(0.0);
        return t;
    }

    Tensor Tensor::ones(const std::vector<size_t> &shape, Device device, DataType dtype, bool require_grad)
    {
        Tensor t = empty(shape, device, dtype, require_grad);
        t.fill_(1.0);
        return t;
    }

    Tensor Tensor::randn(const std::vector<size_t> &shape, Device device, DataType dtype, bool require_grad)
    {
        MCN_CHECK(is_floating_point(dtype), "randn: requires a floating point dtype, got {}", dtype_to_string(dtype));

        // 先在本机生成，再按需搬到设备
        Tensor t = empty(shape, Device::CPU, dtype, require_grad);
        if (t.size_ != 0)
        {
            static thread_local std::mt19937 gen{std::random_device{}()};
            std::normal_distribution<double> dist(0.0, 1.0);

            if (dtype == DataType::FLOAT32)
            {
                float *p = static_cast<float *>(t.byte_ptr(0));
                for (size_t i = 0; i < t.size_; ++i)
                    p[i] = static_cast<float>(dist(gen));
            }
            else
            {
                double *p = static_cast<double *>(t.byte_ptr(0));
                for (size_t i = 0; i < t.size_; ++i)
                    p[i] = dist(gen);
            }
        }

        if (device != Device::CPU)
            t.to_(device);
        return t;
    }

    // --------------元数据---------------
    void Tensor::calculate_size()
    {
        size_ = 1;
        for (size_t dim : shape_)
        {
            size_ *= dim;
        }
    }

    void Tensor::calculate_strides()
    {
        strides_.assign(shape_.size(), 1);
        for (size_t i = shape_.size(); i-- > 1;)
        {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
    }

    // --------------寻址---------------
    void *Tensor::byte_ptr(size_t elem) const
    {
        if (data_ == nullptr)
            return nullptr;
        return static_cast<char *>(data_.get()) + (offset_ + elem) * itemsize(dtype_);
    }

    size_t Tensor::view_offset(size_t linear) const
    {
        size_t offset = 0;
        for (size_t d = shape_.size(); d-- > 0;)
        {
            offset += (linear % shape_[d]) * strides_[d];
            linear /= shape_[d];
        }
        return offset;
    }

    bool Tensor::is_contiguous() const
    {
        size_t expected_stride = 1;
        for (size_t d = shape_.size(); d-- > 0;)
        {
            // 长度为 1 的维度不贡献跨步，任何 stride 都等价
            if (shape_[d] == 1)
                continue;
            if (strides_[d] != expected_stride)
                return false;
            expected_stride *= shape_[d];
        }
        return true;
    }

    // --------------内存管理---------------
    void Tensor::allocate_memory()
    {
        offset_ = 0;
        calculate_strides();
        data_ = make_buffer(size_ * itemsize(dtype_), device_);
    }

    void Tensor::copy_memory(const void *src, Device src_device)
    {
        if (src == nullptr || size_ == 0)
            return;
        void *dst = byte_ptr(0);
        const size_t nbytes = size_ * itemsize(dtype_);
        if (device_ == Device::CPU && src_device == Device::CPU)
            std::memcpy(dst, src, nbytes);
        else if (device_ == Device::CUDA && src_device == Device::CUDA)
            CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
        else if (device_ == Device::CPU && src_device == Device::CUDA)
            CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
        else if (device_ == Device::CUDA && src_device == Device::CPU)
            CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    }

    // --------------打印---------------
    void Tensor::append_elem_str(std::string &s, size_t elem) const
    {
        const void *elem_ptr = byte_ptr(elem);
        if (elem_ptr == nullptr)
        {
            s += "?";
            return;
        }
        switch (dtype_)
        {
        case DataType::FLOAT32:
            s += std::format("{:.4g}", *static_cast<const float *>(elem_ptr));
            break;
        case DataType::FLOAT64:
            s += std::format("{:.4g}", *static_cast<const double *>(elem_ptr));
            break;
        case DataType::INT32:
            s += std::format("{}", *static_cast<const int *>(elem_ptr));
            break;
        case DataType::INT64:
            s += std::format("{}", *static_cast<const int64_t *>(elem_ptr));
            break;
        case DataType::INT16:
            s += std::format("{}", *static_cast<const int16_t *>(elem_ptr));
            break;
        case DataType::INT8:
            s += std::format("{}", *static_cast<const int8_t *>(elem_ptr));
            break;
        case DataType::BOOL:
            s += std::format("{}", *static_cast<const bool *>(elem_ptr));
            break;
        default:
            break;
        }
    }

    void Tensor::append_dim_str(std::string &s, size_t dim, size_t offset) const
    {
        const bool is_leaf = (dim == shape_.size() - 1);
        s += "[";
        for (size_t i = 0; i < shape_[dim]; ++i)
        {
            const size_t child = offset + i * strides_[dim];
            if (is_leaf)
            {
                append_elem_str(s, child);
                if (i != shape_[dim] - 1)
                    s += ", ";
            }
            else
            {
                append_dim_str(s, dim + 1, child);
                if (i != shape_[dim] - 1)
                {
                    s += ",\n";
                }
            }
        }
        s += "]";
    }

    std::string Tensor::to_string() const
    {
        // 设备上的数据不能直接解引用：先搬一份到主机，只用于取值
        Tensor host;
        const Tensor *view = this;
        if (device_ != Device::CPU && size_ != 0)
        {
            MCN_CHECK(is_contiguous(), "printing a non-contiguous CUDA view is not supported yet");
            host.size_ = size_;
            host.shape_ = shape_;
            host.strides_ = strides_;
            host.dtype_ = dtype_;
            host.device_ = Device::CPU;
            host.offset_ = 0;
            host.data_ = make_buffer(size_ * itemsize(dtype_), Device::CPU);

            CUDA_CHECK(cudaMemcpy(host.data_.get(), byte_ptr(0),
                                  size_ * itemsize(dtype_), cudaMemcpyDeviceToHost));
            view = &host;
        }

        std::string tensor_str = "Tensor(";
        if (size_ == 0 || data_ == nullptr)
        {
            tensor_str += "[]";
        }
        else if (shape_.empty())
        {
            view->append_elem_str(tensor_str, 0);
        }
        else
        {
            view->append_dim_str(tensor_str, 0, 0);
        }
        if (dtype_ != DataType::FLOAT32)
        {
            tensor_str += ", dtype=";
            tensor_str += dtype_to_string(dtype_);
        }
        if (device_ != Device::CPU)
        {
            tensor_str += ", device=cuda";
        }
        if (require_grad_)
        {
            tensor_str += ", requires_grad=True";
        }
        tensor_str += ")";
        return tensor_str;
    }

    std::ostream &operator<<(std::ostream &os, const mcn::Tensor &tensor)
    {
        os << tensor.to_string();
        return os;
    }

    // --------------视图---------------
    Tensor Tensor::make_view() const
    {
        Tensor v;
        v.require_grad_ = require_grad_;
        v.grad_ = nullptr; // 视图不继承计算图
        v.grad_fn_ = nullptr;
        v.data_ = data_;     // 共享存储（引用计数 +1）
        v.offset_ = offset_; // 继承偏移，别写 0
        v.size_ = size_;
        v.shape_ = shape_;
        v.strides_ = strides_;
        v.device_ = device_;
        v.dtype_ = dtype_;
        return v;
    }

    // --------------操作符重载---------------
    Tensor Tensor::operator[](size_t index) const
    {
        MCN_CHECK(!shape_.empty(), "operator[] on a 0-dim tensor");
        MCN_CHECK(shape_[0] && index < shape_[0], "index {} out of range for dim 0 of size {}", index, shape_[0]);

        Tensor v = make_view();
        v.offset_ += index * strides_[0];
        v.size_ = size_ / shape_[0];
        v.shape_.erase(v.shape_.begin());
        v.strides_.erase(v.strides_.begin());
        return v;
    }

    Tensor Tensor::clone() const
    {
        return Tensor(*this);
    }

    // --------------原地操作---------------
    void Tensor::check_inplace(const char *who) const
    {
        MCN_CHECK(is_contiguous(), "{}: requires a contiguous tensor", who);
        MCN_CHECK(grad_fn_ == nullptr,
                  "{}: cannot modify a tensor that is part of an autograd graph in place", who);
    }

    Tensor &Tensor::fill_(double value)
    {
        check_inplace("fill_");
        if (size_ == 0)
            return *this;

        void *dst = byte_ptr(0);
        if (device_ != Device::CPU)
        {
            detail::launch_fill(dst, value, size_, dtype_);
            return *this;
        }

        detail::dispatch_dtype(dtype_, [&]<typename T>()
                               { host_fill<T>(dst, value, size_); });
        return *this;
    }

    Tensor &Tensor::zero_()
    {
        return fill_(0.0);
    }

    Tensor &Tensor::copy_(const Tensor &src)
    {
        check_inplace("copy_");
        MCN_CHECK(shape_ == src.shape_, "copy_: shape mismatch {} vs {}",
                  format_shape(shape_), format_shape(src.shape_));
        MCN_CHECK(dtype_ == src.dtype_, "copy_: dtype mismatch {} vs {}",
                  dtype_to_string(dtype_), dtype_to_string(src.dtype_));
        MCN_CHECK(src.is_contiguous(), "copy_: source must be contiguous");

        if (size_ == 0)
            return *this;

        void *dst = byte_ptr(0);
        const void *s = src.byte_ptr(0);
        const size_t nbytes = size_ * itemsize(dtype_);

        if (dst == s)
            return *this;

        if (device_ == Device::CPU && src.device_ == Device::CPU)
            std::memmove(dst, s, nbytes);
        else if (device_ == Device::CUDA && src.device_ == Device::CUDA)
            CUDA_CHECK(cudaMemcpy(dst, s, nbytes, cudaMemcpyDeviceToDevice));
        else if (device_ == Device::CUDA)
            CUDA_CHECK(cudaMemcpy(dst, s, nbytes, cudaMemcpyHostToDevice));
        else
            CUDA_CHECK(cudaMemcpy(dst, s, nbytes, cudaMemcpyDeviceToHost));
        return *this;
    }

    Tensor &Tensor::add_(const Tensor &other)
    {
        check_inplace("add_");
        MCN_CHECK(shape_ == other.shape_, "add_: shape mismatch {} vs {}",
                  format_shape(shape_), format_shape(other.shape_));
        MCN_CHECK(dtype_ == other.dtype_, "add_: dtype mismatch {} vs {}",
                  dtype_to_string(dtype_), dtype_to_string(other.dtype_));
        MCN_CHECK(device_ == other.device_, "add_: device mismatch");
        MCN_CHECK(other.is_contiguous(), "add_: other must be contiguous");

        if (size_ == 0)
            return *this;

        void *dst = byte_ptr(0);
        const void *s = other.byte_ptr(0);
        if (device_ != Device::CPU)
        {
            // 原地累加 = binary_add 且输出与第一个输入别名（读改写同一元素，无跨线程依赖）
            detail::launch_binary(detail::BinaryOp::Add, dst, dst, s, size_, dtype_);
            return *this;
        }

        detail::dispatch_dtype(dtype_, [&]<typename T>()
                               { host_add<T>(dst, s, size_); });
        return *this;
    }

    Tensor &Tensor::to_(Device device)
    {
        if (device == device_)
            return *this;

        MCN_CHECK(is_contiguous(), "to_: requires a contiguous tensor; clone() it first");
        const size_t nbytes = size_ * itemsize(dtype_);
        std::shared_ptr<void> new_data = make_buffer(nbytes, device);
        if (nbytes != 0)
        {
            if (device_ == Device::CPU)
                CUDA_CHECK(cudaMemcpy(new_data.get(), byte_ptr(0), nbytes, cudaMemcpyHostToDevice));
            else
                CUDA_CHECK(cudaMemcpy(new_data.get(), byte_ptr(0), nbytes, cudaMemcpyDeviceToHost));
        }

        data_ = std::move(new_data);
        offset_ = 0;
        device_ = device;
        calculate_strides();
        return *this;
    }

    Tensor Tensor::to(Device device) const
    {
        Tensor copy = clone();
        copy.to_(device);
        return copy;
    }

    Tensor Tensor::view(const std::vector<int> &new_shape) const
    {
        std::vector<size_t> shp = resolve_shape(new_shape, size_, "view");
        MCN_CHECK(is_contiguous(), "view() requires a contiguous tensor; use reshape() instead");

        Tensor v = make_view();
        v.shape_ = std::move(shp);
        v.calculate_strides(); // 连续布局重算 strides_
        return v;
    }

    Tensor Tensor::reshape(const std::vector<int> &new_shape) const
    {
        std::vector<size_t> shp = resolve_shape(new_shape, size_, "reshape");

        // 不连续时先物化成连续副本，再在此基础上改形状
        Tensor v = is_contiguous() ? make_view() : clone().make_view();
        v.shape_ = std::move(shp);
        v.calculate_strides();
        return v;
    }

    Tensor Tensor::transpose(int dim0, int dim1) const
    {
        const int nd = static_cast<int>(ndim());
        MCN_CHECK(nd >= 2, "transpose: requires at least 2 dims, got {}", format_shape(shape_));

        const int d0 = dim0 < 0 ? dim0 + nd : dim0;
        const int d1 = dim1 < 0 ? dim1 + nd : dim1;
        MCN_CHECK(d0 >= 0 && d0 < nd, "transpose: dim0 {} out of range for {}", dim0, format_shape(shape_));
        MCN_CHECK(d1 >= 0 && d1 < nd, "transpose: dim1 {} out of range for {}", dim1, format_shape(shape_));

        Tensor v = make_view();
        std::swap(v.shape_[d0], v.shape_[d1]);
        std::swap(v.strides_[d0], v.strides_[d1]);
        return v;
    }

    Tensor Tensor::t() const
    {
        MCN_CHECK(ndim() == 2, "t(): requires a 2-D tensor, got {}", format_shape(shape_));
        return transpose(0, 1);
    }

    Tensor Tensor::contiguous() const
    {
        return is_contiguous() ? make_view() : clone();
    }

    Tensor Tensor::binary_op(const Tensor &other, detail::BinaryOp op) const
    {
        MCN_CHECK(shape_ == other.shape_,
                  "{}: shape mismatch {} vs {}", op_name(op),
                  format_shape(shape_), format_shape(other.shape_));
        MCN_CHECK(device_ == other.device_, "{}: device mismatch", op_name(op));
        MCN_CHECK(dtype_ == other.dtype_, "{}: dtype mismatch {} vs {}", op_name(op),
                  dtype_to_string(dtype_), dtype_to_string(other.dtype_));
        MCN_CHECK(is_contiguous() && other.is_contiguous(),
                  "{}: requires contiguous tensors", op_name(op));

        Tensor out = empty(shape_, device_, dtype_, require_grad_ || other.require_grad_);
        if (size_ == 0)
            return out;

        if (device_ == Device::CPU)
            detail::dispatch_dtype(dtype_, [&]<typename T>()
                                   { host_binary<T>(op, out.byte_ptr(0), byte_ptr(0), other.byte_ptr(0), size_); });
        else
            detail::launch_binary(op, out.byte_ptr(0), byte_ptr(0), other.byte_ptr(0), size_, dtype_);
        return out;
    }

    constexpr const char *Tensor::op_name(detail::BinaryOp op)
    {
        switch (op)
        {
        case detail::BinaryOp::Add:
            return "add";
        case detail::BinaryOp::Sub:
            return "sub";
        case detail::BinaryOp::Mul:
            return "mul";
        }
        return "binary_op";
    }

    Tensor Tensor::operator+(const Tensor &other) const
    {
        return binary_op(other, detail::BinaryOp::Add);
    }

    Tensor Tensor::operator-(const Tensor &other) const
    {
        return binary_op(other, detail::BinaryOp::Sub);
    }

    Tensor Tensor::operator*(const Tensor &other) const
    {
        return binary_op(other, detail::BinaryOp::Mul);
    }

    Tensor Tensor::matmul(const Tensor &other) const
    {
        MCN_CHECK(!shape_.empty() && ndim() <= 2,
                  "matmul: lhs must be 1-D or 2-D, got {}", format_shape(shape_));
        MCN_CHECK(!other.shape_.empty() && other.ndim() <= 2,
                  "matmul: rhs must be 1-D or 2-D, got {}", format_shape(other.shape_));
        MCN_CHECK(device_ == other.device_, "matmul: device mismatch");
        MCN_CHECK(dtype_ == other.dtype_, "matmul: dtype mismatch {} vs {}",
                  dtype_to_string(dtype_), dtype_to_string(other.dtype_));
        MCN_CHECK(is_floating_point(dtype_), "matmul: requires a floating point dtype, got {}",
                  dtype_to_string(dtype_));

        // 非连续输入（例如 a.t()）自动物化；已连续时不产生拷贝
        const bool lhs_cloned = !is_contiguous();
        const bool rhs_cloned = !other.is_contiguous();
        const Tensor lhs_c = contiguous();
        const Tensor rhs_c = other.contiguous();

        // 1-D 张量按 PyTorch 语义处理：
        //   [K] @ [K,N] -> [N]
        //   [M,K] @ [K] -> [M]
        //   [K] @ [K]   -> 0 维标量
        const bool lhs_1d = (ndim() == 1);
        const bool rhs_1d = (other.ndim() == 1);

        const size_t M = lhs_1d ? 1 : shape_[0];
        const size_t K = lhs_1d ? shape_[0] : shape_[1];
        const size_t K2 = other.shape_[0];
        const size_t N = rhs_1d ? 1 : other.shape_[1];
        MCN_CHECK(K == K2, "matmul: inner dimensions do not match, {} vs {}",
                  format_shape(shape_), format_shape(other.shape_));

        std::vector<size_t> out_shape;
        if (lhs_1d && rhs_1d)
            out_shape = {};
        else if (lhs_1d)
            out_shape = {N};
        else if (rhs_1d)
            out_shape = {M};
        else
            out_shape = {M, N};

        Tensor out = empty(out_shape, device_, dtype_, require_grad_ || other.require_grad_);
        if (out.size_ == 0)
            return out;

        void *dst = out.byte_ptr(0);
        const void *lhs = lhs_c.data_ptr();
        const void *rhs = rhs_c.data_ptr();

        if (device_ == Device::CPU)
        {
            // matmul 只支持浮点（上面已校验），bool 分支不实例化，避免 bool 算术告警
            detail::dispatch_dtype(dtype_, [&]<typename T>()
                                   {
                                       if constexpr (!std::is_same_v<T, bool>)
                                           host_matmul<T>(dst, lhs, rhs, M, K, N); });
        }
        else
        {
            detail::launch_matmul(dst, lhs, rhs, M, K, N, dtype_);
            // kernel 是异步的；若上面 clone 出了临时 device buffer，
            // 它会在函数返回时释放，必须先等 kernel 读完
            if (lhs_cloned || rhs_cloned)
                CUDA_CHECK(cudaDeviceSynchronize());
        }
        return out;
    }
}