#include "mini_cuda_nn/core/tensor.h"
#include "mini_cuda_nn/core/cuda_check.h"

#include <cstring>

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

    // --------------操作符重载---------------
    Tensor Tensor::operator[](size_t index) const
    {
        MCN_CHECK(!shape_.empty(), "operator[] on a 0-dim tensor");
        MCN_CHECK(shape_[0] && index < shape_[0], "index {} out of range for dim 0 of size {}", index, shape_[0]);

        Tensor view;
        view.require_grad_ = require_grad_;
        view.grad_ = nullptr;
        view.grad_fn_ = nullptr;
        view.data_ = data_;
        view.offset_ = offset_ + index * strides_[0];
        view.size_ = size_ / shape_[0];
        view.shape_.assign(shape_.begin() + 1, shape_.end());
        view.strides_.assign(strides_.begin() + 1, strides_.end());
        view.device_ = device_;
        view.dtype_ = dtype_;
        return view;
    }

    Tensor Tensor::clone() const
    {
        return Tensor(*this);
    }

    void Tensor::to(Device device)
    {
        if (device == device_)
            return;

        MCN_CHECK(is_contiguous(), "to() requires a contiguous tensor; clone() it first");
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
    }
}