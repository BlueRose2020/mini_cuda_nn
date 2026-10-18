#pragma once
#include <vector>
#include <memory>
#include <string>
#include <format>
#include <cstdio>
#include <cstdlib>
#include <ostream>

#include "device.h"
#include "dtype.h"
#include "mini_cuda_nn/autograd/grad_fn.h"
#include "mini_cuda_nn/core/cuda_check.h"

#define MCN_UNUSED_EXPRESSION [[nodiscard("expression result is unused - did you forget to use the returned Tensor?")]]
#define MCN_UNUSED_RETURN(func) [[nodiscard("result of " #func " is unused")]]
#define MCN_CHECK(cond, ...)                                  \
    do {                                                      \
        if (!(cond)) {                                        \
            std::fputs(std::format("MCN check failed: {} at {}:{}\n",  \
                                   #cond, __FILE__, __LINE__).c_str(), stderr); \
            std::fputs(std::format(__VA_ARGS__).c_str(), stderr);   \
            std::fputc('\n', stderr);                         \
            std::abort();                                     \
        }                                                     \
    } while (0)

namespace mcn
{
    inline std::string format_shape(const std::vector<size_t> &shape)
    {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size())
                s += ", ";
        }
        s += "]";
        return s;
    }

    struct Slice
    {
        int start;
        int stop;
        int step;
        Slice(int start = 0, int stop = -1, int step = 1) : start(start), stop(stop), step(step) {}
    };

    struct All
    {
    };
    inline constexpr All all{};

    template <typename T>
    struct is_initializer_list : std::false_type
    {
    };
    template <typename T>
    struct is_initializer_list<std::initializer_list<T>> : std::true_type
    {
    };
    template <typename T>
    constexpr bool is_initializer_list_v = is_initializer_list<T>::value;

    // 类型萃取工具，用于获取嵌套初始化列表的叶子类型
    template <typename T>
    struct leaf_type
    {
        using type = T;
    };

    template <typename T>
    struct leaf_type<std::initializer_list<T>>
    {
        using type = typename leaf_type<T>::type;
    };

    template <typename T>
    using leaf_type_t = typename leaf_type<T>::type;

    class Tensor
    {
    public:
        Tensor() noexcept = default;
        template <typename T>
            requires std::is_arithmetic_v<leaf_type_t<T>>
        explicit Tensor(const std::initializer_list<T> &src, Device device = Device::CPU,
                        DataType dtype = DataType::FLOAT32, bool require_grad = false)
            : require_grad_(require_grad),
              device_(device),
              dtype_(dtype_of_v<leaf_type_t<T>>)
        {
            deduce_shape(src, shape_);
            std::vector<leaf_type_t<T>> flat_data;
            flatten_data(src, flat_data);
            size_ = flat_data.size();
            allocate_memory();
            copy_memory(flat_data.data(), Device::CPU);
        }
        template <typename T>
        explicit Tensor(const std::initializer_list<std::initializer_list<T>> &src, Device device = Device::CPU,
                        DataType dtype = DataType::FLOAT32, bool require_grad = false)
            : require_grad_(require_grad),
              device_(device),
              dtype_(dtype_of_v<leaf_type_t<T>>)
        {
            deduce_shape(src, shape_);
            std::vector<leaf_type_t<T>> flat_data;
            flatten_data(src, flat_data);
            size_ = flat_data.size();
            allocate_memory();
            copy_memory(flat_data.data(), Device::CPU);
        }
        template <typename T>
            requires std::is_arithmetic_v<T>
        explicit Tensor(T value, Device device = Device::CPU, bool require_grad = false)
            : require_grad_(require_grad),
              shape_({}),
              device_(device),
              dtype_(dtype_of_v<T>)
        {
            size_ = 1;
            allocate_memory();
            copy_memory(&value, Device::CPU);
        }
        Tensor(const std::vector<size_t> &shape, const void *data, Device device = Device::CPU,
               DataType dtype = DataType::FLOAT32, bool require_grad = false);
        Tensor(const Tensor &other);
        Tensor(Tensor &&other) noexcept = default;
        ~Tensor();

    private:
        template <typename T>
        void deduce_shape(const std::initializer_list<T> &list, std::vector<size_t> &shape)
        {
            shape.push_back(list.size());
            if constexpr (is_initializer_list_v<T>)
            {
                deduce_shape(*list.begin(), shape);
            }
        }
        template <typename ListT, typename T>
        void flatten_data(const std::initializer_list<ListT> &list, std::vector<T> &data)
        {
            if constexpr (is_initializer_list_v<ListT>)
            {
                for (const auto &sublist : list)
                {
                    flatten_data(sublist, data);
                }
            }
            else
            {
                for (const auto &elem : list)
                {
                    data.push_back(static_cast<T>(elem));
                }
            }
        }

    public:
        // 运算符重载
        Tensor &operator=(const Tensor &other) = default;
        Tensor &operator=(Tensor &&other) noexcept = default;
        MCN_UNUSED_EXPRESSION
        Tensor operator+(const Tensor &other) const;
        MCN_UNUSED_EXPRESSION
        Tensor operator-(const Tensor &other) const;
        MCN_UNUSED_EXPRESSION
        Tensor operator*(const Tensor &other) const;
        template <typename T>
        MCN_UNUSED_EXPRESSION
            Tensor
            operator*(const T &scalar) const;
        friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
        // template <typename T>
        // MCN_UNUSED_EXPRESSION
        // friend Tensor operator*(const T &scalar, const Tensor &tensor);
        Tensor operator[](size_t index) const;

        // 静态方法
        MCN_UNUSED_RETURN(zeros)
        static Tensor zeros(const std::vector<size_t> &shape, Device device = Device::CPU, DataType dtype = DataType::FLOAT32, bool require_grad = false);
        MCN_UNUSED_RETURN(ones)
        static Tensor ones(const std::vector<size_t> &shape, Device device = Device::CPU, DataType dtype = DataType::FLOAT32, bool require_grad = false);
        MCN_UNUSED_RETURN(randn)
        static Tensor randn(const std::vector<size_t> &shape, Device device = Device::CPU, DataType dtype = DataType::FLOAT32, bool require_grad = false);
        MCN_UNUSED_RETURN(empty)
        static Tensor empty(const std::vector<size_t> &shape, Device device = Device::CPU, DataType dtype = DataType::FLOAT32, bool require_grad = false);

        // 梯度相关方法
        bool require_grad() const { return require_grad_; };
        const Tensor *grad() const { return grad_.get(); };
        std::shared_ptr<GradFn> grad_fn() const { return grad_fn_; };
        void set_grad_fn(std::shared_ptr<GradFn> grad_fn) { grad_fn_ = grad_fn; };

        void backward();

        // 形状操作
        MCN_UNUSED_RETURN(reshape)
        Tensor reshape(const std::vector<int> &new_shape) const;
        MCN_UNUSED_RETURN(view)
        Tensor view(const std::vector<int> &new_shape) const;

        // 属性访问
        std::vector<size_t> shape() const { return shape_; };
        size_t size() const { return size_; };
        Device device() const { return device_; };
        DataType dtype() const { return dtype_; };
        size_t numel() const { return size_; };
        size_t ndim() const { return shape_.size(); };

        // 数据访问
        void *data_ptr() { return byte_ptr(0); }
        const void *data_ptr() const { return byte_ptr(0); }
        template <typename T>
        T *data_ptr() { return static_cast<T *>(byte_ptr(0)); }
        template <typename T>
        const T *data_ptr() const { return static_cast<const T *>(byte_ptr(0)); }
        //
        void to(Device device);
        Tensor clone() const;

    private:
        void calculate_size();
        void append_elem_str(std::string &s, size_t idx) const;
        void append_dim_str(std::string &s, size_t dim, size_t offset) const;
        std::string to_string() const;
        void allocate_memory();
        void copy_memory(const void *src, Device src_device);
        // void free_memory();
        void calculate_strides();
        void *byte_ptr(size_t elem) const;
        size_t view_offset(size_t linear) const;
        bool is_contiguous() const;

    private:
        bool require_grad_ = false;
        std::shared_ptr<Tensor> grad_ = nullptr;
        std::shared_ptr<GradFn> grad_fn_ = nullptr;

        std::shared_ptr<void> data_;
        size_t offset_ = 0;
        size_t size_ = 1;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        Device device_ = Device::CPU;
        DataType dtype_ = DataType::FLOAT32;
    };
} // namespace mcn
