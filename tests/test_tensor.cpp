// Tensor 单元测试：构造、元数据、视图、reshape、clone、设备迁移
#include "mini_cuda_nn/core/tensor.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace
{
    int g_checks = 0;
    int g_failures = 0;

    void fail(const char *file, int line, const std::string &msg)
    {
        std::printf("    [FAIL] %s:%d  %s\n", file, line, msg.c_str());
        ++g_failures;
    }

    void check_true(bool cond, const char *expr, const char *file, int line)
    {
        ++g_checks;
        if (!cond)
            fail(file, line, std::string("expected true: ") + expr);
    }

    void check_shape(const mcn::Tensor &t, std::vector<size_t> expect, const char *file, int line)
    {
        ++g_checks;
        if (t.shape() != expect)
            fail(file, line, "shape " + mcn::format_shape(t.shape()) +
                                 " != expected " + mcn::format_shape(expect));
    }

    void check_numel(const mcn::Tensor &t, size_t expect, const char *file, int line)
    {
        ++g_checks;
        if (t.numel() != expect)
            fail(file, line, "numel " + std::to_string(t.numel()) +
                                 " != expected " + std::to_string(expect));
    }

    void check_float(float got, float expect, float eps, const char *file, int line)
    {
        ++g_checks;
        const float diff = got > expect ? got - expect : expect - got;
        if (!(diff <= eps))
            fail(file, line, "value " + std::to_string(got) +
                                 " != expected " + std::to_string(expect));
    }

    void check_int(long long got, long long expect, const char *file, int line)
    {
        ++g_checks;
        if (got != expect)
            fail(file, line, "value " + std::to_string(got) +
                                 " != expected " + std::to_string(expect));
    }

    void check_dtype(const mcn::Tensor &t, mcn::DataType expect, const char *file, int line)
    {
        ++g_checks;
        if (t.dtype() != expect)
            fail(file, line, std::string("dtype ") + mcn::dtype_to_string(t.dtype()) +
                                 " != expected " + mcn::dtype_to_string(expect));
    }

    void check_device(const mcn::Tensor &t, mcn::Device expect, const char *file, int line)
    {
        ++g_checks;
        const bool ok = (t.device() == expect);
        if (!ok)
            fail(file, line, std::string("device mismatch"));
    }

    // 按索引逐层取子张量，再从标量张量读值。
    // 只能用于 CPU 张量（设备内存不能直接解引用）。
    template <typename T>
    T value_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        mcn::Tensor cur = t; // 深拷贝，安全（也顺带覆盖拷贝构造）
        for (size_t i : idx)
            cur = cur[i];
        return *static_cast<const T *>(cur.data_ptr());
    }

    float f_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        return value_at<float>(t, idx);
    }

    int i_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        return value_at<int>(t, idx);
    }
}

#define CHECK_TRUE(expr) ::check_true((expr), #expr, __FILE__, __LINE__)
#define CHECK_SHAPE(t, ...) ::check_shape((t), std::vector<size_t>{__VA_ARGS__}, __FILE__, __LINE__)
#define CHECK_NUMEL(t, n) ::check_numel((t), (n), __FILE__, __LINE__)
#define CHECK_FLOAT(got, expect, eps) ::check_float((got), (expect), (eps), __FILE__, __LINE__)
#define CHECK_FLOAT_EQ(got, expect) ::check_float((got), (expect), 1e-6f, __FILE__, __LINE__)
#define CHECK_INT(got, expect) ::check_int((got), (expect), __FILE__, __LINE__)
#define CHECK_DTYPE(t, d) ::check_dtype((t), (d), __FILE__, __LINE__)
#define CHECK_DEVICE(t, d) ::check_device((t), (d), __FILE__, __LINE__)

namespace
{
    // ---------------- 构造 ----------------

    void test_scalar()
    {
        std::puts("  test_scalar");

        mcn::Tensor f(2.5f);
        CHECK_TRUE(f.ndim() == 0);
        CHECK_TRUE(f.shape().empty());
        CHECK_NUMEL(f, 1);
        CHECK_DTYPE(f, mcn::DataType::FLOAT32);
        CHECK_DEVICE(f, mcn::Device::CPU);
        CHECK_FLOAT_EQ(*static_cast<const float *>(f.data_ptr()), 2.5f);

        mcn::Tensor i(7);
        CHECK_TRUE(i.shape().empty());
        CHECK_DTYPE(i, mcn::DataType::INT32);
        CHECK_INT(*static_cast<const int *>(i.data_ptr()), 7);

        mcn::Tensor d(1.5);
        CHECK_DTYPE(d, mcn::DataType::FLOAT64);
        CHECK_FLOAT_EQ(static_cast<float>(*static_cast<const double *>(d.data_ptr())), 1.5f);
    }

    void test_init_list()
    {
        std::puts("  test_init_list");

        mcn::Tensor a({1.f, 2.f, 3.f});
        CHECK_SHAPE(a, 3);
        CHECK_NUMEL(a, 3);
        CHECK_DTYPE(a, mcn::DataType::FLOAT32);
        CHECK_FLOAT_EQ(f_at(a, {0}), 1.f);
        CHECK_FLOAT_EQ(f_at(a, {2}), 3.f);

        mcn::Tensor b({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        CHECK_SHAPE(b, 2, 3);
        CHECK_NUMEL(b, 6);
        CHECK_TRUE(b.ndim() == 2);
        CHECK_FLOAT_EQ(f_at(b, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(b, {1, 2}), 6.f);

        // int 列表推断成 INT32
        mcn::Tensor c({1, 2, 3});
        CHECK_DTYPE(c, mcn::DataType::INT32);
        CHECK_INT(i_at(c, {1}), 2);
    }

    void test_shape_data()
    {
        std::puts("  test_shape_data");

        const float buf[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        const std::vector<size_t> shape{2, 3};
        mcn::Tensor t(shape, buf);

        CHECK_SHAPE(t, 2, 3);
        CHECK_NUMEL(t, 6);
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(t, {1, 1}), 5.f);
        CHECK_FLOAT_EQ(f_at(t, {1, 2}), 6.f);
    }

    void test_empty()
    {
        std::puts("  test_empty");

        mcn::Tensor e = mcn::Tensor::empty({2, 3});
        CHECK_SHAPE(e, 2, 3);
        CHECK_NUMEL(e, 6);
        CHECK_TRUE(e.data_ptr() != nullptr);

        // 0 元素张量不应分配内存
        mcn::Tensor z = mcn::Tensor::empty({0, 3});
        CHECK_NUMEL(z, 0);
        CHECK_TRUE(z.data_ptr() == nullptr);
        CHECK_DEVICE(z, mcn::Device::CPU);
    }

    // ---------------- 视图 ----------------

    void test_index()
    {
        std::puts("  test_index");

        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});

        mcn::Tensor row = m[1];
        CHECK_SHAPE(row, 3);
        CHECK_NUMEL(row, 3);
        CHECK_FLOAT_EQ(f_at(row, {0}), 4.f);
        CHECK_FLOAT_EQ(f_at(row, {2}), 6.f);

        // operator[] 必须是共享存储的视图，而不是拷贝
        static_cast<float *>(row.data_ptr())[0] = 40.f;
        CHECK_FLOAT_EQ(f_at(m, {1, 0}), 40.f);

        // 逐层取到 0 维标量
        mcn::Tensor s = m[0][2];
        CHECK_TRUE(s.shape().empty());
        CHECK_NUMEL(s, 1);
        CHECK_FLOAT_EQ(*static_cast<const float *>(s.data_ptr()), 3.f);
    }

    void test_view()
    {
        std::puts("  test_view");

        mcn::Tensor a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

        mcn::Tensor v = a.view({2, 3});
        CHECK_SHAPE(v, 2, 3);
        CHECK_NUMEL(v, 6);
        CHECK_FLOAT_EQ(f_at(v, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(v, {1, 0}), 4.f);

        // 视图共享存储
        static_cast<float *>(v.data_ptr())[0] = 99.f;
        CHECK_FLOAT_EQ(f_at(a, {0}), 99.f);

        // 带 offset_ 的视图再 view：不能丢掉偏移
        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        mcn::Tensor offset_view = m[1].view({3, 1});
        CHECK_SHAPE(offset_view, 3, 1);
        CHECK_FLOAT_EQ(f_at(offset_view, {0, 0}), 4.f);
        CHECK_FLOAT_EQ(f_at(offset_view, {2, 0}), 6.f);
    }

    void test_reshape()
    {
        std::puts("  test_reshape");

        mcn::Tensor a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

        mcn::Tensor r1 = a.reshape({2, 3});
        CHECK_SHAPE(r1, 2, 3);
        CHECK_FLOAT_EQ(f_at(r1, {1, 2}), 6.f);

        mcn::Tensor r2 = a.reshape({3, 2});
        CHECK_SHAPE(r2, 3, 2);
        CHECK_FLOAT_EQ(f_at(r2, {2, 1}), 6.f);

        // -1 推断
        mcn::Tensor r3 = a.reshape({3, -1});
        CHECK_SHAPE(r3, 3, 2);

        mcn::Tensor r4 = a.reshape({-1, 2});
        CHECK_SHAPE(r4, 3, 2);
        CHECK_FLOAT_EQ(f_at(r4, {0, 1}), 2.f);

        // reshape 出的张量共享存储
        static_cast<float *>(r1.data_ptr())[0] = 77.f;
        CHECK_FLOAT_EQ(f_at(a, {0}), 77.f);
    }

    // ---------------- 拷贝 ----------------

    void test_clone()
    {
        std::puts("  test_clone");

        mcn::Tensor m({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor c = m.clone();

        CHECK_SHAPE(c, 2, 2);
        CHECK_FLOAT_EQ(f_at(c, {1, 1}), 4.f);

        // clone 是深拷贝，互不影响
        static_cast<float *>(c.data_ptr())[0] = 9.f;
        CHECK_FLOAT_EQ(f_at(c, {0, 0}), 9.f);
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 1.f);

        // clone 一个视图，结果应物化为连续副本
        mcn::Tensor v = m[1].clone();
        CHECK_SHAPE(v, 2);
        CHECK_FLOAT_EQ(f_at(v, {0}), 3.f);
        CHECK_FLOAT_EQ(f_at(v, {1}), 4.f);
    }

    // ---------------- 设备迁移 ----------------

    void test_to_device()
    {
        std::puts("  test_to_device");

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            std::puts("    (skip: no CUDA device)");
            return;
        }

        mcn::Tensor t({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        t.to(mcn::Device::CUDA);
        CHECK_DEVICE(t, mcn::Device::CUDA);
        CHECK_SHAPE(t, 2, 3);

        t.to(mcn::Device::CPU);
        CHECK_DEVICE(t, mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(t, {1, 2}), 6.f);

        // 带 offset_ 的视图往返，验证偏移没被丢掉
        mcn::Tensor row = t[1];
        row.to(mcn::Device::CUDA);
        row.to(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(row, {0}), 4.f);
        CHECK_FLOAT_EQ(f_at(row, {2}), 6.f);
    }
}

int main()
{
    std::puts("=== tensor tests ===");

    test_scalar();
    test_init_list();
    test_shape_data();
    test_empty();
    test_index();
    test_view();
    test_reshape();
    test_clone();
    test_to_device();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
