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

    // ---------------- 填充 / 原地操作 ----------------

    void test_zeros_ones()
    {
        std::puts("  test_zeros_ones");

        mcn::Tensor z = mcn::Tensor::zeros({2, 3});
        CHECK_SHAPE(z, 2, 3);
        CHECK_DTYPE(z, mcn::DataType::FLOAT32);
        CHECK_FLOAT_EQ(f_at(z, {0, 0}), 0.f);
        CHECK_FLOAT_EQ(f_at(z, {1, 2}), 0.f);

        mcn::Tensor o = mcn::Tensor::ones({2, 2});
        CHECK_FLOAT_EQ(f_at(o, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(o, {1, 1}), 1.f);

        // 非 float32 dtype
        mcn::Tensor zi = mcn::Tensor::zeros({3}, mcn::Device::CPU, mcn::DataType::INT32);
        CHECK_DTYPE(zi, mcn::DataType::INT32);
        CHECK_INT(i_at(zi, {0}), 0);
        CHECK_INT(i_at(zi, {2}), 0);

        mcn::Tensor oi = mcn::Tensor::ones({3}, mcn::Device::CPU, mcn::DataType::INT32);
        CHECK_INT(i_at(oi, {1}), 1);

        // require_grad 标记应被尊重
        mcn::Tensor g = mcn::Tensor::zeros({2}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        CHECK_TRUE(g.require_grad());

        // 0 元素张量不应崩
        mcn::Tensor e = mcn::Tensor::zeros({0, 3});
        CHECK_NUMEL(e, 0);
    }

    void test_fill()
    {
        std::puts("  test_fill");

        mcn::Tensor t = mcn::Tensor::empty({2, 2});
        t.fill_(3.5);
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 3.5f);
        CHECK_FLOAT_EQ(f_at(t, {1, 1}), 3.5f);

        // 返回自身引用，可链式
        t.fill_(1.0).fill_(2.0);
        CHECK_FLOAT_EQ(f_at(t, {0, 1}), 2.f);

        t.zero_();
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 0.f);

        // 视图上原地填充，应写回源张量
        mcn::Tensor m({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor row = m[1];
        row.fill_(9.0);
        CHECK_FLOAT_EQ(f_at(m, {1, 0}), 9.f);
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 1.f);

        mcn::Tensor i = mcn::Tensor::empty({2}, mcn::Device::CPU, mcn::DataType::INT32);
        i.fill_(7.0);
        CHECK_INT(i_at(i, {1}), 7);
    }

    void test_copy()
    {
        std::puts("  test_copy");

        mcn::Tensor src({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor dst = mcn::Tensor::zeros({2, 2});

        dst.copy_(src);
        CHECK_FLOAT_EQ(f_at(dst, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(dst, {1, 1}), 4.f);

        static_cast<float *>(dst.data_ptr())[0] = 9.f;
        CHECK_FLOAT_EQ(f_at(src, {0, 0}), 1.f); // 深拷贝，互不影响

        // 自拷不应崩
        dst.copy_(dst);
        CHECK_FLOAT_EQ(f_at(dst, {0, 0}), 9.f);
    }

    void test_add()
    {
        std::puts("  test_add");

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b({{10.f, 20.f}, {30.f, 40.f}});

        a.add_(b);
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(a, {1, 1}), 44.f);

        // 累加两次
        a.add_(b);
        CHECK_FLOAT_EQ(f_at(a, {0, 1}), 42.f);

        mcn::Tensor ia = mcn::Tensor({1, 2, 3});
        mcn::Tensor ib = mcn::Tensor({4, 5, 6});
        ia.add_(ib);
        CHECK_INT(i_at(ia, {0}), 5);
        CHECK_INT(i_at(ia, {2}), 9);
    }

    void test_randn()
    {
        std::puts("  test_randn");

        mcn::Tensor r = mcn::Tensor::randn({256});
        CHECK_SHAPE(r, 256);
        CHECK_DTYPE(r, mcn::DataType::FLOAT32);

        // 粗略统计检查：均值应接近 0，且存在非零元素
        const float *p = static_cast<const float *>(r.data_ptr());
        double sum = 0.0;
        int nonzero = 0;
        for (size_t i = 0; i < r.numel(); ++i)
        {
            sum += p[i];
            if (p[i] != 0.f)
                ++nonzero;
        }
        const double mean = sum / static_cast<double>(r.numel());
        CHECK_TRUE(mean > -0.3 && mean < 0.3);
        CHECK_TRUE(nonzero > static_cast<int>(r.numel()) / 2);

        mcn::Tensor rd = mcn::Tensor::randn({4}, mcn::Device::CPU, mcn::DataType::FLOAT64);
        CHECK_DTYPE(rd, mcn::DataType::FLOAT64);
    }

    // ---------------- 二元算子 ----------------

    void test_binary_ops()
    {
        std::puts("  test_binary_ops");

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b({{10.f, 20.f}, {30.f, 40.f}});

        mcn::Tensor sum = a + b;
        CHECK_SHAPE(sum, 2, 2);
        CHECK_FLOAT_EQ(f_at(sum, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(sum, {1, 1}), 44.f);

        mcn::Tensor diff = a - b;
        CHECK_FLOAT_EQ(f_at(diff, {0, 1}), -18.f);
        CHECK_FLOAT_EQ(f_at(diff, {1, 0}), -27.f);

        mcn::Tensor prod = a * b;
        CHECK_FLOAT_EQ(f_at(prod, {0, 0}), 10.f);
        CHECK_FLOAT_EQ(f_at(prod, {1, 1}), 160.f);

        // 原张量不应被修改
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(b, {0, 0}), 10.f);

        // int 类型
        mcn::Tensor ia({1, 2, 3});
        mcn::Tensor ib({4, 5, 6});
        mcn::Tensor isum = ia + ib;
        CHECK_DTYPE(isum, mcn::DataType::INT32);
        CHECK_INT(i_at(isum, {0}), 5);
        CHECK_INT(i_at(isum, {2}), 9);

        // require_grad 应传播
        mcn::Tensor g1 = mcn::Tensor::ones({2}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor g2 = mcn::Tensor::ones({2});
        CHECK_TRUE((g1 + g2).require_grad());
        CHECK_TRUE((g2 + g2).require_grad() == false);
    }

    // ---------------- 矩阵乘法 ----------------

    // 已知结果的手算用例：
    //   [1 2]   [5 6]     [19 22]
    //   [3 4] @ [7 8]  =  [43 50]
    void check_known_matmul(const mcn::Tensor &a, const mcn::Tensor &b)
    {
        mcn::Tensor c = a.matmul(b);
        CHECK_SHAPE(c, 2, 2);
        CHECK_DTYPE(c, mcn::DataType::FLOAT32);
        CHECK_FLOAT_EQ(f_at(c, {0, 0}), 19.f);
        CHECK_FLOAT_EQ(f_at(c, {0, 1}), 22.f);
        CHECK_FLOAT_EQ(f_at(c, {1, 0}), 43.f);
        CHECK_FLOAT_EQ(f_at(c, {1, 1}), 50.f);
    }

    void test_matmul()
    {
        std::puts("  test_matmul");

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b({{5.f, 6.f}, {7.f, 8.f}});

        // 2D @ 2D
        check_known_matmul(a, b);

        // 非方阵： [2,3] @ [3,2] -> [2,2]
        mcn::Tensor m23({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        mcn::Tensor m32({{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}});
        mcn::Tensor c = m23.matmul(m32);
        CHECK_SHAPE(c, 2, 2);
        CHECK_FLOAT_EQ(f_at(c, {0, 0}), 1.f * 7 + 2.f * 9 + 3.f * 11);  // 58
        CHECK_FLOAT_EQ(f_at(c, {0, 1}), 1.f * 8 + 2.f * 10 + 3.f * 12); // 64
        CHECK_FLOAT_EQ(f_at(c, {1, 0}), 4.f * 7 + 5.f * 9 + 6.f * 11);  // 139
        CHECK_FLOAT_EQ(f_at(c, {1, 1}), 4.f * 8 + 5.f * 10 + 6.f * 12); // 154

        // 1D @ 2D -> 1D
        mcn::Tensor v({1.f, 2.f});
        mcn::Tensor vc = v.matmul(b);
        CHECK_SHAPE(vc, 2);
        CHECK_FLOAT_EQ(f_at(vc, {0}), 1.f * 5 + 2.f * 7);  // 19
        CHECK_FLOAT_EQ(f_at(vc, {1}), 1.f * 6 + 2.f * 8);  // 22

        // 2D @ 1D -> 1D
        mcn::Tensor cv = a.matmul(v);
        CHECK_SHAPE(cv, 2);
        CHECK_FLOAT_EQ(f_at(cv, {0}), 1.f * 1 + 2.f * 2); // 5
        CHECK_FLOAT_EQ(f_at(cv, {1}), 3.f * 1 + 4.f * 2); // 11

        // 1D @ 1D -> 0 维标量（点积）
        mcn::Tensor dot = v.matmul(v);
        CHECK_TRUE(dot.shape().empty());
        CHECK_NUMEL(dot, 1);
        CHECK_FLOAT_EQ(*static_cast<const float *>(dot.data_ptr()), 5.f);

        // 单位矩阵
        mcn::Tensor eye = mcn::Tensor::zeros({3, 3});
        static_cast<float *>(eye.data_ptr())[0] = 1.f;
        static_cast<float *>(eye.data_ptr())[4] = 1.f;
        static_cast<float *>(eye.data_ptr())[8] = 1.f;
        mcn::Tensor m33({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}, {7.f, 8.f, 9.f}});
        mcn::Tensor ident = m33.matmul(eye);
        CHECK_FLOAT_EQ(f_at(ident, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(ident, {1, 2}), 6.f);
        CHECK_FLOAT_EQ(f_at(ident, {2, 1}), 8.f);

        // 尺寸大于一个 tile（16），验证分块边界的 padding 逻辑
        // （CPU 路径没有 tile，这里主要保证不越界；CUDA 路径会实际走分块）
        mcn::Tensor big_a = mcn::Tensor::ones({17, 20});
        mcn::Tensor big_b = mcn::Tensor::ones({20, 19});
        mcn::Tensor big_c = big_a.matmul(big_b);
        CHECK_SHAPE(big_c, 17, 19);
        CHECK_FLOAT_EQ(f_at(big_c, {0, 0}), 20.f);
        CHECK_FLOAT_EQ(f_at(big_c, {16, 18}), 20.f);

        // 原张量不被修改
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(b, {0, 0}), 5.f);
    }

    // ---------------- 转置 / 连续化 ----------------

    void test_transpose()
    {
        std::puts("  test_transpose");

        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}); // [2,3]

        // 2D 转置
        mcn::Tensor tr = m.t();
        CHECK_SHAPE(tr, 3, 2);
        CHECK_FLOAT_EQ(f_at(tr, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(tr, {0, 1}), 4.f);
        CHECK_FLOAT_EQ(f_at(tr, {1, 0}), 2.f);
        CHECK_FLOAT_EQ(f_at(tr, {2, 1}), 6.f);

        // 转置是视图：共享存储，改一个另一个也变
        static_cast<float *>(tr.data_ptr())[0] = 99.f;
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 99.f);
        static_cast<float *>(m.data_ptr())[0] = 1.f;

        // 转置两次回到原状
        mcn::Tensor back = m.t().t();
        CHECK_SHAPE(back, 2, 3);
        CHECK_FLOAT_EQ(f_at(back, {0, 2}), 3.f);

        // 转置后非连续，contiguous() 应物化出副本
        mcn::Tensor tc = tr.contiguous();
        CHECK_SHAPE(tc, 3, 2);
        CHECK_FLOAT_EQ(f_at(tc, {0, 1}), 4.f);
        CHECK_FLOAT_EQ(f_at(tc, {2, 0}), 3.f);
        static_cast<float *>(tc.data_ptr())[0] = 7.f;
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 1.f); // 副本独立

        // 已连续的张量 contiguous() 不拷贝
        mcn::Tensor cc = m.contiguous();
        static_cast<float *>(cc.data_ptr())[0] = 55.f;
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 55.f);
        static_cast<float *>(m.data_ptr())[0] = 1.f;

        // 3D 任意两维交换
        mcn::Tensor t3 = mcn::Tensor::zeros({2, 3, 4});
        mcn::Tensor p = t3.transpose(0, 2);
        CHECK_SHAPE(p, 4, 3, 2);
        CHECK_TRUE(p.transpose(0, 2).shape() == t3.shape());

        // 负索引
        CHECK_TRUE(t3.transpose(-1, -3).shape() == p.shape());

        // 转置后可以直接参与 matmul（内部自动连续化）
        mcn::Tensor x({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor y({{5.f, 6.f}, {7.f, 8.f}});
        mcn::Tensor prod = x.matmul(y.t()); // [1 2;3 4] @ [5 7;6 8]
        CHECK_FLOAT_EQ(f_at(prod, {0, 0}), 17.f); // 1*5+2*6
        CHECK_FLOAT_EQ(f_at(prod, {1, 1}), 53.f); // 3*7+4*8
    }

    void test_cuda_matmul()
    {
        std::puts("  test_cuda_matmul");

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            std::puts("    (skip: no CUDA device)");
            return;
        }

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b({{5.f, 6.f}, {7.f, 8.f}});
        a.to_(mcn::Device::CUDA);
        b.to_(mcn::Device::CUDA);

        mcn::Tensor c = a.matmul(b);
        CHECK_DEVICE(c, mcn::Device::CUDA);
        c.to_(mcn::Device::CPU);
        CHECK_SHAPE(c, 2, 2);
        CHECK_FLOAT_EQ(f_at(c, {0, 0}), 19.f);
        CHECK_FLOAT_EQ(f_at(c, {0, 1}), 22.f);
        CHECK_FLOAT_EQ(f_at(c, {1, 0}), 43.f);
        CHECK_FLOAT_EQ(f_at(c, {1, 1}), 50.f);

        // 非 tile 整数倍尺寸，验证 kernel 里的边界 padding
        mcn::Tensor big_a = mcn::Tensor::ones({17, 20});
        mcn::Tensor big_b = mcn::Tensor::ones({20, 19});
        big_a.to_(mcn::Device::CUDA);
        big_b.to_(mcn::Device::CUDA);
        mcn::Tensor big_c = big_a.matmul(big_b);
        big_c.to_(mcn::Device::CPU);
        CHECK_SHAPE(big_c, 17, 19);
        CHECK_FLOAT_EQ(f_at(big_c, {0, 0}), 20.f);
        CHECK_FLOAT_EQ(f_at(big_c, {16, 18}), 20.f);
        CHECK_FLOAT_EQ(f_at(big_c, {8, 9}), 20.f);
    }

    // ---------------- 设备迁移----------------

    void test_to_copy()
    {
        std::puts("  test_to_copy");

        // to() 返回副本，原张量不变
        mcn::Tensor t({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor c = t.to(mcn::Device::CPU);
        CHECK_DEVICE(t, mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(c, {0, 0}), 1.f);

        static_cast<float *>(c.data_ptr())[0] = 9.f;
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 1.f); // 副本独立
    }

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
        t.to_(mcn::Device::CUDA);
        CHECK_DEVICE(t, mcn::Device::CUDA);
        CHECK_SHAPE(t, 2, 3);

        t.to_(mcn::Device::CPU);
        CHECK_DEVICE(t, mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(t, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(t, {1, 2}), 6.f);

        // 带 offset_ 的视图往返，验证偏移没被丢掉
        mcn::Tensor row = t[1];
        row.to_(mcn::Device::CUDA);
        row.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(row, {0}), 4.f);
        CHECK_FLOAT_EQ(f_at(row, {2}), 6.f);
    }

    void test_cuda_ops()
    {
        std::puts("  test_cuda_ops");

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            std::puts("    (skip: no CUDA device)");
            return;
        }

        // fill_ / zero_ 的 device kernel
        mcn::Tensor z = mcn::Tensor::zeros({4}, mcn::Device::CUDA);
        z.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(z, {0}), 0.f);
        CHECK_FLOAT_EQ(f_at(z, {3}), 0.f);

        mcn::Tensor o = mcn::Tensor::ones({3}, mcn::Device::CUDA);
        o.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(o, {2}), 1.f);

        // add_ 的 device kernel，与 CPU 结果对比
        mcn::Tensor a = mcn::Tensor({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b = mcn::Tensor({{10.f, 20.f}, {30.f, 40.f}});
        a.to_(mcn::Device::CUDA);
        b.to_(mcn::Device::CUDA);
        a.add_(b);
        CHECK_DEVICE(a, mcn::Device::CUDA);
        a.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(a, {1, 1}), 44.f);

        // randn 直接生成到设备上
        mcn::Tensor r = mcn::Tensor::randn({8}, mcn::Device::CUDA);
        CHECK_DEVICE(r, mcn::Device::CUDA);
        r.to_(mcn::Device::CPU);
        CHECK_NUMEL(r, 8);
    }

    void test_cuda_binary()
    {
        std::puts("  test_cuda_binary");

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            std::puts("    (skip: no CUDA device)");
            return;
        }

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor b({{10.f, 20.f}, {30.f, 40.f}});
        a.to_(mcn::Device::CUDA);
        b.to_(mcn::Device::CUDA);

        mcn::Tensor sum = a + b;
        CHECK_DEVICE(sum, mcn::Device::CUDA);
        CHECK_SHAPE(sum, 2, 2);
        sum.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(sum, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(sum, {1, 1}), 44.f);

        mcn::Tensor diff = a - b;
        diff.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(diff, {0, 1}), -18.f);

        mcn::Tensor prod = a * b;
        prod.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(prod, {0, 1}), 40.f);
        CHECK_FLOAT_EQ(f_at(prod, {1, 1}), 160.f);

        // 设备上的原张量不应被修改
        mcn::Tensor check = a.to(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(check, {0, 0}), 1.f);
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
    test_zeros_ones();
    test_fill();
    test_copy();
    test_add();
    test_randn();
    test_binary_ops();
    test_matmul();
    test_transpose();
    test_to_copy();
    test_to_device();
    test_cuda_ops();
    test_cuda_binary();
    test_cuda_matmul();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
