// Tensor 单元测试：构造、元数据、视图、reshape、clone、设备迁移
#include "test_common.h"

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

    void test_copy_semantics()
    {
        std::puts("  test_copy_semantics");

        mcn::Tensor a({1.f, 2.f, 3.f});

        // 拷贝构造 = 浅拷贝：只是同一块存储的另一个句柄
        mcn::Tensor b = a;
        CHECK_SHAPE(b, 3);
        CHECK_FLOAT_EQ(f_at(b, {1}), 2.f);
        static_cast<float *>(b.data_ptr())[1] = 20.f;
        CHECK_FLOAT_EQ(f_at(a, {1}), 20.f);

        // 拷贝赋值同样是浅拷贝（与拷贝构造一致）
        mcn::Tensor c;
        c = a;
        static_cast<float *>(c.data_ptr())[2] = 30.f;
        CHECK_FLOAT_EQ(f_at(a, {2}), 30.f);

        // clone 仍然是深拷贝
        mcn::Tensor d = a.clone();
        static_cast<float *>(d.data_ptr())[0] = 99.f;
        CHECK_FLOAT_EQ(f_at(a, {0}), 1.f);
        CHECK_FLOAT_EQ(f_at(d, {0}), 99.f);

        // 浅拷贝保留 offset_：拷贝一个视图，仍指向原存储的同一位置
        mcn::Tensor m({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor row = m[1];
        CHECK_FLOAT_EQ(f_at(row, {0}), 3.f);
        static_cast<float *>(row.data_ptr())[0] = 33.f;
        CHECK_FLOAT_EQ(f_at(m, {1, 0}), 33.f);

        // clone 视图则物化成独立连续副本
        mcn::Tensor m2({{1.f, 2.f}, {3.f, 4.f}});
        mcn::Tensor rc = m2[1].clone();
        static_cast<float *>(rc.data_ptr())[0] = 77.f;
        CHECK_FLOAT_EQ(f_at(m2, {1, 0}), 3.f);

        // 拷贝共享自动微分元数据：指向同一个节点
        mcn::Tensor x({2.f, 3.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = x * x;
        mcn::Tensor y2 = y;
        CHECK_TRUE(y2.grad_fn() == y.grad_fn());
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

    // ---------------- 一元取负 ----------------

    void test_neg()
    {
        std::puts("  test_neg");

        mcn::Tensor a({{1.f, -2.f}, {3.5f, 0.f}});

        mcn::Tensor n = a.neg();
        CHECK_SHAPE(n, 2, 2);
        CHECK_FLOAT_EQ(f_at(n, {0, 0}), -1.f);
        CHECK_FLOAT_EQ(f_at(n, {0, 1}), 2.f);
        CHECK_FLOAT_EQ(f_at(n, {1, 0}), -3.5f);
        CHECK_FLOAT_EQ(f_at(n, {1, 1}), 0.f);

        // operator-() 与 neg() 等价
        mcn::Tensor m = -a;
        CHECK_FLOAT_EQ(f_at(m, {0, 1}), 2.f);

        // 原张量不被修改
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 1.f);

        // 双重取负回到原值
        CHECK_FLOAT_EQ(f_at(-(-a), {1, 0}), 3.5f);

        // 0 维标量
        mcn::Tensor s(7.f);
        CHECK_TRUE(s.neg().shape().empty());
        CHECK_FLOAT_EQ(*static_cast<const float *>(s.neg().data_ptr()), -7.f);

        // int dtype
        mcn::Tensor i({1, -5});
        mcn::Tensor ni = -i;
        CHECK_DTYPE(ni, mcn::DataType::INT32);
        CHECK_INT(i_at(ni, {0}), -1);
        CHECK_INT(i_at(ni, {1}), 5);

        // 映射到 0 元素
        mcn::Tensor e = mcn::Tensor::empty({0, 3});
        CHECK_NUMEL(e.neg(), 0);

        // 视图上的取负：结果应是独立的新张量
        mcn::Tensor v = a[1];
        mcn::Tensor nv = -v;
        CHECK_SHAPE(nv, 2);
        CHECK_FLOAT_EQ(f_at(nv, {0}), -3.5f);
    }

    // ---------------- 标量乘 / zero_grad ----------------

    void test_scalar_mul()
    {
        std::puts("  test_scalar_mul");

        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}});

        mcn::Tensor s = a * 2.f;
        CHECK_SHAPE(s, 2, 2);
        CHECK_FLOAT_EQ(f_at(s, {0, 0}), 2.f);
        CHECK_FLOAT_EQ(f_at(s, {1, 1}), 8.f);

        // int 标量也应接受（走 static_cast）
        CHECK_FLOAT_EQ(f_at(a * 3, {0, 1}), 6.f);
        CHECK_FLOAT_EQ(f_at(a * 0.5, {1, 0}), 1.5f);

        // 负数、零
        CHECK_FLOAT_EQ(f_at(a * -1, {0, 0}), -1.f);
        CHECK_FLOAT_EQ(f_at(a * 0, {1, 1}), 0.f);

        // 原张量不被修改
        CHECK_FLOAT_EQ(f_at(a, {0, 0}), 1.f);

        // 与 neg() 一致
        CHECK_FLOAT_EQ(f_at(a * -1, {1, 1}), f_at(-a, {1, 1}));

        // int dtype
        mcn::Tensor i({2, 5});
        mcn::Tensor si = i * 3;
        CHECK_DTYPE(si, mcn::DataType::INT32);
        CHECK_INT(i_at(si, {0}), 6);
        CHECK_INT(i_at(si, {1}), 15);

        // 0 维标量
        mcn::Tensor z(2.f);
        CHECK_TRUE((z * 5).shape().empty());
        CHECK_FLOAT_EQ(*static_cast<const float *>((z * 5).data_ptr()), 10.f);

        // 0 元素
        CHECK_NUMEL(mcn::Tensor::empty({0, 3}) * 2.f, 0);
    }

    void test_zero_grad()
    {
        std::puts("  test_zero_grad");

        mcn::Tensor x({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::Tensor y1 = x * x;
        y1.backward();
        CHECK_TRUE(x.grad() != nullptr);
        CHECK_FLOAT_EQ(grad_at(x, {0}), 2.f);

        // 不清空就再 backward，梯度会累加
        mcn::Tensor y2 = x * x;
        y2.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), 4.f); // 2 + 2

        // 清空后回到单次的值
        x.zero_grad();
        CHECK_TRUE(x.grad() == nullptr);

        // 返回自身引用，可链式
        mcn::Tensor z({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        (z * z).backward();
        mcn::Tensor &ref = z.zero_grad();
        CHECK_TRUE(&ref == &z);
        CHECK_TRUE(z.grad() == nullptr);
    }

    void test_autograd_scalar_mul()
    {
        std::puts("  test_autograd_scalar_mul");

        // y = x * 3  =>  dy/dx = 3
        mcn::Tensor x({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = x * 3.f;
        CHECK_TRUE(y.grad_fn() != nullptr);
        y.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), 3.f);
        CHECK_FLOAT_EQ(grad_at(x, {1}), 3.f);

        // 链式：y = (x + x) * 2  =>  dy/dx = 4
        mcn::Tensor a({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor s = (a + a) * 2.f;
        s.backward();
        CHECK_FLOAT_EQ(grad_at(a, {0}), 4.f);

        // 复合：y = (x * 2) * (x * 3) = 6x^2  =>  dy/dx = 12x = 12
        mcn::Tensor b({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor m = (b * 2.f) * (b * 3.f);
        m.backward();
        CHECK_FLOAT_EQ(grad_at(b, {0}), 12.f);

        // 模拟一步 SGD：w = w - lr * grad
        mcn::Tensor w({2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor loss = w * w; // 4
        loss.backward();          // dw = 2w = 4
        mcn::Tensor step = (*w.grad()) * 0.1f;
        w = w - step;
        CHECK_FLOAT_EQ(f_at(w, {0}), 1.6f); // 2 - 0.4
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

        // 一元取负的设备 kernel
        mcn::Tensor n = -a;
        CHECK_DEVICE(n, mcn::Device::CUDA);
        n.to_(mcn::Device::CPU);
        CHECK_FLOAT_EQ(f_at(n, {0, 0}), -1.f);
        CHECK_FLOAT_EQ(f_at(n, {1, 1}), -4.f);

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

    // ---------------- 广播 ----------------

    void test_broadcast()
    {
        std::puts("  test_broadcast");

        // broadcast_to 是元数据操作：广播维 stride = 0，数据共享
        mcn::Tensor v({1.f, 2.f, 3.f});
        mcn::Tensor b = v.broadcast_to({2, 3});
        CHECK_SHAPE(b, 2, 3);
        CHECK_NUMEL(b, 6);
        CHECK_FLOAT_EQ(f_at(b, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(b, {0, 2}), 3.f);
        CHECK_FLOAT_EQ(f_at(b, {1, 0}), 1.f); // 第二行重复第一行
        CHECK_FLOAT_EQ(f_at(b, {1, 2}), 3.f);

        // 共享存储：改源张量，广播视图跟着变
        static_cast<float *>(v.data_ptr())[1] = 99.f;
        CHECK_FLOAT_EQ(f_at(b, {1, 1}), 99.f);
        static_cast<float *>(v.data_ptr())[1] = 2.f;

        // 同一形状广播到自身
        CHECK_SHAPE(v.broadcast_to({1, 1, 3}), 1, 1, 3);

        // 不可广播时应中止（这里只验证合法路径）
        // 标量广播到任意形状
        mcn::Tensor s(5.f);
        mcn::Tensor sb = s.broadcast_to({2, 2});
        CHECK_FLOAT_EQ(f_at(sb, {1, 1}), 5.f);

        // ---- 逐元素运算自动广播 ----
        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}); // [2,3]
        mcn::Tensor row({10.f, 20.f, 30.f});               // [3]

        mcn::Tensor sum = m + row;
        CHECK_SHAPE(sum, 2, 3);
        CHECK_FLOAT_EQ(f_at(sum, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(sum, {1, 2}), 36.f);

        mcn::Tensor sub = m - row;
        CHECK_FLOAT_EQ(f_at(sub, {1, 0}), -6.f); // 4 - 10

        mcn::Tensor mul = m * row;
        CHECK_FLOAT_EQ(f_at(mul, {0, 2}), 90.f);  // 3 * 30
        CHECK_FLOAT_EQ(f_at(mul, {1, 1}), 100.f); // 5 * 20

        // [2,1] + [3] -> [2,3]
        mcn::Tensor col({{1.f}, {2.f}});
        mcn::Tensor outer = col + row;
        CHECK_SHAPE(outer, 2, 3);
        CHECK_FLOAT_EQ(f_at(outer, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(f_at(outer, {1, 0}), 12.f);
        CHECK_FLOAT_EQ(f_at(outer, {1, 2}), 32.f);

        // 标量张量广播
        mcn::Tensor mm = m + s;
        CHECK_SHAPE(mm, 2, 3);
        CHECK_FLOAT_EQ(f_at(mm, {0, 0}), 6.f);

        // 原张量不变
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(f_at(row, {0}), 10.f);
    }

    void test_sum_to()
    {
        std::puts("  test_sum_to");

        // [2,3] 归约到 [3]
        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        mcn::Tensor s = m.sum_to({3});
        CHECK_SHAPE(s, 3);
        CHECK_FLOAT_EQ(f_at(s, {0}), 5.f);  // 1+4
        CHECK_FLOAT_EQ(f_at(s, {1}), 7.f);  // 2+5
        CHECK_FLOAT_EQ(f_at(s, {2}), 9.f);  // 3+6

        // 归约到标量
        mcn::Tensor all = m.sum_to({});
        CHECK_TRUE(all.shape().empty());
        CHECK_FLOAT_EQ(*static_cast<const float *>(all.data_ptr()), 21.f);

        // 归约到 [2,1]
        mcn::Tensor col = m.sum_to({2, 1});
        CHECK_SHAPE(col, 2, 1);
        CHECK_FLOAT_EQ(f_at(col, {0, 0}), 6.f);  // 1+2+3
        CHECK_FLOAT_EQ(f_at(col, {1, 0}), 15.f); // 4+5+6

        // 同形状直接返回（不拷贝）
        mcn::Tensor same = m.sum_to({2, 3});
        CHECK_SHAPE(same, 2, 3);

        // 部分维保持： [2,3] -> [2,3] 以外的 [1,3]
        mcn::Tensor partial = m.sum_to({1, 3});
        CHECK_FLOAT_EQ(f_at(partial, {0, 0}), 5.f);

        // 3 维： [2,2,2] -> [1,2,2]
        mcn::Tensor t3 = mcn::Tensor::ones({2, 2, 2});
        mcn::Tensor r3 = t3.sum_to({1, 2, 2});
        CHECK_SHAPE(r3, 1, 2, 2);
        CHECK_FLOAT_EQ(f_at(r3, {0, 0, 0}), 2.f);
    }

    void test_autograd_broadcast()
    {
        std::puts("  test_autograd_broadcast");

        // y = m + row，m=[2,3], row=[3]
        //   dm = ones[2,3]
        //   drow = sum over dim0 => [2,2,2]
        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor row({10.f, 20.f, 30.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::Tensor y = m + row;
        y.backward();

        CHECK_FLOAT_EQ(grad_at(m, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(grad_at(m, {1, 2}), 1.f);
        CHECK_FLOAT_EQ(grad_at(row, {0}), 2.f); // 两行相加
        CHECK_FLOAT_EQ(grad_at(row, {1}), 2.f);
        CHECK_FLOAT_EQ(grad_at(row, {2}), 2.f);

        // y = m * row =>  dm = row 广播, drow = sum(m 的两行)
        mcn::Tensor a({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({10.f, 20.f, 30.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor z = a * b;
        z.backward();

        CHECK_FLOAT_EQ(grad_at(a, {0, 1}), 20.f); // = b[1]
        CHECK_FLOAT_EQ(grad_at(a, {1, 1}), 20.f);
        CHECK_FLOAT_EQ(grad_at(b, {0}), 5.f);  // 1 + 4
        CHECK_FLOAT_EQ(grad_at(b, {2}), 9.f);  // 3 + 6

        // 标量广播：y = m * scalar_tensor => dm = ones，dscalar = sum(m)
        mcn::Tensor p({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor q(2.f, mcn::Device::CPU, true);
        mcn::Tensor w = p * q;
        w.backward();

        CHECK_FLOAT_EQ(grad_at(p, {0, 0}), 2.f); // = q
        CHECK_FLOAT_EQ(grad_at(q, {}), 10.f);    // 1+2+3+4 = 10（q 是 0 维）

        // 列 + 行 -> [2,3]，双向归约
        mcn::Tensor col({{1.f}, {2.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor rowv({10.f, 20.f, 30.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor outer = col + rowv;
        outer.backward();

        CHECK_FLOAT_EQ(grad_at(col, {0, 0}), 3.f); // 3 列 => 3
        CHECK_FLOAT_EQ(grad_at(col, {1, 0}), 3.f);
        CHECK_FLOAT_EQ(grad_at(rowv, {0}), 2.f); // 2 行 => 2
        CHECK_FLOAT_EQ(grad_at(rowv, {2}), 2.f);
    }

    // ---------------- 归约 ----------------

    void test_sum_mean()
    {
        std::puts("  test_sum_mean");

        mcn::Tensor m({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}); // [2,3]

        // 全归约
        mcn::Tensor all = m.sum();
        CHECK_TRUE(all.shape().empty());
        CHECK_FLOAT_EQ(*static_cast<const float *>(all.data_ptr()), 21.f);

        // 沿 dim 0（跨行相加）-> [3]
        mcn::Tensor s0 = m.sum(0);
        CHECK_SHAPE(s0, 3);
        CHECK_FLOAT_EQ(f_at(s0, {0}), 5.f); // 1+4
        CHECK_FLOAT_EQ(f_at(s0, {2}), 9.f); // 3+6

        // 沿 dim 1（跨列相加）-> [2]
        mcn::Tensor s1 = m.sum(1);
        CHECK_SHAPE(s1, 2);
        CHECK_FLOAT_EQ(f_at(s1, {0}), 6.f);  // 1+2+3
        CHECK_FLOAT_EQ(f_at(s1, {1}), 15.f); // 4+5+6

        // keepdim
        mcn::Tensor k0 = m.sum(0, true);
        CHECK_SHAPE(k0, 1, 3);
        CHECK_FLOAT_EQ(f_at(k0, {0, 1}), 7.f);
        mcn::Tensor k1 = m.sum(1, true);
        CHECK_SHAPE(k1, 2, 1);
        CHECK_FLOAT_EQ(f_at(k1, {1, 0}), 15.f);

        // 负维索引
        CHECK_SHAPE(m.sum(-1), 2);
        CHECK_SHAPE(m.sum(-2), 3);

        // mean
        mcn::Tensor ma = m.mean();
        CHECK_TRUE(ma.shape().empty());
        CHECK_FLOAT_EQ(*static_cast<const float *>(ma.data_ptr()), 3.5f);

        mcn::Tensor m0 = m.mean(0);
        CHECK_SHAPE(m0, 3);
        CHECK_FLOAT_EQ(f_at(m0, {0}), 2.5f); // (1+4)/2
        CHECK_FLOAT_EQ(f_at(m0, {1}), 3.5f);

        mcn::Tensor m1 = m.mean(1);
        CHECK_SHAPE(m1, 2);
        CHECK_FLOAT_EQ(f_at(m1, {0}), 2.f);  // 6/3
        CHECK_FLOAT_EQ(f_at(m1, {1}), 5.f);  // 15/3

        // 3 维
        mcn::Tensor t3 = mcn::Tensor::ones({2, 3, 4});
        CHECK_SHAPE(t3.sum(1), 2, 4);
        CHECK_FLOAT_EQ(f_at(t3.sum(1), {0, 0}), 3.f);
        CHECK_SHAPE(t3.sum(1, true), 2, 1, 4);
        CHECK_FLOAT_EQ(f_at(t3.mean(2), {1, 2}), 1.f);

        // 原张量不变
        CHECK_FLOAT_EQ(f_at(m, {0, 0}), 1.f);
    }

    void test_autograd_sum_mean()
    {
        std::puts("  test_autograd_sum_mean");

        // y = sum(x)  =>  dx = 全 1
        mcn::Tensor x({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = x.sum();
        CHECK_TRUE(y.grad_fn() != nullptr);
        y.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(grad_at(x, {1, 1}), 1.f);

        // y = mean(x)  =>  dx = 1/n = 1/4
        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        a.mean().backward();
        CHECK_FLOAT_EQ(grad_at(a, {0, 0}), 0.25f);
        CHECK_FLOAT_EQ(grad_at(a, {1, 1}), 0.25f);

        // y = sum(x, dim=0) -> [3]  =>  dx 每行都是 1
        mcn::Tensor b({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        b.sum(0).backward();
        CHECK_FLOAT_EQ(grad_at(b, {0, 0}), 1.f);
        CHECK_FLOAT_EQ(grad_at(b, {1, 0}), 1.f);

        // y = sum(x, dim=1) -> [2]  =>  dx 每个元素都是 1
        mcn::Tensor c({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        c.sum(1).backward();
        CHECK_FLOAT_EQ(grad_at(c, {0, 2}), 1.f);
        CHECK_FLOAT_EQ(grad_at(c, {1, 0}), 1.f);

        // y = mean(x, dim=0)，n=2  =>  dx = 1/2
        mcn::Tensor d({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        d.mean(0).backward();
        CHECK_FLOAT_EQ(grad_at(d, {0, 0}), 0.5f);
        CHECK_FLOAT_EQ(grad_at(d, {1, 1}), 0.5f);

        // 组合：y = mean(x * x) = mean(x^2)  =>  dy/dx = 2x/n
        // x = [1,2,3,4]，n = 4  =>  [0.5, 1, 1.5, 2]
        mcn::Tensor e({1.f, 2.f, 3.f, 4.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        (e * e).mean().backward();
        CHECK_FLOAT_EQ(grad_at(e, {0}), 0.5f);
        CHECK_FLOAT_EQ(grad_at(e, {1}), 1.f);
        CHECK_FLOAT_EQ(grad_at(e, {2}), 1.5f);
        CHECK_FLOAT_EQ(grad_at(e, {3}), 2.f);
    }

    // ---------------- 自动梯度 ----------------

    void test_autograd_basic()
    {
        std::puts("  test_autograd_basic");

        // y = x * x  =>  dy/dx = 2x
        mcn::Tensor x({2.f, 3.f, 4.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = x * x;
        CHECK_TRUE(y.require_grad());
        CHECK_TRUE(y.grad_fn() != nullptr);

        y.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), 4.f);
        CHECK_FLOAT_EQ(grad_at(x, {1}), 6.f);
        CHECK_FLOAT_EQ(grad_at(x, {2}), 8.f);

        // y = a - b  =>  da = 1, db = -1
        mcn::Tensor a({1.f, 1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({5.f, 7.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor d = a - b;
        d.backward();
        CHECK_FLOAT_EQ(grad_at(a, {0}), 1.f);
        CHECK_FLOAT_EQ(grad_at(a, {1}), 1.f);
        CHECK_FLOAT_EQ(grad_at(b, {0}), -1.f);
        CHECK_FLOAT_EQ(grad_at(b, {1}), -1.f);

        // 只有一边 require_grad：另一边不应有梯度
        mcn::Tensor p({2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor q({3.f});
        mcn::Tensor r = p * q;
        r.backward();
        CHECK_FLOAT_EQ(grad_at(p, {0}), 3.f);
        CHECK_TRUE(q.grad() == nullptr);
    }

    void test_autograd_chain()
    {
        std::puts("  test_autograd_chain");

        // z = (a + b) * c
        // dz/da = c, dz/db = c, dz/dc = a + b
        mcn::Tensor a({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({3.f, 4.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor c({10.f, 20.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::Tensor z = (a + b) * c;
        z.backward();

        CHECK_FLOAT_EQ(grad_at(a, {0}), 10.f);
        CHECK_FLOAT_EQ(grad_at(a, {1}), 20.f);
        CHECK_FLOAT_EQ(grad_at(b, {0}), 10.f);
        CHECK_FLOAT_EQ(grad_at(b, {1}), 20.f);
        CHECK_FLOAT_EQ(grad_at(c, {0}), 4.f);  // a+b = [4, 6]
        CHECK_FLOAT_EQ(grad_at(c, {1}), 6.f);

        // 菱形结构：y = (x + x) * (x * x)
        // y = 2x * x^2 = 2x^3  =>  dy/dx = 6x^2
        mcn::Tensor x({2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = (x + x) * (x * x);
        y.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), 24.f); // 6 * 4

        // 同一张量被多次使用，梯度应累加而不是覆盖
        mcn::Tensor m({2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor s = m + m + m; // ds/dm = 3
        s.backward();
        CHECK_FLOAT_EQ(grad_at(m, {0}), 3.f);
    }

    void test_autograd_neg()
    {
        std::puts("  test_autograd_neg");

        // y = -x  =>  dy/dx = -1
        mcn::Tensor x({1.f, 2.f, 3.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = -x;
        CHECK_TRUE(y.require_grad());
        CHECK_TRUE(y.grad_fn() != nullptr);
        y.backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), -1.f);
        CHECK_FLOAT_EQ(grad_at(x, {1}), -1.f);
        CHECK_FLOAT_EQ(grad_at(x, {2}), -1.f);

        // 链式：z = -(a - b)  =>  dz/da = -1, dz/db = 1
        // （SubBackward 现在用 -g，这里正好反向验证两条路径互通）
        mcn::Tensor a({2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({5.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor z = -(a - b);
        z.backward();
        CHECK_FLOAT_EQ(grad_at(a, {0}), -1.f);
        CHECK_FLOAT_EQ(grad_at(b, {0}), 1.f);

        // 双重取负：dz/dx = 1
        mcn::Tensor p({4.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor q = -(-p);
        q.backward();
        CHECK_FLOAT_EQ(grad_at(p, {0}), 1.f);
    }

    void test_autograd_matmul()
    {
        std::puts("  test_autograd_matmul");
        // y = a @ b，a=[1 2; 3 4], b=[5 6; 7 8]
        //   y = [19 22; 43 50]
        //   da = g @ b^T,  db = a^T @ g，g 全 1
        mcn::Tensor a({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({{5.f, 6.f}, {7.f, 8.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::Tensor y = a.matmul(b);
        CHECK_TRUE(y.grad_fn() != nullptr);
        y.backward();

        // da = g @ b^T：b^T = [5 7; 6 8]，ones @ b^T 每行都是 b^T 的列和
        //   row = [1,1] @ [5 7; 6 8] = [5+6, 7+8] = [11, 15]
        CHECK_FLOAT_EQ(grad_at(a, {0, 0}), 11.f);
        CHECK_FLOAT_EQ(grad_at(a, {0, 1}), 15.f);
        CHECK_FLOAT_EQ(grad_at(a, {1, 0}), 11.f);
        CHECK_FLOAT_EQ(grad_at(a, {1, 1}), 15.f);

        // db = a^T @ ones：a^T = [1 3; 2 4]，每行之和是 4 和 6
        CHECK_FLOAT_EQ(grad_at(b, {0, 0}), 4.f); // 1+3
        CHECK_FLOAT_EQ(grad_at(b, {0, 1}), 4.f);
        CHECK_FLOAT_EQ(grad_at(b, {1, 0}), 6.f); // 2+4
        CHECK_FLOAT_EQ(grad_at(b, {1, 1}), 6.f);

        // 链式：z = (a @ b) 再逐元素乘 c
        mcn::Tensor p({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor q({{1.f, 0.f}, {0.f, 1.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor c({{2.f, 2.f}, {2.f, 2.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::Tensor mm = p.matmul(q); // = p
        mcn::Tensor z = mm * c;
        z.backward();

        // dz/dp = (g*c) @ q^T = 2*ones @ I = [[2,2],[2,2]]
        CHECK_FLOAT_EQ(grad_at(p, {0, 0}), 2.f);
        CHECK_FLOAT_EQ(grad_at(p, {1, 1}), 2.f);
        // dz/dq = p^T @ (g*c) = [[1,3],[2,4]] @ [[2,2],[2,2]] = [[8,8],[12,12]]
        CHECK_FLOAT_EQ(grad_at(q, {0, 0}), 8.f);
        CHECK_FLOAT_EQ(grad_at(q, {1, 1}), 12.f);
        // dz/dc = mm
        CHECK_FLOAT_EQ(grad_at(c, {0, 1}), 2.f);
        CHECK_FLOAT_EQ(grad_at(c, {1, 0}), 3.f);
    }

    void test_cuda_ops()
    {
        std::puts("  test_cuda_ops");        int device_count = 0;
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
    test_copy_semantics();
    test_zeros_ones();
    test_fill();
    test_copy();
    test_add();
    test_randn();
    test_binary_ops();
    test_neg();
    test_scalar_mul();
    test_zero_grad();
    test_matmul();
    test_transpose();
    test_broadcast();
    test_sum_to();
    test_sum_mean();
    test_autograd_basic();
    test_autograd_chain();
    test_autograd_broadcast();
    test_autograd_sum_mean();
    test_autograd_neg();
    test_autograd_scalar_mul();
    test_autograd_matmul();
    test_to_copy();
    test_to_device();
    test_cuda_ops();
    test_cuda_binary();
    test_cuda_matmul();

    std::printf("\n%d checks, %d failure(s)\n", mcn_test::g_checks, mcn_test::g_failures);
    return mcn_test::g_failures == 0 ? 0 : 1;
}
