// 优化器单元测试：基类契约 + SGD
#include "test_common.h"

#include "mini_cuda_nn/optim/sgd.h"

#include <cstdio>

namespace
{
    // 给叶子张量塞一个固定梯度（绕过计算图，专测优化器本身）
    void set_grad(mcn::Tensor &t, const mcn::Tensor &g)
    {
        t.zero_grad();
        t.accumulate_grad(g);
    }

    void test_sgd_basic()
    {
        std::puts("  test_sgd_basic");

        // p = [1, 2]，g = [3, 4]，lr = 0.1  =>  p = [0.7, 1.6]
        mcn::Tensor w({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(w, mcn::Tensor({3.f, 4.f}));

        mcn::optim::SGD sgd({&w}, 0.1);
        CHECK_FLOAT_EQ(sgd.lr(), 0.1);
        CHECK_TRUE(sgd.step_count() == 0);
        CHECK_TRUE(std::string(sgd.name()) == "sgd");

        sgd.step();
        CHECK_TRUE(sgd.step_count() == 1);
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.7f);
        CHECK_FLOAT_EQ(f_at(w, {1}), 1.6f);

        // 梯度还在，再走一步 => 再减 0.3 / 0.4
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.4f);
        CHECK_FLOAT_EQ(f_at(w, {1}), 1.2f);
        CHECK_TRUE(sgd.step_count() == 2);
    }

    void test_sgd_updates_original()
    {
        std::puts("  test_sgd_updates_original");

        // 优化器存的必须是指向调用方那个张量的指针，不能是副本
        mcn::Tensor w({5.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(w, mcn::Tensor({1.f}));

        mcn::optim::SGD sgd({&w}, 1.0);
        CHECK_TRUE(sgd.params().size() == 1);
        CHECK_TRUE(sgd.params()[0] == &w);

        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 4.f);

        // 调用方自己再读一次，确认看到的是同一块存储
        CHECK_FLOAT_EQ(*static_cast<const float *>(w.data_ptr()), 4.f);
    }

    void test_sgd_zero_lr()
    {
        std::puts("  test_sgd_zero_lr");

        mcn::Tensor w({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(w, mcn::Tensor({3.f, 4.f}));

        mcn::optim::SGD sgd({&w}, 0.0);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 1.f);
        CHECK_FLOAT_EQ(f_at(w, {1}), 2.f);

        // set_lr 之后才真正开始更新
        sgd.set_lr(0.5);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), -0.5f); // 1 - 0.5*3
        CHECK_FLOAT_EQ(f_at(w, {1}), 0.f);   // 2 - 0.5*4
    }

    void test_sgd_skips_missing_grad()
    {
        std::puts("  test_sgd_skips_missing_grad");

        // 只有 a 有梯度；b 没有参与前向，step() 必须安静地跳过而不是崩
        mcn::Tensor a({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor b({10.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(a, mcn::Tensor({2.f}));

        mcn::optim::SGD sgd({&a, &b}, 0.1);
        sgd.step();

        CHECK_FLOAT_EQ(f_at(a, {0}), 0.8f);
        CHECK_FLOAT_EQ(f_at(b, {0}), 10.f); // 没动
        CHECK_TRUE(b.grad() == nullptr);
    }

    void test_sgd_momentum()
    {
        std::puts("  test_sgd_momentum");

        // p=1, g=2, lr=0.1, momentum=0.9
        //   step1: b = 0.9*0 + 2   = 2    -> p = 1   - 0.2  = 0.8
        //   step2: b = 0.9*2 + 2   = 3.8  -> p = 0.8 - 0.38 = 0.42
        //   step3: b = 0.9*3.8 + 2 = 5.42 -> p = 0.42 - 0.542 = -0.122
        mcn::Tensor w({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        const mcn::Tensor g({2.f});

        mcn::optim::SGD sgd({&w}, 0.1, /*momentum=*/0.9);
        CHECK_FLOAT_EQ(sgd.momentum(), 0.9);
        CHECK_FLOAT_EQ(sgd.weight_decay(), 0.0);

        set_grad(w, g);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.8f);

        set_grad(w, g);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.42f);

        set_grad(w, g);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), -0.122f);
    }

    void test_sgd_weight_decay()
    {
        std::puts("  test_sgd_weight_decay");

        // p=1, g=2, wd=0.1, lr=0.5  =>  g_eff = 2 + 0.1*1 = 2.1, p = 1 - 1.05 = -0.05
        mcn::Tensor w({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(w, mcn::Tensor({2.f}));

        mcn::optim::SGD sgd({&w}, 0.5, /*momentum=*/0.0, /*weight_decay=*/0.1);
        CHECK_FLOAT_EQ(sgd.weight_decay(), 0.1);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), -0.05f);

        // momentum + weight_decay 同时开启
        //   p=1, g=2, wd=0.1, lr=0.5, m=0.5
        //   b1 = 0.5*0 + 2 + 0.1 = 2.1  -> p = 1 - 1.05 = -0.05
        //   b2 = 0.5*2.1 + 2 + 0.1*(-0.05) = 1.05 + 1.995 = 3.045 -> p = -0.05 - 1.5225 = -1.5725
        mcn::Tensor u({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(u, mcn::Tensor({2.f}));

        mcn::optim::SGD sgd2({&u}, 0.5, 0.5, 0.1);
        sgd2.step();
        CHECK_FLOAT_EQ(f_at(u, {0}), -0.05f);
        set_grad(u, mcn::Tensor({2.f}));
        sgd2.step();
        CHECK_FLOAT_EQ(f_at(u, {0}), -1.5725f);
    }

    void test_sgd_zero_grad()
    {
        std::puts("  test_sgd_zero_grad");

        mcn::Tensor w({1.f, 2.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor g({1.f, 1.f});

        mcn::optim::SGD sgd({&w}, 0.1);

        set_grad(w, mcn::Tensor({1.f, 1.f}));
        sgd.step(); // w = [0.9, 1.9]

        // 不清梯度：第二轮累积成 [2, 2]
        w.accumulate_grad(mcn::Tensor({1.f, 1.f}));
        sgd.step(); // w = [0.7, 1.7]
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.7f);
        CHECK_FLOAT_EQ(f_at(w, {1}), 1.7f);

        // 走优化器的 zero_grad()：梯度清空，之后 step() 不会再改参数
        sgd.zero_grad();
        CHECK_TRUE(w.grad() == nullptr);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 0.7f);
        CHECK_FLOAT_EQ(f_at(w, {1}), 1.7f);
    }

    void test_sgd_autograd_e2e()
    {
        std::puts("  test_sgd_autograd_e2e");

        // loss = sum(w * w)，d loss/dw = 2w；w=3 => 6，lr=0.1 => w = 3 - 0.6 = 2.4
        mcn::Tensor w({3.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::optim::SGD sgd({&w}, 0.1);

        mcn::Tensor loss = (w * w).sum();
        loss.backward();
        CHECK_FLOAT_EQ(grad_at(w, {0}), 6.f);

        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 2.4f);

        // 第二轮：梯度由新的 w 重新算出来
        sgd.zero_grad();
        mcn::Tensor loss2 = (w * w).sum();
        loss2.backward();
        CHECK_FLOAT_EQ(grad_at(w, {0}), 4.8f);
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w, {0}), 1.92f); // 2.4 - 0.1*4.8
    }

    void test_sgd_linear_regression()
    {
        std::puts("  test_sgd_linear_regression");

        // 线性回归：y = x @ W，真值 W = [[2], [3]]
        mcn::Tensor x({{1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}, {2.f, 1.f}});
        mcn::Tensor y({{2.f}, {3.f}, {5.f}, {7.f}});

        mcn::Tensor W = mcn::Tensor::randn({2, 1}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::optim::SGD sgd({&W}, 0.05);

        auto loss_fn = [&]()
        {
            mcn::Tensor diff = x.matmul(W) - y;
            return (diff * diff).mean();
        };

        const float first = *static_cast<const float *>(loss_fn().data_ptr());

        // x^T x 的特征值约 7.2 / 1.9，最慢方向每步只收缩 (1 - lr*0.95)，
        // lr=0.05 要几百步才能把残差压到 float 精度
        for (int it = 0; it < 800; ++it)
        {
            sgd.zero_grad();
            loss_fn().backward();
            sgd.step();
        }

        const float last = *static_cast<const float *>(loss_fn().data_ptr());
        CHECK_TRUE(last < first);
        CHECK_TRUE(last < 1e-4f);
        CHECK_FLOAT_NEAR(f_at(W, {0, 0}), 2.f);
        CHECK_FLOAT_NEAR(f_at(W, {1, 0}), 3.f);
        CHECK_TRUE(sgd.step_count() == 800);
    }

    void test_sgd_cuda()
    {
        std::puts("  test_sgd_cuda");

        // 与 CPU 版本数值一致
        mcn::Tensor w({1.f, 2.f}, mcn::Device::CUDA, mcn::DataType::FLOAT32, true);
        set_grad(w, mcn::Tensor({3.f, 4.f}, mcn::Device::CUDA));

        mcn::optim::SGD sgd({&w}, 0.1);
        sgd.step();

        mcn::Tensor host = w.to(mcn::Device::CPU);
        CHECK_FLOAT_NEAR(f_at(host, {0}), 0.7f);
        CHECK_FLOAT_NEAR(f_at(host, {1}), 1.6f);

        // 动量缓冲必须建在参数所在设备上
        mcn::Tensor v({1.f}, mcn::Device::CUDA, mcn::DataType::FLOAT32, true);
        set_grad(v, mcn::Tensor({2.f}, mcn::Device::CUDA));
        mcn::optim::SGD sgd2({&v}, 0.1, 0.9);
        sgd2.step();
        set_grad(v, mcn::Tensor({2.f}, mcn::Device::CUDA));
        sgd2.step();

        mcn::Tensor host_v = v.to(mcn::Device::CPU);
        CHECK_FLOAT_NEAR(f_at(host_v, {0}), 0.42f);
    }

    void test_sgd_add_param_group()
    {
        std::puts("  test_sgd_add_param_group");

        // 分批加入参数：后加入的那个也要拿到自己的动量缓冲（否则 update_one 会越界）
        mcn::Tensor w1({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor w2({1.f}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);

        mcn::optim::SGD sgd({&w1}, 0.1, /*momentum=*/0.9);
        set_grad(w1, mcn::Tensor({2.f}));
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w1, {0}), 0.8f);

        sgd.add_param_group({&w2});
        CHECK_TRUE(sgd.params().size() == 2);

        // w2 的动量从 0 开始：b = 0.9*0 + 2 = 2 => p = 1 - 0.2 = 0.8
        set_grad(w1, mcn::Tensor({2.f}));
        set_grad(w2, mcn::Tensor({2.f}));
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w1, {0}), 0.42f); // b1 = 0.9*2 + 2 = 3.8
        CHECK_FLOAT_EQ(f_at(w2, {0}), 0.8f);

        // 再走一步，两个缓冲各自独立演进
        set_grad(w1, mcn::Tensor({2.f}));
        set_grad(w2, mcn::Tensor({2.f}));
        sgd.step();
        CHECK_FLOAT_EQ(f_at(w1, {0}), -0.122f); // b1 = 0.9*3.8 + 2 = 5.42
        CHECK_FLOAT_EQ(f_at(w2, {0}), 0.42f);   // b2 = 0.9*2   + 2 = 3.8
    }

    void test_sgd_float64()
    {
        std::puts("  test_sgd_float64");

        // float64 参数
        mcn::Tensor d({1.0, 2.0}, mcn::Device::CPU, mcn::DataType::FLOAT64, true);
        set_grad(d, mcn::Tensor({3.0, 4.0}, mcn::Device::CPU, mcn::DataType::FLOAT64));

        mcn::optim::SGD sgd({&d}, 0.1);
        sgd.step();

        CHECK_DTYPE(d, mcn::DataType::FLOAT64);
        const double *dp = d.data_ptr<double>();
        CHECK_FLOAT_EQ(dp[0], 0.7);
        CHECK_FLOAT_EQ(dp[1], 1.6);

        // 3-D 参数：更新必须按线性下标正确进行，不能假定只有 1-D
        mcn::Tensor t3 = mcn::Tensor::ones({2, 2, 2}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        set_grad(t3, mcn::Tensor::ones({2, 2, 2}));
        mcn::optim::SGD sgd3({&t3}, 0.25);
        sgd3.step();

        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                for (size_t k = 0; k < 2; ++k)
                    CHECK_FLOAT_EQ(f_at(t3, {i, j, k}), 0.75f);
    }
} // namespace

int main()
{
    std::puts("=== optim tests ===");

    test_sgd_basic();
    test_sgd_updates_original();
    test_sgd_zero_lr();
    test_sgd_skips_missing_grad();
    test_sgd_momentum();
    test_sgd_weight_decay();
    test_sgd_zero_grad();
    test_sgd_autograd_e2e();
    test_sgd_linear_regression();
    test_sgd_cuda();
    test_sgd_add_param_group();
    test_sgd_float64();

    std::printf("\n%d checks, %d failure(s)\n", mcn_test::g_checks, mcn_test::g_failures);
    return mcn_test::g_failures == 0 ? 0 : 1;
}
