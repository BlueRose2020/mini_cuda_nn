// nn 单元测试：Module 基类契约、Linear、ReLU、Sequential、MSELoss
#include "test_common.h"

#include "mini_cuda_nn/autograd/grad_mode.h"
#include "mini_cuda_nn/nn/activation.h"
#include "mini_cuda_nn/nn/linear.h"
#include "mini_cuda_nn/nn/loss.h"
#include "mini_cuda_nn/nn/sequential.h"
#include "mini_cuda_nn/optim/sgd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace
{
    mcn::Tensor cpu_f32(std::initializer_list<float> v, bool require_grad = false)
    {
        return mcn::Tensor(v, mcn::Device::CPU, mcn::DataType::FLOAT32, require_grad);
    }

    void test_linear_forward()
    {
        std::puts("  test_linear_forward");

        mcn::nn::Linear lin(3, 2);
        CHECK_TRUE(std::string(lin.name()) == "linear");
        CHECK_TRUE(lin.in_features() == 3);
        CHECK_TRUE(lin.out_features() == 2);
        CHECK_SHAPE(lin.weight(), 2, 3);
        CHECK_SHAPE(lin.bias(), 2);
        CHECK_TRUE(lin.weight().require_grad());
        CHECK_TRUE(lin.num_parameters() == 2 * 3 + 2);

        // 固定权重，方便手算
        lin.weight().copy_(mcn::Tensor({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
        lin.bias().copy_(cpu_f32({0.5f, -0.5f}));

        mcn::Tensor x({{1.f, 1.f, 1.f}, {2.f, 0.f, 1.f}});
        mcn::Tensor y = lin(x);

        CHECK_SHAPE(y, 2, 2);
        CHECK_FLOAT_NEAR(f_at(y, {0, 0}), 6.5f);  // 1+2+3+0.5
        CHECK_FLOAT_NEAR(f_at(y, {0, 1}), 14.5f); // 4+5+6-0.5
        CHECK_FLOAT_NEAR(f_at(y, {1, 0}), 5.5f);  // 2*1+0+1*3+0.5
        CHECK_FLOAT_NEAR(f_at(y, {1, 1}), 13.5f); // 2*4+0+1*6-0.5

        // 单样本也应该成立（1 行）
        mcn::Tensor one({{1.f, 1.f, 1.f}});
        CHECK_SHAPE(lin(one), 1, 2);
    }

    void test_linear_backward()
    {
        std::puts("  test_linear_backward");

        mcn::nn::Linear lin(3, 2);
        lin.weight().copy_(mcn::Tensor({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
        lin.bias().copy_(cpu_f32({0.5f, -0.5f}));
        lin.zero_grad();

        mcn::Tensor x({{1.f, 1.f, 1.f}, {2.f, 0.f, 1.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        mcn::Tensor y = lin(x);

        // loss = sum(y) => dloss/dy 全为 1
        y.sum().backward();

        // dW[j][k] = sum_i x[i][k] = [3, 1, 2]，对每一行 j 都相同
        const mcn::Tensor *gw = lin.weight().grad();
        CHECK_TRUE(gw != nullptr);
        if (gw)
        {
            CHECK_FLOAT_NEAR(f_at(*gw, {0, 0}), 3.f);
            CHECK_FLOAT_NEAR(f_at(*gw, {0, 1}), 1.f);
            CHECK_FLOAT_NEAR(f_at(*gw, {0, 2}), 2.f);
            CHECK_FLOAT_NEAR(f_at(*gw, {1, 0}), 3.f);
            CHECK_FLOAT_NEAR(f_at(*gw, {1, 2}), 2.f);
        }

        // db[j] = batch size = 2
        const mcn::Tensor *gb = lin.bias().grad();
        CHECK_TRUE(gb != nullptr);
        if (gb)
        {
            CHECK_FLOAT_NEAR(f_at(*gb, {0}), 2.f);
            CHECK_FLOAT_NEAR(f_at(*gb, {1}), 2.f);
        }

        // dx[i][k] = sum_j W[j][k] = [5, 7, 9]
        CHECK_FLOAT_NEAR(grad_at(x, {0, 0}), 5.f);
        CHECK_FLOAT_NEAR(grad_at(x, {0, 1}), 7.f);
        CHECK_FLOAT_NEAR(grad_at(x, {1, 2}), 9.f);
    }

    void test_relu()
    {
        std::puts("  test_relu");

        mcn::Tensor x = cpu_f32({-1.f, 0.f, 2.f, -3.f}, true);
        mcn::Tensor r = x.relu();
        CHECK_SHAPE(r, 4);
        CHECK_FLOAT_EQ(f_at(r, {0}), 0.f);
        CHECK_FLOAT_EQ(f_at(r, {1}), 0.f);
        CHECK_FLOAT_EQ(f_at(r, {2}), 2.f);
        CHECK_FLOAT_EQ(f_at(r, {3}), 0.f);
        CHECK_TRUE(r.grad_fn() != nullptr);

        r.sum().backward();
        CHECK_FLOAT_EQ(grad_at(x, {0}), 0.f);
        CHECK_FLOAT_EQ(grad_at(x, {1}), 0.f);
        CHECK_FLOAT_EQ(grad_at(x, {2}), 1.f);
        CHECK_FLOAT_EQ(grad_at(x, {3}), 0.f);

        // 复合：y = relu(x) * 2 => dy/dx = 2 * [x > 0]
        mcn::Tensor a = cpu_f32({-1.f, 3.f}, true);
        (a.relu() * 2).sum().backward();
        CHECK_FLOAT_EQ(grad_at(a, {0}), 0.f);
        CHECK_FLOAT_EQ(grad_at(a, {1}), 2.f);

        // relu 不变动原张量
        CHECK_FLOAT_EQ(f_at(x, {0}), -1.f);
    }

    void test_mse_loss()
    {
        std::puts("  test_mse_loss");

        mcn::nn::MSELoss loss_fn;
        CHECK_TRUE(std::string(loss_fn.name()) == "mse_loss");
        CHECK_TRUE(loss_fn.reduction() == mcn::nn::Reduction::Mean);

        mcn::Tensor pred = cpu_f32({1.f, 2.f, 3.f}, true);
        mcn::Tensor target = cpu_f32({1.f, 1.f, 1.f});

        mcn::Tensor loss = loss_fn(pred, target);
        CHECK_TRUE(loss.ndim() == 0);
        CHECK_FLOAT_NEAR(loss.item(), 5.0 / 3.0); // (0+1+4)/3

        loss.backward();
        // d/dpred = 2*(pred-target)/n = [0, 2/3, 4/3]
        CHECK_FLOAT_NEAR(grad_at(pred, {0}), 0.f);
        CHECK_FLOAT_NEAR(grad_at(pred, {1}), 2.0 / 3.0);
        CHECK_FLOAT_NEAR(grad_at(pred, {2}), 4.0 / 3.0);

        // Sum / None
        mcn::nn::MSELoss sum_loss(mcn::nn::Reduction::Sum);
        mcn::Tensor p2 = cpu_f32({1.f, 2.f, 3.f}, true);
        CHECK_FLOAT_NEAR(sum_loss(p2, target).item(), 5.0);

        mcn::nn::MSELoss none_loss(mcn::nn::Reduction::None);
        mcn::Tensor p3 = cpu_f32({1.f, 2.f, 3.f}, true);
        mcn::Tensor e = none_loss(p3, target);
        CHECK_SHAPE(e, 3);
        CHECK_FLOAT_NEAR(f_at(e, {2}), 4.f);

        // target 带 require_grad 也不该收到梯度（detach 过）
        mcn::Tensor t_grad = cpu_f32({1.f, 1.f, 1.f}, true);
        mcn::Tensor p4 = cpu_f32({1.f, 2.f, 3.f}, true);
        loss_fn(p4, t_grad).backward();
        CHECK_TRUE(t_grad.grad() == nullptr);
        CHECK_TRUE(p4.grad() != nullptr);
    }

    void test_sequential()
    {
        std::puts("  test_sequential");

        mcn::nn::Sequential net;
        net.add<mcn::nn::Linear>(2, 4).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(4, 1);

        CHECK_TRUE(std::string(net.name()) == "sequential");
        CHECK_TRUE(net.size() == 3);
        // 2*4 + 4 + 4*1 + 1
        CHECK_TRUE(net.num_parameters() == 17);
        CHECK_TRUE(net.parameters().size() == 4);

        const auto named = net.named_parameters();
        CHECK_TRUE(named.size() == 4);
        CHECK_TRUE(named[0].first == "0.weight");
        CHECK_TRUE(named[1].first == "0.bias");
        CHECK_TRUE(named[2].first == "2.weight");
        CHECK_TRUE(named[3].first == "2.bias");

        mcn::Tensor x({{1.f, 2.f}, {3.f, 4.f}});
        CHECK_SHAPE(net(x), 2, 1);

        // describe() 至少别崩、且带层类型名
        const std::string d = net.describe();
        CHECK_TRUE(d.find("sequential") != std::string::npos);
        CHECK_TRUE(d.find("0.weight") != std::string::npos);
    }

    void test_sequential_training()
    {
        std::puts("  test_sequential_training");

        // 拟合 y = 2*x0 + 3*x1 + 1
        mcn::Tensor x({{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}});
        mcn::Tensor y({{1.f}, {4.f}, {3.f}, {6.f}});

        // 固定初始化：不固定的话偶尔会抽到让 ReLU 全死的权重，测试变成随机失败
        mcn::nn::manual_seed(2024);
        mcn::nn::Sequential net;
        net.add<mcn::nn::Linear>(2, 8).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(8, 1);

        mcn::optim::SGD opt(net.parameters(), 0.05, /*momentum=*/0.9);
        mcn::nn::MSELoss loss_fn;

        auto loss_value = [&]()
        {
            mcn::detail::NoGradGuard guard;
            return loss_fn(net(x), y).item();
        };

        const double first = loss_value();

        for (int it = 0; it < 600; ++it)
        {
            opt.zero_grad();
            mcn::Tensor loss = loss_fn(net(x), y);
            loss.backward();
            opt.step();
        }

        const double last = loss_value();
        CHECK_TRUE(last < first);
        CHECK_TRUE(last < 1e-3);

        // 预测值应接近真值
        mcn::detail::NoGradGuard guard;
        mcn::Tensor pred = net(x);
        CHECK_FLOAT(f_at(pred, {0, 0}), 1.f, 1e-3f);
        CHECK_FLOAT(f_at(pred, {1, 0}), 4.f, 1e-3f);
        CHECK_FLOAT(f_at(pred, {2, 0}), 3.f, 1e-3f);
        CHECK_FLOAT(f_at(pred, {3, 0}), 6.f, 1e-3f);
    }

    void test_copy_parameters()
    {
        std::puts("  test_copy_parameters");

        mcn::nn::Sequential online;
        online.add<mcn::nn::Linear>(2, 3).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(3, 1);

        mcn::nn::Sequential target;
        target.add<mcn::nn::Linear>(2, 3).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(3, 1);

        // 复制后数值必须一致
        mcn::nn::copy_parameters(target, online);

        auto a = online.named_parameters();
        auto b = target.named_parameters();
        CHECK_TRUE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            const float *pa = a[i].second->data_ptr<float>();
            const float *pb = b[i].second->data_ptr<float>();
            for (size_t j = 0; j < a[i].second->numel(); ++j)
                CHECK_FLOAT_EQ(pb[j], pa[j]);
        }

        // 关键：不共享存储。把 target 的权重写成 1，online 不应该跟着变
        const float before = a[0].second->data_ptr<float>()[0];
        b[0].second->fill_(1.0);
        CHECK_FLOAT_EQ(a[0].second->data_ptr<float>()[0], before);
        CHECK_FLOAT_EQ(b[0].second->data_ptr<float>()[0], 1.f);
    }

    void test_module_zero_grad_and_detach()
    {
        std::puts("  test_module_zero_grad_and_detach");

        mcn::nn::Linear lin(2, 2);
        mcn::Tensor x({{1.f, 2.f}}, mcn::Device::CPU, mcn::DataType::FLOAT32, true);
        lin(x).sum().backward();
        CHECK_TRUE(lin.weight().grad() != nullptr);

        lin.zero_grad();
        CHECK_TRUE(lin.weight().grad() == nullptr);
        CHECK_TRUE(lin.bias().grad() == nullptr);

        // detach：共享存储但不参与计算图
        mcn::Tensor a = cpu_f32({1.f, 2.f}, true);
        mcn::Tensor d = a.detach();
        CHECK_TRUE(!d.require_grad());
        CHECK_TRUE(d.grad_fn() == nullptr);
        static_cast<float *>(d.data_ptr())[0] = 7.f;
        CHECK_FLOAT_EQ(f_at(a, {0}), 7.f); // 共享存储
    }

    // 数值梯度校验：用中心差分验证 autograd 算的梯度。
    // 这是框架正确性的硬护栏 —— 之前 DQN 训练不出东西时，
    // 靠它 + 一个监督学习基线才排除了"反向传播写错"的可能。
    void test_gradient_check()
    {
        std::puts("  test_gradient_check");

        mcn::nn::manual_seed(99);
        mcn::nn::Linear lin(4, 2);
        mcn::nn::MSELoss loss_fn;

        std::vector<float> xd = {1.f, -2.f, 0.5f, 3.f};
        mcn::Tensor x({1, 4}, xd.data());
        std::vector<float> td = {0.3f, -0.7f};
        mcn::Tensor t({1, 2}, td.data());

        // 绕开 autograd 手算 loss
        auto loss_at = [&]()
        {
            mcn::detail::NoGradGuard ng;
            mcn::Tensor out = x.matmul(lin.weight().t()) + lin.bias();
            mcn::Tensor d = out - t;
            return (d * d).mean().item();
        };

        // 必须对**标量 loss** 求导：解析梯度和数值梯度比的是同一个量。
        // （之前误写成 lin(x).backward()，那算的是 d(sum(out))/dW，自然对不上。）
        mcn::Tensor loss_var = loss_fn(lin(x), t);
        loss_var.backward();
        CHECK_TRUE(lin.weight().grad() != nullptr);

        const float *gp = lin.weight().grad()->data_ptr<float>();
        float *wp = lin.weight().data_ptr<float>();

        const double eps = 1e-3;
        double max_err = 0.0;
        for (size_t i = 0; i < lin.weight().numel(); ++i)
        {
            const float orig = wp[i];
            wp[i] = orig + static_cast<float>(eps);
            const double lp = loss_at();
            wp[i] = orig - static_cast<float>(eps);
            const double lm = loss_at();
            wp[i] = orig;

            max_err = std::max(max_err, std::abs((lp - lm) / (2 * eps) - gp[i]));
        }
        // 单精度 + eps=1e-3，1e-4 已经是很紧的界（实测 ~1.6e-5）
        CHECK_TRUE(max_err < 1e-4);
        if (max_err >= 1e-4)
            std::printf("    max_err = %.3e\n", max_err);
    }

    // 监督学习基线：前向 + 反向 + SGD 全链路能否真的把 loss 压下去。
    // 与 DQN 无关，纯粹用于在"训练不收敛"时区分是框架问题还是算法问题。
    void test_supervised_baseline()
    {
        std::puts("  test_supervised_baseline");

        mcn::nn::manual_seed(7);
        mcn::nn::Sequential net;
        net.add<mcn::nn::Linear>(8, 32).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(32, 3);

        // 线性可分的随机二分类数据
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> bit(0, 1);
        std::vector<float> xs, ys;
        const int N = 256;
        for (int i = 0; i < N; ++i)
        {
            float s[8];
            for (float &v : s)
                v = static_cast<float>(bit(rng));
            const int label = (s[0] + 2 * s[1] >= s[2] + 2 * s[3]) ? 0 : 1;
            xs.insert(xs.end(), s, s + 8);
            for (int a = 0; a < 3; ++a)
                ys.push_back(a == label ? 1.f : 0.f);
        }

        mcn::Tensor X({static_cast<size_t>(N), 8}, xs.data());
        mcn::Tensor Y({static_cast<size_t>(N), 3}, ys.data());

        mcn::optim::SGD opt(net.parameters(), 0.1);
        mcn::nn::MSELoss loss_fn;

        double first = 0.0, last = 0.0;
        for (int it = 1; it <= 200; ++it)
        {
            opt.zero_grad();
            mcn::Tensor loss = loss_fn(net(X), Y);
            loss.backward();
            opt.step();
            if (it == 1)
                first = loss.item();
            last = loss.item();
        }

        CHECK_TRUE(last < first * 0.5); // 至少砍掉一半
    }

    void test_module_to_cuda()
    {
        std::puts("  test_module_to_cuda");

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            std::puts("    (skip: no CUDA device)");
            return;
        }

        mcn::nn::Sequential net;
        net.add<mcn::nn::Linear>(2, 4).add<mcn::nn::ReLU>().add<mcn::nn::Linear>(4, 1);

        // 迁移前后优化器持有的指针必须还是同一个对象
        std::vector<mcn::Tensor *> params_before = net.parameters();
        net.to(mcn::Device::CUDA);
        std::vector<mcn::Tensor *> params_after = net.parameters();
        CHECK_TRUE(params_before == params_after);
        CHECK_TRUE(net.parameters()[0]->device() == mcn::Device::CUDA);

        mcn::Tensor x({{1.f, 2.f}, {3.f, 4.f}}, mcn::Device::CUDA, mcn::DataType::FLOAT32);
        mcn::Tensor y = net(x);
        CHECK_SHAPE(y, 2, 1);
        CHECK_TRUE(y.device() == mcn::Device::CUDA);

        mcn::nn::MSELoss loss_fn;
        mcn::Tensor tgt = mcn::Tensor::zeros({2, 1}, mcn::Device::CUDA, mcn::DataType::FLOAT32);
        mcn::Tensor loss = loss_fn(y, tgt);
        loss.backward();
        CHECK_TRUE(net.parameters()[0]->grad() != nullptr);

        mcn::optim::SGD opt(net.parameters(), 0.1);
        opt.step();
        opt.zero_grad();
        CHECK_TRUE(net.parameters()[0]->grad() == nullptr);
    }
} // namespace

int main()
{
    std::puts("=== nn tests ===");

    test_gradient_check();
    test_supervised_baseline();
    test_linear_forward();
    test_linear_backward();
    test_relu();
    test_mse_loss();
    test_sequential();
    test_sequential_training();
    test_copy_parameters();
    test_module_zero_grad_and_detach();
    test_gradient_check();
    test_supervised_baseline();
    test_module_to_cuda();

    std::printf("\n%d checks, %d failure(s)\n", mcn_test::g_checks, mcn_test::g_failures);
    return mcn_test::g_failures == 0 ? 0 : 1;
}
