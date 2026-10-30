#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "mini_cuda_nn/mini_cuda_nn.h"

#include "snake.h"

namespace dqn
{
    struct Config
    {
        int state_dim = snake::kStateDim;
        int hidden = 64;
        int actions = snake::kNumActions;

        double gamma = 0.9;
        double lr = 0.002;
        double momentum = 0.9;

        double eps_start = 1.0;
        double eps_end = 0.02;
        int eps_decay_steps = 8000;

        size_t replay_capacity = 50000;
        size_t batch = 64;
        // 回放池里先存够这么多条再开始学：样本太少时全是同质数据，
        // 头几百次更新会把网络拉向“刚才碰巧做的事”
        size_t warmup = 1000;
        // 每个环境步做几次梯度下降（DQN 惯例是 1，小网络可以多跑几次）
        int train_per_step = 2;
        int target_sync_steps = 500;

        mcn::Device device = mcn::Device::CPU;
    };

    // 经验回放 + 双网络 DQN。
    //
    // 目标值只对"实际执行的那个动作"那一列生效，其余列用当前 Q 值填充 ——
    // 这样 MSE 的梯度天然只回传到被选中的动作上，等价于 gather，
    // 而框架里暂时没有 gather / index_select 算子。
    class Agent
    {
    public:
        explicit Agent(Config cfg, unsigned seed = 0);

        // 选动作。greedy=true 时纯贪心（评估用），否则 epsilon-greedy
        int act(const snake::State &s, bool greedy);

        void remember(const snake::State &s, int action, double reward,
                      const snake::State &s_next, bool done);

        // 采样一个 batch 做一次更新，返回 loss；样本不足时返回 -1
        double train_step();

        void sync_target();

        double epsilon() const;
        size_t replay_size() const { return buffer_.size(); }
        size_t train_steps() const { return train_steps_; }
        size_t env_steps() const { return env_steps_; }
        const Config &config() const { return cfg_; }
        mcn::nn::Sequential &online() { return online_; }

        // 诊断用：某个状态下的 Q 值
        std::array<double, 3> q_values(const snake::State &s);

    private:
        struct Transition
        {
            snake::State s;
            snake::State s_next;
            int action = 0;
            double reward = 0.0;
            bool done = false;
        };

        Config cfg_;

        mcn::nn::Sequential online_;
        mcn::nn::Sequential target_;
        mcn::nn::MSELoss loss_fn_;
        // 优化器持有 online_ 参数的指针，所以必须在网络建好之后再构造
        std::unique_ptr<mcn::optim::SGD> opt_;

        std::deque<Transition> buffer_;
        std::mt19937 rng_;

        size_t env_steps_ = 0;
        size_t train_steps_ = 0;
    };
} // namespace dqn
