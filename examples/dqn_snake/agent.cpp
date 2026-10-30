#include "agent.h"

#include "mini_cuda_nn/autograd/grad_mode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace dqn
{
    namespace
    {
        mcn::Tensor make_states(const std::vector<snake::State> &rows, size_t count,
                                mcn::Device device)
        {
            std::vector<float> flat;
            flat.reserve(count * snake::kStateDim);
            for (size_t i = 0; i < count; ++i)
                flat.insert(flat.end(), rows[i].begin(), rows[i].end());

            return mcn::Tensor({count, static_cast<size_t>(snake::kStateDim)},
                               flat.data(), device, mcn::DataType::FLOAT32);
        }
    } // namespace

    Agent::Agent(Config cfg, unsigned seed)
        : cfg_(cfg), rng_(seed)
    {
        MCN_CHECK(cfg_.state_dim == snake::kStateDim,
                  "Agent: state_dim {} must match snake::kStateDim {}",
                  cfg_.state_dim, snake::kStateDim);
        MCN_CHECK(cfg_.batch > 0 && cfg_.hidden > 0, "Agent: batch and hidden must be positive");

        // 权重初始化可复现
        mcn::nn::manual_seed(seed);

        for (mcn::nn::Sequential *net : {&online_, &target_})
        {
            net->add<mcn::nn::Linear>(static_cast<size_t>(cfg_.state_dim),
                                      static_cast<size_t>(cfg_.hidden),
                                      /*require_grad=*/true, cfg_.device)
                .add<mcn::nn::ReLU>()
                .add<mcn::nn::Linear>(static_cast<size_t>(cfg_.hidden),
                                      static_cast<size_t>(cfg_.hidden),
                                      /*require_grad=*/true, cfg_.device)
                .add<mcn::nn::ReLU>()
                .add<mcn::nn::Linear>(static_cast<size_t>(cfg_.hidden),
                                      static_cast<size_t>(cfg_.actions),
                                      /*require_grad=*/true, cfg_.device);
        }

        // target 一开始与 online 完全一致
        sync_target();

        opt_ = std::make_unique<mcn::optim::SGD>(online_.parameters(), cfg_.lr, cfg_.momentum);
    }

    void Agent::sync_target()
    {
        mcn::nn::copy_parameters(target_, online_);
    }

    double Agent::epsilon() const
    {
        if (cfg_.eps_decay_steps <= 0)
            return cfg_.eps_end;
        const double frac = std::min(1.0, static_cast<double>(env_steps_) / cfg_.eps_decay_steps);
        return cfg_.eps_start + (cfg_.eps_end - cfg_.eps_start) * frac;
    }

    std::array<double, 3> Agent::q_values(const snake::State &s)
    {
        mcn::detail::NoGradGuard guard;

        const std::vector<snake::State> rows{s};
        mcn::Tensor x = make_states(rows, 1, cfg_.device);
        mcn::Tensor q = online_.forward(x);
        mcn::Tensor host = q.to(mcn::Device::CPU);

        const float *p = host.data_ptr<float>();
        return {p[0], p[1], p[2]};
    }

    int Agent::act(const snake::State &s, bool greedy)
    {
        ++env_steps_;

        if (!greedy)
        {
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            if (uni(rng_) < epsilon())
                return std::uniform_int_distribution<int>(0, cfg_.actions - 1)(rng_);
        }

        const auto q = q_values(s);
        return static_cast<int>(std::max_element(q.begin(), q.end()) - q.begin());
    }

    void Agent::remember(const snake::State &s, int action, double reward,
                         const snake::State &s_next, bool done)
    {
        buffer_.push_back(Transition{s, s_next, action, reward, done});
        if (buffer_.size() > cfg_.replay_capacity)
            buffer_.pop_front();
    }

    double Agent::train_step()
    {
        // 预热：样本不够就别学，否则头几百次更新只会拟合刚刚那一小段轨迹
        if (buffer_.size() < std::max(cfg_.batch, cfg_.warmup))
            return -1.0;

        const size_t B = cfg_.batch;
        std::uniform_int_distribution<size_t> pick(0, buffer_.size() - 1);

        std::vector<snake::State> states(B), next_states(B);
        std::vector<int> actions(B);
        std::vector<double> rewards(B);
        std::vector<char> dones(B);

        for (size_t i = 0; i < B; ++i)
        {
            const Transition &t = buffer_[pick(rng_)];
            states[i] = t.s;
            next_states[i] = t.s_next;
            actions[i] = t.action;
            rewards[i] = t.reward;
            dones[i] = t.done ? 1 : 0;
        }

        mcn::Tensor st = make_states(states, B, cfg_.device);
        mcn::Tensor nst = make_states(next_states, B, cfg_.device);

        // 1) 目标网络算下一状态的最大 Q。整段不建图。
        std::vector<float> max_next(B, 0.f);
        {
            mcn::detail::NoGradGuard guard;
            mcn::Tensor qn = target_.forward(nst);
            mcn::Tensor qn_host = qn.to(mcn::Device::CPU);
            const float *p = qn_host.data_ptr<float>();
            for (size_t i = 0; i < B; ++i)
                max_next[i] = std::max({p[i * 3 + 0], p[i * 3 + 1], p[i * 3 + 2]});
        }

        // 2) 在线网络算 Q（要梯度）
        mcn::Tensor q = online_.forward(st);

        // 3) 目标张量：先用当前 Q 的数值，再把选中动作那一列换成 TD 目标。
        //    这样 MSE 对未选中动作的梯度恰好为 0，效果等价于 gather。
        mcn::Tensor q_host = q.detach().to(mcn::Device::CPU);
        std::vector<float> target_values(q_host.numel());
        std::memcpy(target_values.data(), q_host.data_ptr<float>(),
                    target_values.size() * sizeof(float));

        for (size_t i = 0; i < B; ++i)
        {
            const float bootstrap = dones[i] ? 0.f : static_cast<float>(cfg_.gamma) * max_next[i];
            target_values[i * 3 + static_cast<size_t>(actions[i])] =
                static_cast<float>(rewards[i]) + bootstrap;
        }

        mcn::Tensor target_q({B, static_cast<size_t>(cfg_.actions)},
                             target_values.data(), cfg_.device, mcn::DataType::FLOAT32);

        // 4) 一步 SGD
        opt_->zero_grad();
        mcn::Tensor loss = loss_fn_(q, target_q);
        loss.backward();
        opt_->step();

        ++train_steps_;
        // 目标网络周期同步：直接 copy 权重会共享存储，必须逐个 copy_
        if (cfg_.target_sync_steps > 0 &&
            train_steps_ % static_cast<size_t>(cfg_.target_sync_steps) == 0)
            sync_target();

        return loss.item();
    }
} // namespace dqn
