// DQN 贪吃蛇：用本框架的训练栈训练一个会玩贪吃蛇的智能体。
//
//   强化学习部分是手写的，框架负责的只有：
//     前向（Linear/ReLU/Sequential）、损失（MSELoss）、反向（autograd）、更新（SGD）。
//
//   状态：11 维手写特征（3 个危险位 + 4 个朝向位 + 4 个食物相对位置位）
//   动作：直行 / 右转 / 左转（相对朝向）
//   奖励：吃到食物 +1，死亡 -1，绕圈太久 -0.5
//
// 用法：
//   dqn_snake                      默认训练 500 局
//   dqn_snake --episodes 2000      训练更久（收敛需要上千局）
//   dqn_snake --render             训练完用贪心策略演示一局
//   dqn_snake --device cuda        在 GPU 上算（网络很小，不一定更快）
//   dqn_snake --eval 20            评估局数
#include "agent.h"
#include "snake.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
    struct Options
    {
        int episodes = 500;
        int eval_episodes = 20;
        unsigned seed = 42;
        int hidden = 64;
        bool render = false;
        bool probe = false;
        mcn::Device device = mcn::Device::CPU;
    };

    Options parse(int argc, char **argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            auto next_int = [&]() -> int
            {
                if (i + 1 >= argc)
                {
                    std::fprintf(stderr, "%s 缺少参数\n", a.c_str());
                    std::exit(1);
                }
                return std::atoi(argv[++i]);
            };

            if (a == "--episodes")
                o.episodes = next_int();
            else if (a == "--eval")
                o.eval_episodes = next_int();
            else if (a == "--seed")
                o.seed = static_cast<unsigned>(next_int());
            else if (a == "--hidden")
                o.hidden = next_int();
            else if (a == "--render")
                o.render = true;
            else if (a == "--probe")
                o.probe = true;
            else if (a == "--device")
            {
                if (i + 1 >= argc)
                {
                    std::fprintf(stderr, "--device 需要 cuda 或 cpu\n");
                    std::exit(1);
                }
                const std::string d = argv[++i];
                o.device = (d == "cuda" || d == "gpu") ? mcn::Device::CUDA : mcn::Device::CPU;
            }
            else if (a == "--help" || a == "-h")
            {
                std::puts("用法: dqn_snake [--episodes N] [--eval N] [--seed N] "
                          "[--hidden N] [--device cpu|cuda] [--render] [--probe]");
                std::exit(0);
            }
            else
            {
                std::fprintf(stderr, "未知参数: %s\n", a.c_str());
                std::exit(1);
            }
        }
        return o;
    }

    // 跑一局，返回得分。greedy 决定用不用 epsilon
    int run_episode(dqn::Agent &agent, snake::Game &game, bool greedy,
                    double *loss_sum, int *loss_count)
    {
        game.reset();
        snake::State s = game.state();

        while (true)
        {
            const int action = agent.act(s, greedy);
            const snake::StepResult r = game.step(static_cast<snake::Action>(action));
            const snake::State s_next = game.state();

            // 只有训练时才存经验。评估跑的是贪心策略，它的状态分布和训练策略不同，
            // 混进回放池会把数据带偏。
            if (!greedy)
            {
                agent.remember(s, action, r.reward, s_next, r.done);

                for (int k = 0; k < agent.config().train_per_step; ++k)
                {
                    const double loss = agent.train_step();
                    if (loss >= 0.0)
                    {
                        *loss_sum += loss;
                        ++(*loss_count);
                    }
                }
            }

            s = s_next;
            if (r.done)
                break;
        }
        return game.score();
    }

    // ---- 诊断：贪心对局到底在干什么 ----
    //
    // 判“绕圈”用**短周期**：同一对 (蛇头位置, 朝向) 在 ≤ kLoopWindow 步内重现。
    // 只看“重复次数”会误判 —— 得分 40+、步数 500+ 的对局里，蛇身很长，
    // 自然会把同一格反复踩过，那不是循环。真正的退化是“几秒内绕回同一姿态”。
    constexpr int kLoopWindow = 12;

    struct EpisodeTrace
    {
        int len = 0;
        int score = 0;
        int longest_no_food = 0;
        int tight_loop_steps = 0; // 处于短周期内的步数（只算第 2 圈及以后）
        double loop_ratio = 0.0;  // tight_loop_steps / len
        double final_reward = 0.0;
        snake::StepResult::End end = snake::StepResult::End::None;
        // Q 值诊断：如果 q_spread 极小，argmax 基本是在噪声上选，自然会锁成固定动作序列
        double q_min = 1e9;
        double q_max = -1e9;
        double q_mean_margin = 0.0; // |best - second| 的平均值
    };

    EpisodeTrace probe_episode(dqn::Agent &agent, snake::Game &game)
    {
        EpisodeTrace tr;
        game.reset();
        snake::State s = game.state();

        std::unordered_map<int, int> last_seen; // key -> 上次访问的步号
        int since_food = 0;
        int step = 0;

        while (true)
        {
            const snake::Point h = game.body().front();
            const int key = (((h.y * 16) + h.x) << 2) | static_cast<int>(game.direction());
            auto it = last_seen.find(key);
            if (it != last_seen.end() && step - it->second <= kLoopWindow)
                ++tr.tight_loop_steps;
            last_seen[key] = step;
            ++step;

            const auto q = agent.q_values(s);
            std::array<double, 3> sorted = q;
            std::sort(sorted.begin(), sorted.end());
            tr.q_min = std::min(tr.q_min, sorted[0]);
            tr.q_max = std::max(tr.q_max, sorted[2]);
            tr.q_mean_margin += (sorted[2] - sorted[1]);

            const int action = agent.act(s, /*greedy=*/true);
            const snake::StepResult r = game.step(static_cast<snake::Action>(action));
            tr.final_reward = r.reward;
            s = game.state();

            if (r.ate)
            {
                since_food = 0;
                tr.score = game.score();
            }
            else
            {
                ++since_food;
                if (since_food > tr.longest_no_food)
                    tr.longest_no_food = since_food;
            }

            if (r.done)
            {
                tr.len = game.steps();
                tr.end = r.end;
                if (tr.len > 0)
                {
                    tr.q_mean_margin /= tr.len;
                    tr.loop_ratio = static_cast<double>(tr.tight_loop_steps) / tr.len;
                }
                break;
            }
        }
        return tr;
    }

    void run_probe(dqn::Agent &agent, snake::Game &game, int episodes)
    {
        std::printf("\n=== 贪心对局行为诊断（%d 局）===\n", episodes);
        std::printf(" 局  步数  得分  最长未进食  短周期步数占比  Q范围        argmax差距  结局\n");

        int looped = 0, starved = 0, crashed = 0;
        long total_steps = 0, total_score = 0;

        for (int i = 1; i <= episodes; ++i)
        {
            const EpisodeTrace tr = probe_episode(agent, game);
            total_steps += tr.len;
            total_score += tr.score;
            if (tr.loop_ratio > 0.5)
                ++looped;
            if (tr.end == snake::StepResult::End::Starve)
                ++starved;
            else if (tr.end == snake::StepResult::End::Crash)
                ++crashed;

            std::printf("%3d %5d %5d %11d %14.0f%%  [%+.2f,%+.2f] %9.4f   %s\n", i, tr.len, tr.score,
                        tr.longest_no_food, tr.loop_ratio * 100.0,
                        tr.q_min, tr.q_max, tr.q_mean_margin,
                        tr.end == snake::StepResult::End::Starve ? "饿死(绕圈)" : "撞死");
        }

        std::printf("\n合计：%d 局，平均步数 %.1f，平均得分 %.2f\n",
                    episodes, static_cast<double>(total_steps) / episodes,
                    static_cast<double>(total_score) / episodes);
        std::printf("绕圈退化（短周期步数>50%%）的 %d 局，饿死 %d 局，撞死 %d 局\n",
                    looped, starved, crashed);
    }

    void render_greedy_game(dqn::Agent &agent, snake::Game &game, int delay_ms)
    {
        game.reset();
        snake::State s = game.state();

        std::vector<char> buf(4096);
        while (true)
        {
            const int action = agent.act(s, /*greedy=*/true);
            const snake::StepResult r = game.step(static_cast<snake::Action>(action));
            s = game.state();

            snake::render(game, buf.data(), buf.size());
            std::printf("\x1b[H\x1b[2J%s得分: %d  步数: %d\n", buf.data(), game.score(), game.steps());
            std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

            if (r.done)
                break;
        }
        std::printf("本局结束：得分 %d，共 %d 步\n", game.score(), game.steps());
    }
} // namespace

int main(int argc, char **argv)
{
    const Options opt = parse(argc, argv);

    dqn::Config cfg;
    cfg.hidden = opt.hidden;
    cfg.device = opt.device;

    dqn::Agent agent(cfg, opt.seed);
    snake::Game game(12, 12, opt.seed);

    std::printf("=== DQN 贪吃蛇 ===\n");
    std::printf("设备 %s，隐藏层 %d，状态 %d 维，动作 %d 个\n",
                (opt.device == mcn::Device::CUDA ? "CUDA" : "CPU"),
                cfg.hidden, snake::kStateDim, snake::kNumActions);
    std::printf("参数总量 %zu\n\n", agent.online().num_parameters());

    const auto t0 = std::chrono::steady_clock::now();

    double window_score = 0.0;
    double window_loss = 0.0;
    int window_loss_n = 0;
    int window_games = 0;

    for (int ep = 1; ep <= opt.episodes; ++ep)
    {
        double ep_loss = 0.0;
        int ep_loss_n = 0;
        const int score = run_episode(agent, game, /*greedy=*/false, &ep_loss, &ep_loss_n);

        window_score += score;
        window_loss += ep_loss;
        window_loss_n += ep_loss_n;
        ++window_games;

        if (window_games == 25 || ep == opt.episodes)
        {
            const auto now = std::chrono::steady_clock::now();
            const double secs = std::chrono::duration<double>(now - t0).count();
            std::printf("局 %4d | 近 %2d 局均分 %5.2f | eps %.3f | loss %.5f | 回放 %zu | %.1fs\n",
                        ep, window_games, window_score / window_games,
                        agent.epsilon(),
                        window_loss_n ? window_loss / window_loss_n : 0.0,
                        agent.replay_size(), secs);
            std::fflush(stdout);

            window_score = 0.0;
            window_loss = 0.0;
            window_loss_n = 0;
            window_games = 0;
        }
    }

    // 贪心评估：此时不再探索，看真实水平
    std::printf("\n=== 贪心评估（%d 局）===\n", opt.eval_episodes);
    double total = 0.0;
    int best = 0;
    for (int i = 0; i < opt.eval_episodes; ++i)
    {
        double unused_loss = 0.0;
        int unused_n = 0;
        const int score = run_episode(agent, game, /*greedy=*/true, &unused_loss, &unused_n);
        total += score;
        if (score > best)
            best = score;
    }
    std::printf("平均得分 %.2f，最高 %d\n", total / opt.eval_episodes, best);

    if (opt.probe)
        run_probe(agent, game, opt.eval_episodes);

    if (opt.render)
    {
        std::puts("\n=== 演示 ===");
        render_greedy_game(agent, game, 60);
    }

    return 0;
}
