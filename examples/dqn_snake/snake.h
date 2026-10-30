#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <random>

namespace snake
{
    // 状态维度：
    //   0- 2  直行/右转/左转的邻格是否危险
    //   3- 6  当前朝向 one-hot
    //   7-10  食物在蛇头的哪一侧
    //  11-12  到食物的归一化位移 dx, dy
    //  13-15  沿直行/右转/左转方向的**可达空格数**（归一化）
    //
    // 最后一项是关键：前 11 维只能看到紧邻的一格，智能体无法判断
    // “往这个方向走会不会把自己困死”。可达空格数是一次洪泛得到的前瞻信息。
    inline constexpr int kStateDim = 16;

    // 动作是相对于当前朝向的（贪吃蛇不能"向后转"，所以只有 3 个）
    enum class Action : int
    {
        Straight = 0,
        Right = 1,
        Left = 2
    };
    inline constexpr int kNumActions = 3;

    enum Dir : int
    {
        Up = 0,
        Right = 1,
        Down = 2,
        Left = 3
    };

    struct Point
    {
        int x = 0;
        int y = 0;
        bool operator==(const Point &) const = default;
    };

    struct StepResult
    {
        // 终止原因：塑形之后 reward 不再是固定值，不能靠数值反推怎么死的
        enum class End
        {
            None,
            Crash,  // 撞墙或撞到自己
            Starve, // 太久没吃到东西
            Win     // 铺满棋盘
        };

        bool done = false;
        bool ate = false;
        double reward = 0.0;
        End end = End::None;
    };

    using State = std::array<float, kStateDim>;

    // 奖励塑形使用的折扣因子，必须与 dqn::Config::gamma 一致，
    // 否则塑形不再是势函数形式，最优策略会被改变。
    inline constexpr double kShapingGamma = 0.9;

    // 网格贪吃蛇。坐标原点在左下角，y 轴向上。
    //
    // 奖励（势函数塑形后）：
    //   吃到食物        +1
    //   撞墙 / 撞自己    -1
    //   绕圈太久           -1（与撞死相同价格，否则“原地绕圈”比冒险吃食物更划算）
    //   其余为 0，但额外加上到食物距离变化的塑形项（见 kShapingGamma）
    class Game
    {
    public:
        Game(int width = 12, int height = 12, unsigned seed = 0);

        void reset();
        StepResult step(Action action);

        // 11 维手写特征：网络不需要卷积就能"看见"周围
        State state() const;

        int width() const { return w_; }
        int height() const { return h_; }
        int score() const { return score_; }
        int steps() const { return steps_; }
        bool alive() const { return alive_; }
        Dir direction() const { return static_cast<Dir>(dir_); }
        const std::deque<Point> &body() const { return body_; }
        Point food() const { return food_; }

    private:
        bool out_of_bounds(const Point &p) const;
        bool hit_body(const Point &p, bool ignore_tail) const;
        Point next_cell(const Point &p, int dir) const;
        int free_area(const Point &start) const;
        // 势函数：离食物越近越大。用于 reward shaping，不改变最优策略。
        double potential(const Point &head) const;
        void spawn_food();

        int w_;
        int h_;
        std::deque<Point> body_; // front 是蛇头
        Point food_{};
        int dir_ = Dir::Right;
        bool alive_ = true;
        int steps_ = 0;
        int since_food_ = 0;
        int score_ = 0;
        std::mt19937 rng_;
    };

    // 把棋盘画成文本（调试用）
    void render(const Game &game, char *out, size_t out_size);
} // namespace snake
