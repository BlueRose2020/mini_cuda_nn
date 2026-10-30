#include "snake.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace snake
{
    namespace
    {
        // 正下方是"绕圈"检测的阈值：这么多步没吃到东西就判负，
        // 否则智能体会学出"永远绕圈不死"的退化解
        int starve_limit(int w, int h) { return w * h * 2; }
    } // namespace

    Game::Game(int width, int height, unsigned seed)
        : w_(width), h_(height), rng_(seed)
    {
        reset();
    }

    void Game::reset()
    {
        body_.clear();
        const int cx = w_ / 2;
        const int cy = h_ / 2;
        // 朝右，长度 3
        body_.push_back({cx, cy});
        body_.push_back({cx - 1, cy});
        body_.push_back({cx - 2, cy});

        dir_ = Dir::Right;
        alive_ = true;
        steps_ = 0;
        since_food_ = 0;
        score_ = 0;
        spawn_food();
    }

    Point Game::next_cell(const Point &p, int dir) const
    {
        switch (dir)
        {
        case Dir::Up:
            return {p.x, p.y + 1};
        case Dir::Right:
            return {p.x + 1, p.y};
        case Dir::Down:
            return {p.x, p.y - 1};
        default:
            return {p.x - 1, p.y};
        }
    }

    bool Game::out_of_bounds(const Point &p) const
    {
        return p.x < 0 || p.y < 0 || p.x >= w_ || p.y >= h_;
    }

    // ignore_tail：蛇尾这一格在下一步会让出来，所以"撞尾"其实不算死
    bool Game::hit_body(const Point &p, bool ignore_tail) const
    {
        const size_t n = body_.size();
        for (size_t i = 0; i + (ignore_tail ? 1 : 0) < n; ++i)
            if (body_[i] == p)
                return true;
        return false;
    }

    void Game::spawn_food()
    {
        const int total = w_ * h_;
        if (static_cast<int>(body_.size()) >= total)
            return; // 棋盘满了，没有空格

        // 随机试有限次，失败就线性扫描。两个分支都保证不会落在蛇身上。
        std::uniform_int_distribution<int> px(0, w_ - 1);
        std::uniform_int_distribution<int> py(0, h_ - 1);

        for (int attempt = 0; attempt < 200; ++attempt)
        {
            const Point p{px(rng_), py(rng_)};
            if (!hit_body(p, /*ignore_tail=*/false))
            {
                food_ = p;
                return;
            }
        }

        for (int y = 0; y < h_; ++y)
            for (int x = 0; x < w_; ++x)
                if (!hit_body({x, y}, false))
                {
                    food_ = {x, y};
                    return;
                }
    }

    StepResult Game::step(Action action)
    {
        StepResult r;
        if (!alive_)
        {
            r.done = true;
            return r;
        }

        // 动作相对朝向：Right 顺时针 90°，Left 逆时针 90°
        const int turn = (action == Action::Right) ? 1 : (action == Action::Left ? 3 : 0);
        dir_ = (dir_ + turn) % 4;

        const Point head = next_cell(body_.front(), dir_);
        ++steps_;
        ++since_food_;

        const bool grows = (head == food_);

        // 势函数塑形：r' = r + gamma*phi(s') - phi(s)。
        // 没有这一项时，“向食物走一步”与“原地打转”的即时回报完全一样（都是 0），
        // 稀疏奖励下智能体会停在原地绕圈。
        const double phi0 = potential(body_.front());

        // 尾巴这一步会让开，只有吃到食物变长时才真的会撞到自己
        if (out_of_bounds(head) || hit_body(head, /*ignore_tail=*/!grows))
        {
            alive_ = false;
            r.done = true;
            r.end = StepResult::End::Crash;
            r.reward = -1.0; // 终结状态不计塑形
            return r;
        }

        body_.push_front(head);

        if (grows)
        {
            ++score_;
            since_food_ = 0;
            r.ate = true;
            r.reward = 1.0;

            if (static_cast<int>(body_.size()) == w_ * h_)
            {
                // 铺满棋盘，胜利
                alive_ = false;
                r.done = true;
                r.end = StepResult::End::Win;
                return r;
            }
            spawn_food();
        }
        else
        {
            body_.pop_back();
        }

        // 非终结转移：加上势差（食物重生后 phi 按新食物算）
        r.reward += kShapingGamma * potential(body_.front()) - phi0;

        if (since_food_ > starve_limit(w_, h_))
        {
            alive_ = false;
            r.done = true;
            r.end = StepResult::End::Starve;
            // 与撞死同价：否则“安静地绕圈到死”比“冒险去吃”更划算，
            // 智能体会稳定收敛到绕圈的退化解（实测 100% 的贪心对局都如此）。
            r.reward = -1.0;
        }
        return r;
    }

    double Game::potential(const Point &head) const
    {
        const double dist = std::abs(food_.x - head.x) + std::abs(food_.y - head.y);
        return -dist / static_cast<double>(w_ + h_);
    }

    State Game::state() const
    {
        State s{};
        const Point head = body_.front();

        auto blocked = [&](const Point &c)
        {
            return out_of_bounds(c) || hit_body(c, /*ignore_tail=*/true);
        };

        auto danger = [&](int dir) { return blocked(next_cell(head, dir)) ? 1.0f : 0.0f; };

        // 0-2：直行 / 右转 / 左转会撞吗
        s[0] = danger(dir_);
        s[1] = danger((dir_ + 1) % 4);
        s[2] = danger((dir_ + 3) % 4);

        // 3-6：当前朝向 one-hot
        s[3] = (dir_ == Dir::Left) ? 1.0f : 0.0f;
        s[4] = (dir_ == Dir::Right) ? 1.0f : 0.0f;
        s[5] = (dir_ == Dir::Up) ? 1.0f : 0.0f;
        s[6] = (dir_ == Dir::Down) ? 1.0f : 0.0f;

        // 7-10：食物在蛇头的哪一侧
        s[7] = (food_.x < head.x) ? 1.0f : 0.0f;
        s[8] = (food_.x > head.x) ? 1.0f : 0.0f;
        s[9] = (food_.y < head.y) ? 1.0f : 0.0f;
        s[10] = (food_.y > head.y) ? 1.0f : 0.0f;

        // 11-12：归一化位移，给网络一个连续的"还有多远"信号
        const double span = static_cast<double>(w_ + h_);
        s[11] = static_cast<float>((food_.x - head.x) / span);
        s[12] = static_cast<float>((food_.y - head.y) / span);

        // 13-15：三个方向的可达空格数（洪泛）。这是唯一的"前瞻"来源，
        // 前 11 维只见一步，智能体会因为看不到死胡同而反复绕小圈。
        auto reachable = [&](int dir)
        {
            const Point start = next_cell(head, dir);
            if (blocked(start))
                return 0.0f;
            return static_cast<float>(free_area(start)) /
                   static_cast<float>(w_ * h_);
        };
        s[13] = reachable(dir_);
        s[14] = reachable((dir_ + 1) % 4);
        s[15] = reachable((dir_ + 3) % 4);
        return s;
    }

    // 从 start 出发能走到多少个空格（洪泛，用于给网络前瞻信息）
    int Game::free_area(const Point &start) const
    {
        std::vector<char> seen(static_cast<size_t>(w_) * h_, 0);
        std::vector<Point> stack;
        stack.push_back(start);
        seen[static_cast<size_t>(start.y) * w_ + start.x] = 1;

        int count = 0;
        while (!stack.empty())
        {
            const Point p = stack.back();
            stack.pop_back();
            ++count;

            for (int d = 0; d < 4; ++d)
            {
                const Point c = next_cell(p, d);
                if (out_of_bounds(c))
                    continue;
                const size_t idx = static_cast<size_t>(c.y) * w_ + c.x;
                if (seen[idx])
                    continue;
                if (hit_body(c, /*ignore_tail=*/true))
                    continue;
                seen[idx] = 1;
                stack.push_back(c);
            }
        }
        return count;
    }

    void render(const Game &game, char *out, size_t out_size)
    {
        const int w = game.width();
        const int h = game.height();
        const size_t need = static_cast<size_t>((w + 3) * (h + 2) + (w + 3) + 1);
        if (out_size < need)
        {
            if (out_size > 0)
                out[0] = '\0';
            return;
        }

        std::vector<char> grid(static_cast<size_t>(w) * h, '.');
        for (size_t i = 0; i < game.body().size(); ++i)
        {
            const Point &p = game.body()[i];
            grid[static_cast<size_t>(p.y) * w + p.x] = (i == 0) ? '@' : 'o';
        }
        if (game.alive())
        {
            const Point f = game.food();
            grid[static_cast<size_t>(f.y) * w + f.x] = '*';
        }

        size_t k = 0;
        for (int y = h - 1; y >= 0; --y) // y 轴向上，所以从最后一行画起
        {
            out[k++] = '|';
            for (int x = 0; x < w; ++x)
                out[k++] = grid[static_cast<size_t>(y) * w + x];
            out[k++] = '|';
            out[k++] = '\n';
        }
        out[k++] = '+';
        for (int x = 0; x < w; ++x)
            out[k++] = '-';
        out[k++] = '+';
        out[k++] = '\n';
        out[k] = '\0';
    }
} // namespace snake
