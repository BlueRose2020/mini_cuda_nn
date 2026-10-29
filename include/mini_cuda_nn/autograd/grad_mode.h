#pragma once

namespace mcn::detail
{
    inline bool g_grad_enabled = true;

    inline bool grad_enabled() noexcept { return g_grad_enabled; }

    class NoGradGuard
    {
    public:
        NoGradGuard() noexcept : prev_(g_grad_enabled) { g_grad_enabled = false; }
        ~NoGradGuard() noexcept { g_grad_enabled = prev_; }

        NoGradGuard(const NoGradGuard &) = delete;
        NoGradGuard &operator=(const NoGradGuard &) = delete;

    private:
        bool prev_;
    };
} // namespace mcn::detail
