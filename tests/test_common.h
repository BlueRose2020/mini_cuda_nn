// 极简测试框架：多个测试文件共享。
// 全部 inline，可以被多个 TU 直接包含。
#pragma once

#include "mini_cuda_nn/core/tensor.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace mcn_test
{
    inline int g_checks = 0;
    inline int g_failures = 0;

    inline void fail(const char *file, int line, const std::string &msg)
    {
        std::printf("    [FAIL] %s:%d  %s\n", file, line, msg.c_str());
        ++g_failures;
    }

    inline void check_true(bool cond, const char *expr, const char *file, int line)
    {
        ++g_checks;
        if (!cond)
            fail(file, line, std::string("expected true: ") + expr);
    }

    inline void check_shape(const mcn::Tensor &t, std::vector<size_t> expect, const char *file, int line)
    {
        ++g_checks;
        if (t.shape() != expect)
            fail(file, line, "shape " + mcn::format_shape(t.shape()) +
                                 " != expected " + mcn::format_shape(expect));
    }

    inline void check_numel(const mcn::Tensor &t, size_t expect, const char *file, int line)
    {
        ++g_checks;
        if (t.numel() != expect)
            fail(file, line, "numel " + std::to_string(t.numel()) +
                                 " != expected " + std::to_string(expect));
    }

    inline void check_float(double got, double expect, double eps, const char *file, int line)
    {
        ++g_checks;
        const double diff = got > expect ? got - expect : expect - got;
        if (!(diff <= eps))
            fail(file, line, "value " + std::to_string(got) +
                                 " != expected " + std::to_string(expect));
    }

    inline void check_int(long long got, long long expect, const char *file, int line)
    {
        ++g_checks;
        if (got != expect)
            fail(file, line, "value " + std::to_string(got) +
                                 " != expected " + std::to_string(expect));
    }

    inline void check_dtype(const mcn::Tensor &t, mcn::DataType expect, const char *file, int line)
    {
        ++g_checks;
        if (t.dtype() != expect)
            fail(file, line, std::string("dtype ") + mcn::dtype_to_string(t.dtype()) +
                                 " != expected " + mcn::dtype_to_string(expect));
    }

    inline void check_device(const mcn::Tensor &t, mcn::Device expect, const char *file, int line)
    {
        ++g_checks;
        if (t.device() != expect)
            fail(file, line, "device mismatch");
    }

    // 按索引逐层取子张量，再从标量张量读值。
    // 只能用于 CPU 张量（设备内存不能直接解引用）。
    template <typename T>
    T value_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        mcn::Tensor cur = t; // 浅拷贝，operator[] 造视图，不会复制数据
        for (size_t i : idx)
            cur = cur[i];
        return *static_cast<const T *>(cur.data_ptr());
    }

    inline float f_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        return value_at<float>(t, idx);
    }

    inline int i_at(const mcn::Tensor &t, std::initializer_list<size_t> idx)
    {
        return value_at<int>(t, idx);
    }

    inline float grad_at(const mcn::Tensor &leaf, std::initializer_list<size_t> idx)
    {
        const mcn::Tensor *g = leaf.grad();
        check_true(g != nullptr, "leaf has no grad", __FILE__, __LINE__);
        if (!g)
            return 0.f;
        return f_at(*g, idx);
    }
} // namespace mcn_test

// 测试代码里直接写 f_at / grad_at 更顺手；这是测试专用的头文件，污染全局命名空间可以接受。
using mcn_test::f_at;
using mcn_test::grad_at;
using mcn_test::i_at;
using mcn_test::value_at;

#define CHECK_TRUE(expr) ::mcn_test::check_true((expr), #expr, __FILE__, __LINE__)
#define CHECK_SHAPE(t, ...) ::mcn_test::check_shape((t), std::vector<size_t>{__VA_ARGS__}, __FILE__, __LINE__)
#define CHECK_NUMEL(t, n) ::mcn_test::check_numel((t), (n), __FILE__, __LINE__)
#define CHECK_FLOAT(got, expect, eps) ::mcn_test::check_float((got), (expect), (eps), __FILE__, __LINE__)
#define CHECK_FLOAT_EQ(got, expect) ::mcn_test::check_float((got), (expect), 1e-6, __FILE__, __LINE__)
#define CHECK_FLOAT_NEAR(got, expect) ::mcn_test::check_float((got), (expect), 1e-5, __FILE__, __LINE__)
#define CHECK_INT(got, expect) ::mcn_test::check_int((got), (expect), __FILE__, __LINE__)
#define CHECK_DTYPE(t, d) ::mcn_test::check_dtype((t), (d), __FILE__, __LINE__)
#define CHECK_DEVICE(t, d) ::mcn_test::check_device((t), (d), __FILE__, __LINE__)
