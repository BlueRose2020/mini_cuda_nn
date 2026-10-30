#pragma once

#include <cstdio>
#include <cstdlib>

#include "mini_cuda_nn/core/dtype.h"

// 实现细节，不属于公开 API：只在 .cpp / .cu 里把 DataType 派发成模板参数。
// 用户不需要知道它存在。

namespace mcn::detail
{
    template <typename Fn>
    void dispatch_dtype(DataType dtype, Fn &&fn)
    {
        switch (dtype)
        {
        case DataType::FLOAT32: fn.template operator()<float>();   break;
        case DataType::FLOAT64: fn.template operator()<double>();  break;
        case DataType::INT64:   fn.template operator()<int64_t>(); break;
        case DataType::INT32:   fn.template operator()<int32_t>(); break;
        case DataType::INT16:   fn.template operator()<int16_t>(); break;
        case DataType::INT8:    fn.template operator()<int8_t>();  break;
        case DataType::BOOL:    fn.template operator()<bool>();    break;
        default:
            std::fprintf(stderr, "dispatch_dtype: unsupported dtype %d at %s:%d\n",
                         static_cast<int>(dtype), __FILE__, __LINE__);
            std::abort();
        }
    }
} // namespace mcn::detail
