#pragma once

#include <cstdint>

namespace mcn
{
    enum class DataType : int8_t
    {
        FLOAT32,
        FLOAT64,
        INT64,
        INT32,
        INT16,
        INT8,
        BOOL
    };

    template <typename T> struct dtype_of;

    template <> struct dtype_of<float>   { static constexpr DataType value = DataType::FLOAT32; };
    template <> struct dtype_of<double>  { static constexpr DataType value = DataType::FLOAT64; };
    template <> struct dtype_of<int64_t> { static constexpr DataType value = DataType::INT64; };
    template <> struct dtype_of<int32_t> { static constexpr DataType value = DataType::INT32; };
    template <> struct dtype_of<int16_t> { static constexpr DataType value = DataType::INT16; };
    template <> struct dtype_of<int8_t>  { static constexpr DataType value = DataType::INT8; };
    template <> struct dtype_of<bool>    { static constexpr DataType value = DataType::BOOL; };

    template <typename T>
    inline constexpr DataType dtype_of_v = dtype_of<T>::value;

    [[nodiscard("result of itemsize() is unused")]]
    inline constexpr size_t itemsize(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::FLOAT64:
            return 8;
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
            return 8;
        case DataType::INT32:
            return 4;
        case DataType::INT16:
            return 2;
        case DataType::INT8:
            return 1;
        case DataType::BOOL:
            return 1;
        default:
            return 0;
        }
    }

    [[nodiscard("result of to_string() is unused")]]
    inline constexpr const char *dtype_to_string(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::FLOAT64:
            return "float64";
        case DataType::FLOAT32:
            return "float32";
        case DataType::INT64:
            return "int64";
        case DataType::INT32:
            return "int32";
        case DataType::INT16:
            return "int16";
        case DataType::INT8:
            return "int8";
        case DataType::BOOL:
            return "bool";
        default:
            return "unknown";
        }
    }

    [[nodiscard("result of is_floating_point() is unused")]]
    inline constexpr bool is_floating_point(DataType dtype)
    {
        return dtype == DataType::FLOAT32 ||
               dtype == DataType::FLOAT64;
    }
} // namespace mcn