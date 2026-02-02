/**
 * @file
 * @brief The ThunderKittens tensor memory tile struct.
 */

#pragma once

#include "../../common/common.cuh"

/* ----------  MAIN tt STRUCT  ---------- */

// 这些是用于类型推断的辅助结构体
namespace kittens {
// 最大张量行数和列数定义
constexpr int MAX_TENSOR_ROWS = 128;
constexpr int MAX_TENSOR_COLS = 512;

namespace ducks {
/**
 * @namespace tt
 * @brief 共享内存张量块(tile)的概念和抽象类型所在的命名空间
 */
namespace tt {
/**
 * @brief 用于标识张量内存的虚拟类型
 */
struct identifier {};

/**
 * @brief 所有tt张量块(tile)的概念约束
 * @tparam T 要检查是否符合概念要求的类型
 * 
 * 要求:
 * - T必须有一个嵌套类型identifier，且与tt::identifier相同
 */
template<typename T> concept all = requires {
    typename T::identifier; // 检查T::identifier是否存在
} && std::is_same_v<typename T::identifier, identifier>; // 检查T::identifier是否与ducks::tt::identifier相同

/**
 * @brief 半高张量块(tile)的概念约束
 * @tparam T 要检查的类型
 * 
 * 要求:
 * - T满足all概念
 * - T的行数为MAX_TENSOR_ROWS的一半
 */
template<typename T> concept half = all<T> && T::rows == MAX_TENSOR_ROWS / 2;

/**
 * @brief 全高张量块(tile)的概念约束
 * @tparam T 要检查的类型
 * 
 * 要求:
 * - T满足all概念
 * - T的行数为MAX_TENSOR_ROWS
 */
template<typename T> concept full = all<T> && T::rows == MAX_TENSOR_ROWS;
} // namespace tt
} // namespace ducks

/**
 * @class tt
 * @brief 共享内存张量块(tile)结构，支持各种数据类型和布局
 * 
 * @tparam _T 张量元素的数据类型（未打包的原始类型）
 * @tparam _rows 张量的行数（高度）
 * @tparam _cols 张量的列数（宽度）
 * 
 * 这个类提供了对共享内存中张量块的抽象，支持子块提取、地址计算等操作。
 * 它使用基地址加偏移的方式访问共享内存中的不同部分。
 */
template<typename _T, int _rows, int _cols>
struct tt {
    using identifier = ducks::tt::identifier; ///< 共享内存张量块的类型标识符
    using T = base_types::packing<_T>::unpacked_type;///< 解包后的数据类型
    using T2 = base_types::packing<_T>::packed_type;///< 打包后的数据类型
    using dtype = T; ///< 张量元素的数据类型（与T相同）
    // 静态常量：张量的行数和列数
    static constexpr int rows    = _rows;
    static constexpr int cols    = _cols;
    // 静态断言：确保张量尺寸在合理范围内
    static_assert(rows / (4 / sizeof(T)) <= MAX_TENSOR_ROWS, "Row dimension must be less than or equal to MAX_TENSOR_ROWS");
    static_assert(cols / (4 / sizeof(T)) <= MAX_TENSOR_COLS, "Column dimension must be less than or equal to MAX_TENSOR_COLS");
    static_assert(rows % kittens::BASE_TILE_DIM == 0, "Row dimension must be divisible by the 16");
    static_assert(cols % kittens::BASE_TILE_DIM == 0, "Column dimension must be divisible by the 16");
    // 共享内存基地址
    uint32_t addr;
    /**
     * @brief 默认构造函数，地址初始化为0
     */
    __device__ inline tt() : addr(0) {}
    /**
     * @brief 构造函数，使用指定地址初始化
     * @param addr 共享内存地址
     */
    __device__ inline tt(uint32_t addr) : addr(addr) {}


    /**
     * @brief 从当前张量块中提取子块
     * @tparam TT 子块类型（必须满足ducks::tt::all概念）
     * @param row_offset 行偏移量
     * @param col_offset 列偏移量
     * @return 新的张量块对象，表示提取的子块
     * 
     * 在调试模式下，会进行运行时边界检查。
     * 地址计算考虑了行偏移（<<16）和列偏移（需要根据数据类型调整）。
     */
    template<ducks::tt::all TT>  __device__ inline TT subtile(int row_offset, int col_offset) const {
#ifndef NDEBUG
        // 运行时边界检查
        if(row_offset < 0 || row_offset+TT::rows > rows || col_offset < 0 || col_offset+TT::cols > cols) {
            printf("Subtile out of bounds! full tile rows: %d, full tile cols: %d, subtile rows: %d, subtile cols: %d, row_offset: %d, col_offset: %d\n", rows, cols, TT::rows, TT::cols, row_offset, col_offset);
            asm volatile("trap;");// 触发陷阱，用于调试
        }
#endif
        // 计算子块地址：基地址 + 行偏移<<16 + 列偏移/(4/sizeof(T))
        // 行偏移<<16是因为每行在内存中占64KB的空间
        // 列偏移需要根据数据类型调整，因为不同类型占用不同字节数
        return TT(addr + (row_offset<<16) + col_offset/(4/(uint32_t)sizeof(T)));
    }

    /**
     * @brief 从当前张量块中提取子块（无行偏移版本）
     * @tparam TT 子块类型（必须满足ducks::tt::all概念）
     * @param col_offset 列偏移量
     * @return 新的张量块对象，表示提取的子块
     */
    template<ducks::tt::all TT>  __device__ inline TT subtile(int col_offset) const {
        // 计算子块地址：基地址 + 列偏移/(4/sizeof(T))
        return TT(addr + col_offset/(4/(uint32_t)sizeof(T)));
    }


    
    /**
     * @brief 获取张量块中特定块(chunk)的地址，支持转置操作
     * @tparam transpose 是否进行转置
     * @param chunk 块索引
     * @return 计算得到的内存地址
     * 
     * 这个函数用于Tensor Core的MMA（矩阵乘法累加）操作。
     * 根据数据类型和是否转置，计算正确的内存地址。
     */
    template<int transpose> __device__ inline uint32_t chunk_addr(int chunk) const {
        if constexpr (transpose) {
            // 转置模式下的地址计算
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>) {
                // 对于支持的数据类型：基地址 + (16 * chunk) << 16
                // 这意味着每16行作为一个块，每个块在内存中占1MB的空间
                return addr + ((16 * chunk) << 16);
            }
            else {
                // 对于不支持的数据类型，触发编译时错误
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
        else {
            // 非转置模式下的地址计算
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half>) {
                // 对于16位类型：基地址 + (16 * chunk) / (4/sizeof(T))
                // 即每16列作为一个块
                return addr + (16 * chunk / (4/(uint32_t)sizeof(T)));
            }
            else if constexpr (std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>) {
                // 对于8位类型：基地址 + (32 * chunk) / (4/sizeof(T))
                // 即每32列作为一个块，因为8位类型更紧凑
                return addr + (32 * chunk / (4/(uint32_t)sizeof(T)));
            }
            else {
                // 对于不支持的数据类型，触发编译时错误
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
    }
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
// 以下是一系列类型别名，用于创建各种数据类型和尺寸的张量块(tile)

// 各种数据类型的全尺寸张量块
template<int _height, int _width> using tt_bf = tt<bf16, _height, _width>;// bfloat16
template<int _height, int _width> using tt_hf = tt<half, _height, _width>;// half(FP16)
template<int _height, int _width> using tt_fl = tt<float, _height, _width>;// float(FP32)
template<int _height, int _width> using tt_fp8e4m3 = tt<fp8e4m3, _height, _width>;// FP8 E4M3
template<int _height, int _width> using tt_fp8e5m2 = tt<fp8e5m2, _height, _width>;// FP8 E5M2
template<int _height, int _width> using tt_fp8e8m0 = tt<fp8e8m0, _height, _width>;// FP8 E8M0
template<int _height, int _width> using tt_fp4e2m1_2 = tt<fp4e2m1_2, _height, _width>;// FP4 E2M1 (2个4位打包)

// 半高张量块（64行）
template<int _width> using half_tt_bf = tt<bf16, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_hf = tt<half, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_fl = tt<float, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_fp8e4m3 = tt<fp8e4m3, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_fp8e5m2 = tt<fp8e5m2, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_fp8e8m0 = tt<fp8e8m0, MAX_TENSOR_ROWS / 2, _width>;
template<int _width> using half_tt_fp4e2m1_2 = tt<fp4e2m1_2, MAX_TENSOR_ROWS / 2, _width>;

// 全高张量块（128行）
template<int _width> using full_tt_bf = tt<bf16, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_hf = tt<half, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_fl = tt<float, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_fp8e4m3 = tt<fp8e4m3, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_fp8e5m2 = tt<fp8e5m2, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_fp8e8m0 = tt<fp8e8m0, MAX_TENSOR_ROWS, _width>;
template<int _width> using full_tt_fp4e2m1_2 = tt<fp4e2m1_2, MAX_TENSOR_ROWS, _width>;

} // namespace kittens
