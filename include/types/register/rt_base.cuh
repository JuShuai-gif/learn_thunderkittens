/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */
 
#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rv_layout.cuh"

namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

namespace ducks {
/**
 * @namespace rt_base
 * 
 * @brief 定义了寄存器基准瓷砖（16x16）相关的概念和抽象类型。
 */
namespace rt_base {
/**
 * @brief 一个虚拟类型，用于标识寄存器基础瓷砖。
 * 
 * 如果一个类型符合"像"寄存器基础瓷砖（rt_base）的特性，它应该将其标识符定义为 ducks::rt_base::identifier。
 * 只要一个类型具备 ducks::rt_base::identifier 标识符，它就会在编译器检查中被视作一个寄存器基础瓷砖（rt_base）。
 */
struct identifier {};
}
} // namespace ducks


/**
 * @brief 基础瓷砖结构体，用于寄存器中的计算。
 *
 * @tparam _T 元素的打包数据类型
 * @tparam _layout 瓷砖的布局（行优先或列优先）
 *
 * 这个结构体主要用于基于 PTX 原语构建更大的内联模板并管理布局。
 * 
 * 通常，你可能希望使用行优先布局，除非你特别想调用 mma（矩阵乘法指令）。
 */
template<typename _T, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< 标识寄存器基础瓷砖类型的标识符
    using layout = _layout; ///< 瓷砖的布局类型（行优先或列优先）
    static_assert(kittens::ducks::base_types::T1<_T>); // 确保 _T 是支持的类型
    using T = kittens::base_types::packing<_T>::unpacked_type;// 提取未打包的元素类型
    using T2 = kittens::base_types::packing<_T>::packed_type;// 提取已打包的元素类型
    using dtype = T2; ///< 瓷砖元素的数据类型
    
    // 类型检查，确保 dtype 是支持的类型（例如 bf16_2, float2, half_2 等）
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2> || std::is_same_v<dtype, fp8e4m3_4> || std::is_same_v<dtype, fp8e5m2_4>
#if defined(DF_BLACKWELL)
        || std::is_same_v<dtype, fp8e8m0_4> || std::is_same_v<dtype, fp4e2m1_4>
#endif
        ,
        "rt_base was provided an unsupported type."
    );
#else
    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2>,
        "rt_base was provided an unsupported type."
    );
#endif
    // 瓷砖的行和列大小
    static constexpr int tile_size_row        = kittens::TILE_ROW_DIM<T>; // < Tile size is a constant 16 for everyone
    static constexpr int tile_size_col        = kittens::TILE_COL_DIM<T>;
    static constexpr int rows                 = tile_size_row; ///< Number of rows.
    static constexpr int cols                 = tile_size_col; ///< Number of cols.
    static constexpr int num_elements         = rows*cols; // 瓷砖中元素的总数（256，或者对于 fp8e4m3 为 64）
    static constexpr int elements_per_thread  = num_elements / 32; // 每个线程处理的元素数（8，对于 fp8e4m3 为 2）
    
    // 每个线程处理的打包数据数量
    static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num()) ; // 4
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // 4 或 8，寄存器大小为32位字
    
    // 行和列向量布局，取决于是否是行优先布局
    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // 用于存储列求和
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // 用于存储行求和

    dtype data[packed_per_thread]; ///< 实际的基础瓷砖存储
};

// rt_base 是 fp8e4m3 时元素数量为 2x 基础瓷砖的元素数
// 当我们将 16x16 的 float2 转换为瓷砖时，瓷砖中有 512 个元素
// 而当使用 fp8e4m3x4 的打包类型时，瓷砖中有 16x32x4=2048 个元素

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt_base {
/**
 * @brief 所有寄存器基础瓷砖的概念定义
 * @tparam T 需要检查的类型
 *
 * 要求：
 * - T 必须包含一个嵌套类型 identifier，该类型必须与 rt_base::identifier 相同。
 */
template<typename T> concept all = requires {
    typename T::identifier; // 检查 T 是否具有 identifier 类型
} && std::is_same_v<typename T::identifier, identifier>; // 检查 T::identifier 是否等于 ducks::rt_base::identifier
} // namespace rt
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
// 为不同数据类型提供的别名
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;   // float 类型的基础瓷砖
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;    // bf16 类型的基础瓷砖
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;    // half 类型的基础瓷砖
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e4m3 = rt_base<fp8e4m3, L>;    // fp8e4m3 类型的基础瓷砖
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e5m2 = rt_base<fp8e5m2, L>;    // fp8e5m2 类型的基础瓷砖
#endif
#ifdef DF_BLACKWELL
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp8e8m0 = rt_base<fp8e8m0, L>;// fp8e8m0 类型的基础瓷砖
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fp4e2m1_2 = rt_base<fp4e2m1_2, L>;// fp4e2m1_2 类型的基础瓷砖
#endif
}