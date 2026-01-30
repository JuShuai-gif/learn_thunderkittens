/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "rt.cuh"
#include "crv.cuh"

namespace kittens {

namespace ducks {
namespace crt {
/**
 * @brief A dummy type used to identify complex register tiles.
 * * @brief 一个用于标识“复数寄存器块”的占位类型。
 * * For a type to quack like an rt_cmplx, it should define its identifier as ducks::rt::cmplx_identifier.
 * If a type quacks like ducks::rt::cmplx_identifier, it will be treated as an rt_cmplx by compiler checks.
 * 为了让一个类型被识别为复数寄存器块，它必须定义 identifier 为 ducks::crt::identifier。
 */
struct identifier {};
/**
* @brief Concept for register tiles that are complex.
* @brief 复数寄存器块的概念约束（Concept）。
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile. (T 必须是一个寄存器块)
* - T has a complex tile identifier. (T 必须拥有复数块标识符)
*/
template <typename T> concept all = requires {
    typename T::identifier; // 检查 T 是否定义了 identifier 类型
} && 
// 检查 identifier 是否正是上面定义的 struct identifier
std::is_same_v<typename T::identifier, identifier> && 
// 检查 T 的基础组件 (component) 是否符合普通寄存器块 (ducks::rt::all) 的概念
ducks::rt::all<typename T::component>;

/*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::row.
* * 概念约束：行优先布局的复数寄存器块。
* 要求 T 符合 'all' 概念，且其布局 (layout) 为行优先 (row)。
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;

/**
* @brief Concept for register tiles with col layout.
* @brief 概念约束：列优先布局的复数寄存器块。
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::col.
* 要求 T 符合 'all' 概念，且其布局 (layout) 为列优先 (col)。
*/
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt
} // namespace ducks

/**
 * @brief Complex tile structure
 * @brief 复数寄存器块结构体定义
 *
 * @tparam T2 The packed data type used for the matrix elements. (用于矩阵元素的打包数据类型)
 * @tparam _rows The height of the tile in terms of the number of subtiles. (块的高度，以子块数量为单位)
 * @tparam _cols The width of the tile in terms of the number of subtiles. (块的宽度，以子块数量为单位)
 * @tparam _layout The layout of the internal register tiles, either row-major or column-major. (内部寄存器块的布局，行优先或列优先)
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * register tiles, respectively
 * 该结构体旨在将复数运算抽象为分别对实部和虚部寄存器块的操作。
 * * In general, you probably want a row-major tile, unless you specifically want to call mma
 * 通常情况下，你应该使用行优先 (row-major) 的块，除非你明确需要调用矩阵乘加指令 (MMA)。
 */
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct crt {
    // 标识符，用于通过上面定义的 concept 检查
    using identifier = ducks::crt::identifier;
    // 核心组件类型：定义了实部和虚部各自的数据类型。
    // rt 是普通寄存器块 (Register Tile)，这里复用了 rt 的定义。
    using component  = rt<_T, _rows, _cols, _layout>; /// Data type of each internal tile.
    // 布局信息，与组件保持一致，确保兼容性
    using layout     = component::layout; ///< Layout of the matrix tile, ensures compatibility with the rt concepts
    // 基础数据类型导出 (例如 float, half 等)
    using T          = component::T;
    using T2         = component::T2;   // 打包类型 (packed type)
    using dtype      = component::dtype; ///< Data type of the elements in the tile.
    
    // 维度常量导出 (直接使用组件的维度)
    static constexpr int rows       = component::rows;
    static constexpr int cols       = component::cols;
    static constexpr int height     = component::height;
    static constexpr int width      = component::width;

    // Real/imag tiles have same internal layout and size
    // 核心数据成员：实部和虚部。
    // 这里并没有使用 std::complex<T> 那种交错存储，而是使用了 Planar 格式（实部一块，虚部一块）。
    component real;
    component imag;
    
    // 定义与此块对应的行向量和列向量类型 (crv 可能代表 Complex Register Vector)
    using row_vec = crv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = crv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
};

// 常用类型的别名定义，方便用户直接使用
// 浮点数 (float) 类型的复数块
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_fl = crt<float, _rows, _cols, layout>;
// BFloat16 类型的复数块
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_bf = crt<bf16, _rows, _cols, layout>;
// Half (FP16) 类型的复数块
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_hf = crt<half, _rows, _cols, layout>;



}