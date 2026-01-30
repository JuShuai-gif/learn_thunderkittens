/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rv_layout.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// 这是一个辅助命名空间，用于定义与向量相关的概念和类型。
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief 定义了寄存器向量的概念和抽象类型的命名空间。
 */
namespace crv {
/**
 * @brief 一个虚拟类型，用于标识寄存器向量类型。
 * 
 * 如果一个类型符合"像"寄存器向量（rv）的特性，它应该将其标识符定义为 ducks::rv::identifier。
 * 只要一个类型具备 ducks::rv::identifier 标识符，它就会在编译器检查中被视作一个寄存器向量（rv）。
 */
struct identifier {};
/**
 * @brief 寄存器向量的概念定义
 * 
 * @tparam T 需要检查的类型
 *
 * 要求：
 * - T 必须包含一个嵌套类型 identifier，该类型必须与 ducks::rv::identifier 相同。
 */
template<typename T>
concept all = requires {
    typename T::identifier; // 检查 T 是否具有 identifier 类型
} && std::is_same_v<typename T::identifier, identifier>; // 检查 T::identifier 是否等于 ducks::rv::identifier。

/**
 * @brief 检查类型 T 是否使用 naive 布局（无对齐优化）
 */
template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
/**
 * @brief 检查类型 T 是否使用 align 布局（带对齐优化）
 */
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
/**
 * @brief 检查类型 T 是否使用 ortho 布局（正交对齐布局）
 */
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
/**
 * @brief 检查类型 T 是否使用 tile 布局（即对齐或正交布局）
 */
template<typename T> concept tile_layout  = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
}
}
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */

template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct crv {
    using identifier = ducks::crv::identifier;
    using component  = rv<_T, _length, _layout>; /// Data type of each internal tile.
    using layout     = component::layout; ///< Layout of the matrix tile, ensures compatibility with the rv concepts
    
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};


template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_fl = crv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_bf = crv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using crv_hf = crv<half,  _l, layout>;

} // namespace kittens