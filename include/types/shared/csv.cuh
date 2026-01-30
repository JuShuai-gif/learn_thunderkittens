
#pragma once

#include "st.cuh"

namespace kittens {

namespace ducks {
namespace csv {

/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * 用于标识复杂寄存器瓦片的虚拟类型。此类型在编译时检查类型是否符合复杂寄存器瓦片标识符。
 * 
 * 对于一个类型，如果它定义了 `ducks::st::cmplx_identifier` 作为其标识符，那么它将被视为一个复杂寄存器瓦片。
 */
struct identifier {};
/**
 * @brief Concept for shared vectors that are complex.
 * 
 * 用于定义复杂共享向量的概念。
 * 
 * @tparam T The type to check against the concept requirements.
 * 
 * 对类型 `T` 进行检查，确保它符合以下要求：
 * - 类型 `T` 是一个共享向量。
 * - 类型 `T` 必须具有复杂瓦片标识符。
 * - 类型 `T` 的组件满足 `ducks::sv::all` 概念。
 */
template <typename T> concept all = requires {
    typename T::identifier; // 检查 T 是否定义了 identifier
} && std::is_same_v<typename T::identifier, identifier> // 确保 T::identifier 与 ducks::csv::identifier 相同
&& ducks::sv::all<typename T::component>;   // 确保 T 的组件类型满足 ducks::sv::all 概念要求

} // namespace st
} // namespace ducks

/**
 * @brief Complex tile structure
 * 
 * 复杂瓦片结构体，用于表示一个复数的向量。通常由实部和虚部的共享向量组成。
 * 
 * @tparam _T The packed data type used for the matrix elements.
 *        矩阵元素的数据类型（通常是打包数据类型）。
 * @tparam _length The length of the vector, in terms of the number of subtiles.
 *        向量的长度，以子瓦片数为单位。
 *
 * 该结构体用于在实部和虚部的共享向量内抽象化复数的向量运算。
 * 
 */
template<typename _T, int _length>
struct csv {
    using identifier = ducks::csv::identifier;///< 复杂寄存器瓦片的类型标识符，便于区分复杂瓦片类型。
    using component  = sv<_T, _length>; ///< 每个内部瓦片的组件数据类型，表示复数的实部或虚部。
    using T          = component::T;///< 组件的基础数据类型。
    using T2         = component::T2; ///< 组件的打包数据类型。
    using dtype      = component::dtype; ///< 内部瓦片元素的数据类型。

    static constexpr int length     = component::length; ///< 向量的长度（组件的长度）。
    static constexpr int tiles      = component::tiles;///< 向量的瓦片数目（组件的瓦片数目）。

    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.

    // 实部和虚部的瓦片具有相同的内部布局和大小
    component real; ///< 实部向量，表示复数的实数部分。
    component imag; ///< 虚部向量，表示复数的虚数部分。
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
/**
 * @brief Wrapper for complex shared vectors with `bf16` type.
 * 
 * 为 `bf16` 类型的复杂共享向量提供包装。
 */
template<int _length> using csv_bf = csv<bf16,  _length>;

/**
 * @brief Wrapper for complex shared vectors with `half` type.
 * 
 * 为 `half` 类型的复杂共享向量提供包装。
 */
template<int _length> using csv_hf = csv<half,  _length>;

/**
 * @brief Wrapper for complex shared vectors with `float` type.
 * 
 * 为 `float` 类型的复杂共享向量提供包装。
 */
template<int _length> using csv_fl = csv<float, _length>;

}















