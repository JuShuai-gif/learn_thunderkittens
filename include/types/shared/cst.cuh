#pragma once

#include "st.cuh"

namespace kittens {

namespace ducks {
namespace cst {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * 用于标识复杂寄存器瓦片的虚拟类型。此类型是为了在编译时检查一个类型是否符合复杂寄存器瓦片的标识符。
 * 
 * 对于一个类型，如果它定义了 `ducks::st::cmplx_identifier` 作为其标识符，那么它就会被编译器视为一个复杂寄存器瓦片（complex register tile）。
 */
struct identifier {};

/**
 * @brief Concept for shared tiles that are complex.
 * 
 * 用于定义复杂共享瓦片的概念。
 * 
 * @tparam T The type to check against the concept requirements.
 * 
 * 对类型 `T` 进行检查，确保它符合以下要求：
 * - 类型 `T` 是一个共享瓦片（shared tile）。
 * - 类型 `T` 必须具有复杂瓦片标识符。
 * - 类型 `T` 的组件（component）满足 `ducks::st::all` 概念。
 */
template <typename T> concept all = requires {
    typename T::identifier;// 检查 T 是否定义了 identifier
} && std::is_same_v<typename T::identifier, identifier> // 确保 T::identifier 与 ducks::st::cmplx_identifier 相同
&& ducks::st::all<typename T::component>;// 确保 T 的组件类型满足 ducks::st::all 概念要求

} // namespace st
} // namespace ducks

/**
 * @brief Complex tile structure
 * 
 * 复杂瓦片结构体。用于表示一个复杂数的瓦片，通常由实部和虚部的共享瓦片组成。
 *
 * @tparam _T The packed data type used for the matrix elements.
 *        矩阵元素的数据类型（通常是打包数据类型）。
 * @tparam _rows The height of the tile in terms of the number of subtiles.
 *        瓦片的高度，以子瓦片数为单位。
 * @tparam _cols The width of the tile in terms of the number of subtiles.
 *        瓦片的宽度，以子瓦片数为单位。
 * @tparam _layout The layout of the internal register tiles
 *        内部寄存器瓦片的布局。
 * 
 * 这个结构体用于在实部和虚部共享瓦片的内部抽象复杂数操作。
 * 
 */
template<typename _T, int _rows, int _cols>
struct cst {
    using identifier = ducks::cst::identifier;///< 复杂瓦片的类型标识符，便于类型区分。
    using component  = st<_T, _rows, _cols>; ///< 每个内部瓦片的组件数据类型，表示复数的实部或虚部。
    using T          = component::T;///< 组件的基础数据类型。
    using T2         = component::T2;///< 组件的打包数据类型。
    using dtype      = component::dtype; ///< 内部瓦片元素的数据类型。

    static constexpr int rows       = component::rows;///< 组件的行数（瓦片的高度）。
    static constexpr int cols       = component::cols;///< 组件的列数（瓦片的宽度）。

    // Real/imag tiles have same internal layout and size
    // 实部和虚部的瓦片具有相同的内部布局和大小。
    component real;///< 实部瓦片，表示复数的实数部分。
    component imag;///< 虚部瓦片，表示复数的虚数部分。

    // 定义向量类型，用于表示每一列或每一行的数据。
    using col_vec = csv<dtype, rows>;///< 列向量类型，表示每一列的元素。
    using row_vec = csv<dtype, cols>;///< 行向量类型，表示每一行的元素。
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
/**
 * @brief Wrapper for complex shared tiles with `bf16` type.
 * 
 * 为 `bf16` 类型的复杂共享瓦片提供包装。
 */
template<int _rows, int _cols> using cst_bf = cst<bf16,  _rows, _cols>;
/**
 * @brief Wrapper for complex shared tiles with `half` type.
 * 
 * 为 `half` 类型的复杂共享瓦片提供包装。
 */
template<int _rows, int _cols> using cst_hf = cst<half,  _rows, _cols>;
/**
 * @brief Wrapper for complex shared tiles with `float` type.
 * 
 * 为 `float` 类型的复杂共享瓦片提供包装。
 */
template<int _rows, int _cols> using cst_fl = cst<float, _rows, _cols>;

}























