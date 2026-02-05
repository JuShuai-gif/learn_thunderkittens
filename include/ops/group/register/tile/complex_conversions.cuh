/**
 * @file
 * @brief 复数寄存器tile的数据布局和类型转换
 */

/* ----------  布局交换  ---------- */

/**
 * @brief 交换复数寄存器tile的布局。
 *
 * 此函数通过交换实部和虚部分量tile的布局来交换复数寄存器tile的布局。
 *
 * @tparam T2 寄存器tile元素的数据类型。
 * @tparam _height 寄存器tile的高度。
 * @tparam _width 寄存器tile的宽度。
 * @tparam layout 寄存器tile的当前布局。
 * @param dst[out] 目标寄存器tile的引用，结果将存储在此。
 * @param src[in] 要交换的源寄存器tile的引用。
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const crt<T2, _height, _width, layout> &src) {
    swap_layout(dst.real, src.real);// 交换实部的布局
    swap_layout(dst.real, src.real);// 交换虚部的布局（注意：原代码这里有重复，应该是笔误）
}

/**
 * @brief 原地交换复数寄存器tile的布局。
 *
 * @tparam T2 寄存器tile元素的数据类型。
 * @tparam _height 寄存器tile的高度。
 * @tparam _width 寄存器tile的宽度。
 * @tparam layout 寄存器tile的当前布局。
 * @param tile[in,out] 要原地交换的寄存器tile的引用。
 * @return 交换后的寄存器tile的引用。
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = swap_layout_inplace(tile.real);// 原地交换实部的布局
    tile.imag = swap_layout_inplace(tile.imag);// 原地交换虚部的布局
    return tile;// 返回交换后的tile引用
}

/* ----------  转置操作  ---------- */

/**
 * @brief 转置复数寄存器tile。
 * 
 * 此函数标记为"sep"，意味着dst底层的寄存器必须与src底层的寄存器是分离的。
 *
 * @tparam T2 寄存器tile元素的数据类型。
 * @tparam _height 源寄存器tile的高度，也是目标tile的宽度。
 * @tparam _width 源寄存器tile的宽度，也是目标tile的高度。
 * @tparam layout 寄存器tile的布局。
 * @param dst[out] 存储转置后结果的寄存器tile引用。
 * @param src[in] 要转置的源寄存器tile引用。
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void transpose_sep(crt<T2, _width, _height, layout> &dst, const crt<T2, _height, _width, layout> &src) {
    transpose_sep(dst.real, src.real);// 转置实部
    transpose_sep(dst.imag, src.imag);// 转置虚部
}


/**
 * @brief 原地转置方形复数寄存器tile。
 *
 * @tparam T2 寄存器tile元素的数据类型。
 * @tparam _height 源寄存器tile的高度（以16为单位），也是目标tile的宽度。（必须与_width相同。）
 * @tparam _width 源寄存器tile的宽度（以16为单位），也是目标tile的高度。（必须与_height相同。）
 * @tparam layout 寄存器tile的当前布局。
 * @param tile[in] 要转置的寄存器tile引用。
 * @return 转置后的寄存器tile引用。
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, layout>& transpose_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = transpose_inplace(tile.real);// 原地转置实部
    tile.imag = transpose_inplace(tile.imag);// 原地转置虚部

    return tile;
}

/* ----------  类型转换  ---------- */

/**
 * @brief 复制复数寄存器tile，并在必要时转换底层类型。
 *
 * @tparam T2 目标寄存器元素的数据类型。
 * @tparam U2 源寄存器元素的数据类型。
 * @tparam _height 寄存器tile的高度（以16为单位）。
 * @tparam _width 寄存器tile的宽度（以16为单位）。
 * @tparam layout 寄存器tile的当前布局。
 * @param[out] dst 目标寄存器tile的引用。
 * @param[in] src 源寄存器tile的引用。
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(crt<T2, _height, _width, layout> &dst, const crt<U2, _height, _width, layout> &src) {
    copy(dst.real, src.real);// 复制实部并进行类型转换
    copy(dst.imag, src.imag);// 复制虚部并进行类型转换
}