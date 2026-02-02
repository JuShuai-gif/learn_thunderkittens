/**
 * @file
 * @brief 工作组（协作warp）操作 - 在全局内存和共享内存之间加载/存储共享tile
 */

/**
 * @brief 从全局内存加载数据到复数共享内存tile（可配置轴和对齐）
 *
 * @tparam axis 加载操作的轴方向（0=列方向，1=行方向，2=默认平面方向）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数共享内存tile
 * @param[in] src 源复数全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将复数数据的实部和虚部分别加载到共享内存tile中。
 * 通过axis参数可以控制加载的方向，assume_aligned参数可以启用对齐优化。
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    // 分别加载实部和虚部数据到共享内存tile
    // 使用相同的轴方向和对齐假设设置
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

/**
 * @brief 从全局内存加载数据到复数共享内存tile（默认设置版本）
 *
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数共享内存tile
 * @param[in] src 源复数全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是load函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐）
 * 直接调用实部和虚部的组件级加载函数。
 */
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

/**
 * @brief 从复数共享内存tile存储数据到全局内存（可配置轴和对齐）
 *
 * @tparam axis 存储操作的轴方向（0=列方向，1=行方向，2=默认平面方向）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数全局内存数组
 * @param[in] src 源复数共享内存tile
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将复数共享内存tile中的实部和虚部分别存储到全局内存中。
 * 通过axis参数可以控制存储的方向，assume_aligned参数可以启用对齐优化。
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    // 分别存储实部和虚部数据到全局内存
    // 使用相同的轴方向和对齐假设设置
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

/**
 * @brief 从复数共享内存tile存储数据到全局内存（默认设置版本）
 *
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数全局内存数组
 * @param[in] src 源复数共享内存tile
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是store函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐）
 * 直接调用实部和虚部的组件级存储函数。
 */
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    // 使用默认设置（axis=2, assume_aligned=false）分别存储实部和虚部
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

/**
 * @brief 异步从全局内存加载数据到复数共享内存tile（可配置轴和对齐）
 *
 * @tparam axis 加载操作的轴方向（0=列方向，1=行方向，2=默认平面方向）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数共享内存tile
 * @param[in] src 源复数全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 *
 * @note 此函数期望16字节对齐，否则行为未定义。
 * 
 * 这个函数异步加载复数数据的实部和虚部到共享内存tile中。
 * 异步加载可以隐藏内存延迟，提高GPU利用率。
 * assume_aligned=true时要求数据按16字节对齐，可以启用更高效的指令。
 */
template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    // 分别异步加载实部和虚部数据到共享内存tile
    // 使用相同的轴方向和对齐假设设置
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}


/**
 * @brief 异步从全局内存加载数据到复数共享内存tile（默认设置版本）
 *
 * @tparam CST 复数共享tile类型（必须满足ducks::cst::all概念）
 * @tparam CGL 复数全局数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<CST>
 * @param[out] dst 目标复数共享内存tile
 * @param[in] src 源复数全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 *
 * @note 此函数期望16字节对齐，否则行为未定义。
 * 
 * 这是load_async函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐，需要运行时检查）
 * 直接调用实部和虚部的组件级异步加载函数。
 */
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    // 使用默认设置（axis=2, assume_aligned=false）分别异步加载实部和虚部
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
