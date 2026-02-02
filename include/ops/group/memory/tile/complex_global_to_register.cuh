/**
 * @file
 * @brief 协作式数据加载/存储函数 - 支持工作组协作在全局内存和寄存器之间直接传输数据
 */

/**
 * @brief 协作式加载数据（复数版本）- 从源数组加载数据到寄存器tile
 *
 * @tparam axis 加载操作的轴方向
 * @tparam CRT 寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CGL 全局内存数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<crt<...>>
 * @param[out] dst 目标tile，数据将加载到这里（复数，包含real和imag部分）
 * @param[in] src 源数组，从中加载数据（复数，包含real和imag部分）
 * @param[in] idx 坐标索引，用于确定加载位置
 * 
 * 该函数用于复数数据的协作式加载，分别加载实部和虚部
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    // 分别加载实部和虚部数据
    load<axis, CRT::component, CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, CRT::component, CGL::component, COORD>(dst.imag, src.imag, idx);
}

/**
 * @brief 协作式加载数据（默认轴版本）- 从源数组加载数据到寄存器tile
 *
 * @tparam CRT 寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CGL 全局内存数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<crt<...>>
 * @param[out] dst 目标tile，数据将加载到这里（复数）
 * @param[in] src 源数组，从中加载数据（复数）
 * @param[in] idx 坐标索引，用于确定加载位置
 * 
 * 默认使用axis=2进行加载，调用上面的复数版本load函数
 */
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    // 调用axis=2的复数版本加载函数
    load<2, CRT, CGL>(dst, src, idx);
}

/**
 * @brief 协作式存储数据（复数版本）- 从寄存器tile存储数据到目标数组
 *
 * @tparam axis 存储操作的轴方向
 * @tparam CRT 寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CGL 全局内存数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<crt<...>>
 * @param[out] dst 目标数组，数据将存储到这里（复数，包含real和imag部分）
 * @param[in] src 源tile，从中存储数据（复数，包含real和imag部分）
 * @param[in] idx 坐标索引，用于确定存储位置
 * 
 * 该函数用于复数数据的协作式存储，分别存储实部和虚部
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    // 分别存储实部和虚部数据
    store<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, idx);
}

/**
 * @brief 协作式存储数据（默认轴版本）- 从寄存器tile存储数据到目标数组
 *
 * @tparam CRT 寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CGL 全局内存数组类型（必须满足ducks::cgl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<crt<...>>
 * @param[out] dst 目标数组，数据将存储到这里（复数）
 * @param[in] src 源tile，从中存储数据（复数）
 * @param[in] idx 坐标索引，用于确定存储位置
 * 
 * 默认使用axis=2进行存储，调用上面的复数版本store函数
 */
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    // 调用axis=2的复数版本存储函数
    store<2, CRT, CGL>(dst, src, idx);
}
