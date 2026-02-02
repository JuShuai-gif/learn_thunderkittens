/**
 * @file
 * @brief 共享内存与寄存器之间的协作数据传输函数
 * 
 * 该文件提供工作组（协作warp）在共享内存和寄存器之间直接传输数据的函数
 */

/*
使用C++20概念(concepts)确保类型安全，只有满足ducks::crt::all和ducks::cst::all概念的类型才能使用这些函数

*/


/**
 * @brief 从复数共享内存tile加载数据到复数寄存器tile
 *
 * @tparam CRT 复数寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CST 复数共享内存tile类型（必须满足ducks::cst::all概念）
 * @param[out] dst 目标复数寄存器tile
 * @param[in] src 源复数共享内存tile
 * 
 * 该函数将复数共享内存tile中的数据加载到复数寄存器tile中
 * 复数数据由实部(real)和虚部(imag)组成，因此分别加载两个部分
 * 函数假设已经进行了适当的内存对齐和布局匹配
 */
template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void load(CRT &dst, const CST &src) {
    // 分别加载实部和虚部
    // 调用底层组件级别的load函数，处理实际的数据传输
    load(dst.real, src.real);
    load(dst.imag, src.imag);
}

/**
 * @brief 从复数寄存器tile存储数据到复数共享内存tile
 *
 * @tparam CRT 复数寄存器tile类型（必须满足ducks::crt::all概念）
 * @tparam CST 复数共享内存tile类型（必须满足ducks::cst::all概念）
 * @param[out] dst 目标复数共享内存tile
 * @param[in] src 源复数寄存器tile
 * 
 * 该函数将复数寄存器tile中的数据存储到复数共享内存tile中
 * 复数数据由实部(real)和虚部(imag)组成，因此分别存储两个部分
 * 函数假设已经进行了适当的内存对齐和布局匹配
 */
template<ducks::crt::all CRT, ducks::cst::all CST>
__device__ inline static void store(CST &dst, const CRT &src) {
    // 分别存储实部和虚部
    // 调用底层组件级别的store函数，处理实际的数据传输
    store(dst.real, src.real);
    store(dst.imag, src.imag);
}
