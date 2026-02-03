/**
 * @file
 * @brief Functions for a group scope to call tile cluster-wide TMA functions.
 * @brief 用于组范围调用瓦片集群范围内的TMA（Tensor Memory Accelerator）函数的接口。
 * @details 这些函数封装了TMA集群加载操作，确保只有每个warp的第一个线程调用TMA操作，避免竞争条件。
 */

// 针对Blackwell架构的TMA集群加载函数实现

/**
 * @brief 异步加载数据从全局内存到共享内存（TMA集群操作） - Blackwell架构版本
 * @tparam axis 加载轴方向（行优先或列优先）
 * @tparam policy 缓存策略（如NORMAL, STREAMING等）
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型（默认为ST的坐标类型）
 * @param dst[out] 目标共享内存瓦片
 * @param src[in] 源全局内存瓦片
 * @param idx 坐标索引，指定从源瓦片的哪个位置开始加载
 * @param bar 信号量，用于同步TMA操作的完成
 * @param cluster_mask 集群掩码，标识参与协作的线程块集合
 * @param dst_mbar_cta 目标多块屏障的CTA（Cooperative Thread Array）ID，默认为-1
 * @details 这是Blackwell架构特有的版本，增加了dst_mbar_cta参数用于多块屏障管理。
 *          只有每个warp的第一个线程（laneid() == 0）调用实际的TMA操作。
 */
#ifdef DF_BLACKWELL
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}

/**
 * @brief 异步加载数据从全局内存到共享内存（默认参数版本） - Blackwell架构版本
 * @details 使用默认参数：axis=dim::ROW（行优先），policy=cache_policy::NORMAL（正常缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    // 只有warp内的第一个线程（lane 0）执行TMA操作
    if(laneid() == 0) {
        // 调用底层TMA集群异步加载函数，使用默认参数
        ::kittens::tma::cluster::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}
#else
// 非Blackwell架构的TMA集群加载函数实现（如Hopper架构）

/**
 * @brief 异步加载数据从全局内存到共享内存（TMA集群操作） - 非Blackwell架构版本
 * @details 与非Blackwell版本相比，缺少dst_mbar_cta参数。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    // 只有warp内的第一个线程（lane 0）执行TMA操作
    if(laneid() == 0) {
        // 调用底层TMA集群异步加载函数
        ::kittens::tma::cluster::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}

/**
 * @brief 异步加载数据从全局内存到共享内存（默认参数版本） - 非Blackwell架构版本
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    // 只有warp内的第一个线程（lane 0）执行TMA操作
    if(laneid() == 0) {
        // 调用底层TMA集群异步加载函数，使用默认参数
        ::kittens::tma::cluster::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}
#endif
