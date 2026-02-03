/**
 * @file
 * @brief 用于组范围调用集群范围内的向量TMA（Tensor Memory Access）函数的函数。
 */

/**
 * @brief 异步从全局内存加载数据到共享内存向量，并在集群内广播
 *
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作。
 * TMA（Tensor Memory Access）是CUDA中的张量内存访问功能，用于高效的大块数据传输。
 *
 * @tparam policy 缓存策略（如CG、CA等）
 * @tparam SV 共享向量类型，需要具有TMA兼容的布局
 * @param[out] dst 目标共享内存向量
 * @param[in] src 源全局内存对象
 * @param[in] idx 请求向量的坐标
 * @param[in,out] bar 用于异步复制同步的信号量
 * @param[in] cluster_mask 要广播到的集群掩码
 * @param[in] dst_mbar_cta 目标内存屏障的CTA索引，默认为-1
 */
 template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
 __device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    // 将坐标转换为单位坐标（移除向量内部维度，保留三维空间坐标）
     coord<> unit_coord = idx.template unit_coord<-1, 3>();
    // 获取源全局内存对象的TMA指针（用于张量内存访问）
     uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    // 将信号量bar的地址转换为共享内存地址（32位表示）
     uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    // 将目标共享内存向量dst的地址转换为共享内存地址（32位表示）
     uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    // 循环：每个线程处理一部分TMA操作，以WARP_THREADS（通常为32）为步长
     for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        // 复制单位坐标
         coord<> tma_coord = unit_coord;
        // 更新c维度坐标，以处理当前线程负责的部分
         tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        // 计算当前线程负责的目标共享内存地址
         uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        // 调用内部TMA异步加载函数
         ::kittens::detail::tma::cluster::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord, cluster_mask, dst_mbar_cta);
     }
 }
 // 宏定义：为load_async函数定义集群信号量和缓存策略的不同版本
 __KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(load_async) 
