/**
 * @file
 * @brief 用于组范围调用向量TMA（张量内存访问）函数的函数。
 */

/* ----------   预取张量映射  ---------- */

/**
 * @brief 从全局内存预取数据到共享内存向量，同时预取张量映射。
 *
 * TMA预取操作将数据预取到缓存中，以减少后续实际访问的延迟。
 *
 * @tparam policy 缓存策略（如CG、CA等）
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标共享内存向量
 * @param[in] src 源全局内存对象
 * @param[in] idx 请求向量的坐标
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx) {
    // 将坐标转换为单位坐标（移除向量内部维度）
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    // 获取源全局内存对象的TMA指针
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    // 循环：每个线程处理一部分TMA预取操作
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        // 更新c维度坐标，以处理当前线程负责的部分
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        // 调用内部TMA预取函数
        ::kittens::detail::tma::vec_prefetch_tma_internal<policy>(tma_ptr, tma_coord);
    }
}
// 宏定义：为prefetch函数定义默认加载缓存策略的不同版本
__KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(prefetch)


/* ----------   从全局内存/共享内存异步加载和存储数据  ---------- */

/**
 * @brief 异步从共享内存向量存储数据到全局内存。
 *
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作。
 *
 * @tparam policy 缓存策略
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标全局内存对象
 * @param[in] src 源共享内存向量
 * @param[in] idx 向量目标的坐标
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    // 循环：每个线程处理一部分TMA存储操作
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    // 提交存储操作组，确保所有异步操作完成
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_async)

/**
 * @brief 异步从共享内存向量存储数据到指针全局内存对象（PGL版本）。
 *
 * PGL（Pointer Global Memory）是指针类型的全局内存对象。
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_async)


/**
 * @brief 异步执行加法归约操作并将结果存储到全局内存。
 *
 * 此函数使用CUDA的cp.reduce.async.bulk.tensor指令执行异步加法归约操作。
 * 归约操作将源共享内存向量的值加到目标全局内存的现有值上。
 *
 * @tparam policy 缓存策略
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标全局内存对象
 * @param[in] src 源共享内存向量
 * @param[in] idx 向量目标的坐标
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_add_async)


/**
 * @brief 异步执行加法归约操作并将结果存储到指针全局内存对象（PGL版本）。
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_add_async)


/**
 * @brief 异步执行最小值归约操作并将结果存储到全局内存。
 *
 * 此函数使用CUDA的cp.reduce.async.bulk.tensor指令执行异步最小值归约操作。
 * TMA不支持fp32类型的异步最小/最大归约操作。
 *
 * @tparam policy 缓存策略
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标全局内存对象
 * @param[in] src 源共享内存向量
 * @param[in] idx 向量目标的坐标
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx) {
    // 静态断言：TMA不支持fp32类型的异步最小/最大归约
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_min_async)

/**
 * @brief 异步执行最小值归约操作并将结果存储到指针全局内存对象（PGL版本）。
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_min_async)

/**
 * @brief 异步执行最大值归约操作并将结果存储到全局内存。
 *
 * 此函数使用CUDA的cp.reduce.async.bulk.tensor指令执行异步最大值归约操作。
 * TMA不支持fp32类型的异步最小/最大归约操作。
 *
 * @tparam policy 缓存策略
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标全局内存对象
 * @param[in] src 源共享内存向量
 * @param[in] idx 向量目标的坐标
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_max_async)

/**
 * @brief 异步执行最大值归约操作并将结果存储到指针全局内存对象（PGL版本）。
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_max_async)

/**
 * @brief 异步从全局内存加载数据到共享内存向量。
 *
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作。
 * 使用信号量进行异步复制的同步。
 *
 * @tparam policy 缓存策略
 * @tparam SV 具有TMA兼容布局的共享向量类型
 * @param[out] dst 目标共享内存向量
 * @param[in] src 源全局内存对象
 * @param[in] idx 请求向量的坐标
 * @param[in,out] bar 用于异步复制同步的信号量
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(load_async)
