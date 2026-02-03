/**
 * @file
 * @brief Functions for a group scope to call tile TMA functions.
 * @brief 组范围调用瓦片TMA（Tensor Memory Accelerator）函数的接口。
 * @details 这些函数封装了TMA的各种操作（预取、存储、原子操作等），确保只有每个warp的第一个线程调用TMA操作。
 *          TMA是NVIDIA GPU中的张量内存加速器，用于高效的大块数据传输。
 */

/**
 * @brief 预取数据从全局内存到共享内存（TMA操作）
 * @tparam axis 加载轴方向（行优先或列优先）
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型（默认为ST的坐标类型）
 * @param dst 目标共享内存瓦片
 * @param src 源全局内存瓦片
 * @param idx 坐标索引，指定从源瓦片的哪个位置开始加载
 * @details prefetch是预取操作，将数据从全局内存预取到共享内存，但不立即使用。
 *          只有warp内的第一个线程执行TMA操作，避免重复调用。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::prefetch<axis, policy, ST, GL, COORD>(dst, src, idx); // 只由lane 0执行，避免重复调用
    }
}

/**
 * @brief 预取数据从全局内存到共享内存（默认参数版本）
 * @details 使用默认参数：axis=dim::ROW（行优先），policy=cache_policy::NORMAL（正常缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::prefetch<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步存储数据从共享内存到全局内存（TMA操作）
 * @tparam axis 存储轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存瓦片
 * @param src 源共享内存瓦片
 * @param idx 坐标索引，指定存储到目标瓦片的哪个位置
 * @details 将共享内存中的数据异步存储到全局内存。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步存储数据从共享内存到全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

/**
 * @brief 异步存储数据从共享内存到分页全局内存（PGL）（TMA操作）
 * @tparam axis 存储轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam PGL 分页全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标分页全局内存瓦片
 * @param src 源共享内存瓦片
 * @param idx 坐标索引
 * @details 分页全局内存（PGL）可能指使用CUDA统一内存或分页内存的情况。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步存储数据从共享内存到分页全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

/**
 * @brief 异步原子加存储操作从共享内存到全局内存（TMA操作）
 * @tparam axis 存储轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存瓦片
 * @param src 源共享内存瓦片
 * @param idx 坐标索引
 * @details store_add_async执行原子加操作：dst += src。用于归约操作等场景。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子加存储操作从共享内存到全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

/**
 * @brief 异步原子加存储操作从共享内存到分页全局内存（TMA操作）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子加存储操作从共享内存到分页全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

/**
 * @brief 异步原子最小值存储操作从共享内存到全局内存（TMA操作）
 * @tparam axis 存储轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存瓦片
 * @param src 源共享内存瓦片
 * @param idx 坐标索引
 * @details store_min_async执行原子最小值操作：dst = min(dst, src)。用于归约操作等场景。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子最小值存储操作从共享内存到全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

/**
 * @brief 异步原子最小值存储操作从共享内存到分页全局内存（TMA操作）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子最小值存储操作从共享内存到分页全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}


/**
 * @brief 异步原子最大值存储操作从共享内存到全局内存（TMA操作）
 * @tparam axis 存储轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存瓦片
 * @param src 源共享内存瓦片
 * @param idx 坐标索引
 * @details store_max_async执行原子最大值操作：dst = max(dst, src)。用于归约操作等场景。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子最大值存储操作从共享内存到全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}


/**
 * @brief 异步原子最大值存储操作从共享内存到分页全局内存（TMA操作）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}

/**
 * @brief 异步原子最大值存储操作从共享内存到分页全局内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}



/**
 * @brief 异步加载数据从全局内存到共享内存（TMA操作）
 * @tparam axis 加载轴方向
 * @tparam policy 缓存策略
 * @tparam ST 共享内存瓦片类型
 * @tparam GL 全局内存瓦片类型
 * @tparam COORD 坐标类型
 * @param dst 目标共享内存瓦片
 * @param src 源全局内存瓦片
 * @param idx 坐标索引
 * @param bar 信号量，用于同步TMA操作的完成
 * @details load_async与prefetch类似，但使用信号量来同步操作完成。
 *          这是TMA加载操作的标准版本，需要信号量来确保数据可用。
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar); // Don't do the mask
    }
}

/**
 * @brief 异步加载数据从全局内存到共享内存（默认参数版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
    }
}
