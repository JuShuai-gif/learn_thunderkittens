/**
 * @file
 * @brief Tensor Memory Accelerator (TMA) operations targetting ThunderKittens tile types.
 * @brief 针对ThunderKittens瓦片类型的张量内存加速器（TMA）操作
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../../util/util.cuh"

namespace kittens {
namespace tma {
/**
 * @brief 为swizzled tiles生成TMA坐标
 * @tparam ST swizzled tile类型，必须具有swizzle属性
 * @tparam axis 坐标轴：2表示ROW-major，1表示COL-major，0表示BATCH-major
 * @param unit_coord 单元坐标（在tile内的坐标）
 * @return int4 TMA所需的4D/5D坐标
 */
namespace detail {
template<kittens::ducks::st::all ST, int axis> __device__ inline int4 tma_coords(const coord<ducks::default_type> &unit_coord) {
    static_assert(ST::swizzle, "tma_coords only should be called for swizzled tiles");
    // 计算每个swizzle元素包含的基本元素数量
    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(typename ST::dtype);
    // 根据不同的轴重新排列坐标顺序以匹配TMA期望的布局
    if constexpr      (axis == 2) return {unit_coord.r, unit_coord.c / swizzle_elements, unit_coord.d, unit_coord.b};
    else if constexpr (axis == 1) return {unit_coord.d, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.b};
    else if constexpr (axis == 0) return {unit_coord.b, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.d};
}
} // namespace detail


/**
 * @brief Prefetches data from global memory into a shared memory tile, along with the tensormap.
 * @brief 从全局内存预取数据到共享内存瓦片，同时预取张量映射
 *
 * @tparam axis 数据布局轴：ROW(2)、COL(1)或BATCH(0)
 * @tparam policy 缓存策略：NORMAL、STREAMING、PERSISTENT
 * @tparam ST 目标共享内存瓦片类型，必须具有TMA兼容的布局
 * @tparam GL 全局内存源类型
 * @tparam COORD 坐标类型，默认为ST的coord
 * @param[out] dst 目标共享内存瓦片
 * @param[in] src 全局内存源
 * @param[in] idx 请求瓦片的坐标（以完整瓦片为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    // 获取TMA张量映射的指针
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    // 将瓦片坐标转换为单元坐标（考虑axis和3D布局）
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    // 处理swizzled tiles（使用5D TMA指令）
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

        if constexpr (policy == cache_policy::NORMAL) {
            // 使用普通缓存策略的5D TMA预取指令
            asm volatile (
                "cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
                " [%0, {%1, %2, %3, %4, %5}];"
                :
                : "l"(tma_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            // 使用指定缓存策略的5D TMA预取指令
            asm volatile (
                "cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint"
                " [%0, {%1, %2, %3, %4, %5}], %6;"
                :
                : "l"(tma_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 处理非swizzled tiles（使用4D TMA指令）
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        if constexpr (policy == cache_policy::NORMAL) {
            // 使用普通缓存策略的4D TMA预取指令
            asm volatile (
                "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
                " [%0, {%1, %2, %3, %4}];"
                :
                : "l"(tma_ptr),
                  "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            // 使用指定缓存策略的4D TMA预取指令
            asm volatile (
                "cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint"
                " [%0, {%1, %2, %3, %4}], %5;"
                :
                : "l"(tma_ptr),
                  "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
}

/**
 * @brief 默认的预取函数（使用ROW轴和普通缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    prefetch<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/* ----------   Async load and store data from gmem/smem  ---------- */
/* ----------   从全局内存/共享内存异步加载和存储数据  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory tile.
 * @brief 从共享内存瓦片异步存储数据到全局内存
 *
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作
 *
 * @tparam axis 数据布局轴
 * @tparam policy 缓存策略
 * @tparam ST 源共享内存瓦片类型
 * @tparam GL 目标全局内存类型
 * @tparam COORD 坐标类型
 * @param[out] dst 目标全局内存
 * @param[in] src 源共享内存瓦片
 * @param[in] idx 目标瓦片的坐标（以完整瓦片为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    // 获取TMA张量映射的指针
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    // 获取共享内存地址（转换为32位共享内存指针）
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    // 将瓦片坐标转换为单元坐标
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    // 处理swizzled tiles
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 确保所有先前的异步操作对共享内存代理可见
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            // 普通缓存策略的5D TMA存储指令
            asm volatile (
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            // 指定缓存策略的5D TMA存储指令
            asm volatile (
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 处理非swizzled tiles
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        // 确保所有先前的异步操作对共享内存代理可见
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            // 普通缓存策略的4D TMA存储指令
            asm volatile (
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            // 指定缓存策略的4D TMA存储指令
            asm volatile (
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    // 提交存储操作组
    store_commit_group();
}

/**
 * @brief 默认的异步存储函数（使用ROW轴和普通缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 针对padded全局内存的异步存储函数重载
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    store_commit_group();
}

/**
 * @brief 默认的padded全局内存异步存储函数
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief 使用异步TMA（Tensor Memory Accelerator）操作，将共享内存中的数据块进行加法归约后存储到全局内存
 * 
 * 该函数利用CUDA的cp.reduce.async.bulk.tensor指令，执行异步的加法归约和复制操作。
 * 支持4D/5D张量，支持不同的缓存策略，支持swizzled和非swizzled的内存布局。
 * 
 * @tparam axis 归约轴，指定在哪个维度上进行归约
 * @tparam policy 缓存策略，控制数据在L2缓存中的行为
 * @tparam ST 共享内存块类型，需要具有TMA兼容的布局
 * @tparam GL 全局内存类型
 * @tparam COORD 坐标类型，默认为共享内存块的坐标类型
 * @param[out] dst 目标全局内存的tensormap地址
 * @param[in] src 源共享内存块
 * @param[in] idx 目标位置坐标（以完整块为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {

    // 静态断言：检查数据类型是否支持TMA异步加法归约
    // fp8类型目前不支持TMA异步加法归约操作
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址（全局内存）和共享内存地址                    
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    
    // 将块坐标转换为单位坐标（针对指定的归约轴）    
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    // 根据内存布局类型（swizzled或非swizzled）执行不同的代码路径
    if constexpr (ST::swizzle) {
        // 对于swizzled内存布局，需要获取5D TMA坐标
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择不同的指令
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的5D张量异步加法归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            // 带缓存提示的5D张量异步加法归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 对于非swizzled内存布局，只支持在axis=2（通常是深度维度）上进行归约        
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

        // 根据缓存策略选择不同的指令
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的4D张量异步加法归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            // 带缓存提示的4D张量异步加法归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    // 提交存储操作到异步组
    store_commit_group();
}

/**
 * @brief store_add_async的简化版本，使用默认参数（ROW维度和NORMAL缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 针对pinned全局内存的store_add_async版本
 * 
 * 与普通全局内存版本功能相同，但针对pinned内存优化
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    // 静态断言：检查数据类型是否支持TMA异步加法归约
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址（pinned全局内存）和共享内存地址
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

    // 将块坐标转换为单位坐标    
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    // 根据内存布局类型执行不同的代码路径    
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择指令
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 非swizzled布局只支持axis=2
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择指令        
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    // 提交存储操作
    store_commit_group();
}

/**
 * @brief pinned全局内存版本store_add_async的简化版本
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_add_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief 使用异步TMA操作，将共享内存中的数据块进行最小值归约后存储到全局内存
 * 
 * 该函数利用CUDA的cp.reduce.async.bulk.tensor指令，执行异步的最小值归约和复制操作。
 * 支持4D/5D张量，支持不同的缓存策略，支持swizzled和非swizzled的内存布局。
 * 
 * @tparam axis 归约轴，指定在哪个维度上进行归约
 * @tparam policy 缓存策略，控制数据在L2缓存中的行为
 * @tparam ST 共享内存块类型，需要具有TMA兼容的布局
 * @tparam GL 全局内存类型
 * @tparam COORD 坐标类型，默认为共享内存块的坐标类型
 * @param[out] dst 目标全局内存的tensormap地址
 * @param[in] src 源共享内存块
 * @param[in] idx 目标位置坐标（以完整块为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    // 静态断言：检查数据类型是否支持TMA异步最小值归约
    // fp32类型不支持TMA异步最小/最大值归约
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    // fp8类型不支持TMA异步加法归约（这里可能注释有误，应该是最小值归约）
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址和共享内存地址
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    // 将块坐标转换为单位坐标    
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    // 根据内存布局类型执行不同的代码路径    
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择指令        
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的5D张量异步最小值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            // 带缓存提示的5D张量异步最小值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 非swizzled布局只支持axis=2
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

        // 根据缓存策略选择指令
        if constexpr (policy == cache_policy::NORMAL) {

            // 标准缓存策略的4D张量异步最小值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            // 带缓存提示的4D张量异步最小值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
    // 提交存储操作
    store_commit_group();
}

/**
 * @brief store_min_async的简化版本，使用默认参数（ROW维度和NORMAL缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    store_min_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 针对pinned全局内存的store_min_async版本
 * 
 * 与普通全局内存版本功能相同，但针对pinned内存优化
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {

    // 静态断言：检查数据类型是否支持TMA异步最小值归约    
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址和共享内存地址
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));

    // 将块坐标转换为单位坐标
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    // 根据内存布局类型执行不同的代码路径    
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        // 根据缓存策略选择指令
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
        
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {// 非swizzled布局只支持axis=2
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        // 内存屏障
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
                // 根据缓存策略选择指令
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }    
    // 提交存储操作
    store_commit_group();
}

/**
 * @brief pinned全局内存版本store_min_async的简化版本
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_min_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief 使用异步TMA操作，将共享内存中的数据块进行最大值归约后存储到全局内存
 * 
 * 该函数利用CUDA的cp.reduce.async.bulk.tensor指令，执行异步的最大值归约和复制操作。
 * 支持4D/5D张量，支持不同的缓存策略，支持swizzled和非swizzled的内存布局。
 * 
 * @tparam axis 归约轴，指定在哪个维度上进行归约
 * @tparam policy 缓存策略，控制数据在L2缓存中的行为
 * @tparam ST 共享内存块类型，需要具有TMA兼容的布局
 * @tparam GL 全局内存类型
 * @tparam COORD 坐标类型，默认为共享内存块的坐标类型
 * @param[out] dst 目标全局内存的tensormap地址
 * @param[in] src 源共享内存块
 * @param[in] idx 目标位置坐标（以完整块为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    // 静态断言：检查数据类型是否支持TMA异步最大值归约
    // fp32类型不支持TMA异步最小/最大值归约操作    
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    // fp8类型不支持TMA异步加法归约（注释可能需要修正，应该是最大值归约）
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址（全局内存）和共享内存地址
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    // 将块坐标转换为单位坐标（针对指定的归约轴）
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    // 根据内存布局类型（swizzled或非swizzled）执行不同的代码路径    
    if constexpr (ST::swizzle) {
        // 对于swizzled内存布局，需要获取5D TMA坐标
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择不同的指令        
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的5D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {            // 带缓存提示的5D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 对于非swizzled内存布局，只支持在axis=2（通常是深度维度）上进行归约
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");
        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择不同的指令
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的4D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {            // 带缓存提示的4D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }    
    // 提交存储操作到异步组
    store_commit_group();
}

/**
 * @brief store_max_async的简化版本，使用默认参数（ROW维度和NORMAL缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    store_max_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 针对pinned全局内存的store_max_async版本
 * 
 * 与普通全局内存版本功能相同，但针对pinned内存优化
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    // 静态断言：检查数据类型是否支持TMA异步最大值归约
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    // fp8类型不支持TMA异步加法归约（注释可能需要修正，应该是最大值归约）
    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
    // 获取TMA描述符地址（pinned全局内存）和共享内存地址
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    // 将块坐标转换为单位坐标（针对指定的归约轴）
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    // 根据内存布局类型（swizzled或非swizzled）执行不同的代码路径    
    if constexpr (ST::swizzle) {
        // 对于swizzled内存布局，需要获取5D TMA坐标
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);
        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 根据缓存策略选择不同的指令
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的5D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group"
                " [%0, {%2, %3, %4, %5, %6}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {            // 带缓存提示的5D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 对于非swizzled内存布局，只支持在axis=2（通常是深度维度）上进行归约
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

        // 内存屏障：确保之前的异步共享内存操作完成
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        // 根据缓存策略选择不同的指令
        if constexpr (policy == cache_policy::NORMAL) {
            // 标准缓存策略的4D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group"
                " [%0, {%2, %3, %4, %5}], [%1];"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {            // 带缓存提示的4D张量异步最大值归约指令
            asm volatile (
                "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
                " [%0, {%2, %3, %4, %5}], [%1], %6;"
                :
                : "l"(tma_ptr), "r"(src_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }    // 提交存储操作到异步组
    store_commit_group();
}

/**
 * @brief pinned全局内存版本store_max_async的简化版本
 */
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_max_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 * @brief 从全局内存异步加载数据到共享内存瓦片
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @tparam ST 具有TMA兼容布局的共享内存瓦片类型
 * @param[out] dst The destination shared memory tile.
 * @param[out] dst 目标共享内存瓦片
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] src_tma_map 全局内存中的源张量映射地址
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in,out] bar 用于异步复制同步的信号量
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_row_idx 请求瓦片的行坐标（以完整瓦片为单位）
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx 请求瓦片的列坐标（以完整瓦片为单位）
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    // 获取TMA张量映射的指针
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    // 获取信号量在共享内存中的地址
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    // 获取目标共享内存瓦片的地址
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    // 将瓦片坐标转换为单元坐标（考虑axis和3D布局）
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>();
    
    // 处理swizzled tiles（使用5D TMA指令）
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

        if constexpr (policy == cache_policy::NORMAL) {
            // 使用普通缓存策略的5D TMA加载指令，包含内存屏障完成传输的字节数
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
                : "memory"
            );
        }
        else {
            // 使用指定缓存策略的5D TMA加载指令
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        // 处理非swizzled tiles（使用4D TMA指令）
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

        if constexpr (policy == cache_policy::NORMAL) {
            // 使用普通缓存策略的4D TMA加载指令
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, %6}], [%2];"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
                : "memory"
            );
        }
        else {            // 使用指定缓存策略的4D TMA加载指令
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    }
}

/**
 * @brief 默认的异步加载函数（使用ROW轴和普通缓存策略）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile, across a threadblock cluster
 * @brief 从全局内存异步加载数据到共享内存瓦片，跨线程块集群
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 * 此函数使用CUDA的cp.async.bulk.tensor指令执行异步复制操作
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @tparam ST 具有TMA兼容布局的共享内存瓦片类型
 * @param[out] dst The destination shared memory tile.
 * @param[out] dst 目标共享内存瓦片
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] src_tma_map 全局内存中的源张量映射地址
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in,out] bar 用于异步复制同步的信号量
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_row_idx 请求瓦片的行坐标（以完整瓦片为单位）
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx 请求瓦片的列坐标（以完整瓦片为单位）
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 * @param[in] cluster_mask 要广播到的集群掩码
 */
#ifdef DF_BLACKWELL
// Blackwell架构特定版本，支持目标内存屏障线程块参数
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1)
#else
// 非Blackwell架构版本
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask)
#endif
{
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    
    if constexpr (ST::swizzle) {
        int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

#ifdef DF_BLACKWELL
        // Blackwell架构特定代码：支持跨线程块的内存屏障地址映射
        if(dst_mbar_cta != -1) {
            uint32_t neighbor_mbar_ptr;
            // 使用mapa指令将共享内存地址映射到目标线程块
            asm volatile (
                "mapa.shared::cluster.u32  %0, %1, %2;\n"
                : "=r"(neighbor_mbar_ptr)
                : "r"(mbar_ptr), "r"(dst_mbar_cta)
            );
            if constexpr (policy == cache_policy::NORMAL) {
                // 普通缓存策略的5D TMA加载指令，支持线程块组和集群多播
                asm volatile (
                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
                    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                    "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask)
                    : "memory"
                );
            } else {
                // 指定缓存策略的5D TMA加载指令，支持线程块组和集群多播
                asm volatile (
                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster.L2::cache_hint"
                    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                    "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                    : "memory"
                );
            }
        } else
#endif
        if constexpr (policy == cache_policy::NORMAL) {
                // 普通缓存策略的5D TMA加载指令，支持集群多播
                asm volatile (
                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                    "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask)
                    : "memory"
                );
        } else {            // 指定缓存策略的5D TMA加载指令，支持集群多播
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else {
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

#ifdef DF_BLACKWELL
        // Blackwell架构特定代码：支持跨线程块的内存屏障地址映射
        if(dst_mbar_cta != -1) {
            uint32_t neighbor_mbar_ptr;
            asm volatile (
                "mapa.shared::cluster.u32  %0, %1, %2;\n"
                : "=r"(neighbor_mbar_ptr)
                : "r"(mbar_ptr), "r"(dst_mbar_cta)
            );
            if constexpr (policy == cache_policy::NORMAL) {
                // 普通缓存策略的4D TMA加载指令，支持线程块组和集群多播                
                asm volatile (
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
                    " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                    "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "h"(cluster_mask)
                    : "memory"
                );
            }
            else {                // 指定缓存策略的4D TMA加载指令，支持线程块组和集群多播
                asm volatile (
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster.L2::cache_hint"
                    " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                    "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                    : "memory"
                );
            }
        } else 
#endif
        if constexpr (policy == cache_policy::NORMAL) {
                // 普通缓存策略的4D TMA加载指令，支持集群多播
                asm volatile (
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                    "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "h"(cluster_mask)
                    : "memory"
                );
        } else {
                // 指定缓存策略的4D TMA加载指令，支持集群多播
                asm volatile (
                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                    " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
                    :
                    : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
                    "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                    : "memory"
                );
        }
    }
}
#ifdef DF_BLACKWELL

/**
 * @brief 默认的集群异步加载函数（Blackwell架构版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
}
#else
/**
 * @brief 默认的集群异步加载函数（非Blackwell架构版本）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
}
#endif

} // namespace cluster
} // namespace tma

} // namespace kittens























