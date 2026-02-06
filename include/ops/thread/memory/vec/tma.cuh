/**
 * @file
 * @brief Tensor Memory Accelerator (TMA) operations targetting ThunderKittens vector types.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../../util/util.cuh"

// This is a macro that helps us define default cache policy versions of each function.
#define __KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_PGL_DEFAULT_LOAD_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(const GL &dst, const SV &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(const PGL &dst, const SV &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx, semaphore& bar) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar); \
}
#define __KITTENS_TMA_DEFINE_PGL_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx, semaphore& bar) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar); \
}
#define __KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar, cluster_mask, dst_mbar_cta); \
}
#define __KITTENS_TMA_DEFINE_PGL_CLUSTER_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar, cluster_mask, dst_mbar_cta); \
}


namespace kittens {

namespace detail {
namespace tma {

// ==================== TMA（Tensor Memory Access）操作函数集合 ====================
// 这些函数提供了使用TMA进行张量数据移动的底层接口，支持不同的缓存策略和数据操作

// 预取函数：将张量数据预取到L2缓存
// policy: 缓存策略（NORMAL或其他优化策略）
// tma_ptr: TMA描述符指针（描述张量的布局和内存访问模式）
// tma_coord: 4维坐标（c,r,d,b），指定要预取的张量切片
template<cache_policy policy> __device__ static inline void vec_prefetch_tma_internal(uint64_t tma_ptr, coord<> tma_coord) {
    if constexpr (policy == cache_policy::NORMAL) {
        // 普通缓存策略的预取指令
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile: 4维张量数据块预取到L2缓存
        asm volatile (
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
            " [%0, {%1, %2, %3, %4}];"// 使用TMA描述符和坐标访问张量
            :
            : "l"(tma_ptr),// %0: TMA描述符指针
             "r"(tma_coord.c), // %1: c坐标（通常表示列）
             "r"(tma_coord.r), // %2: r坐标（通常表示行）
             "r"(tma_coord.d), // %3: d坐标（通常表示深度/通道）
             "r"(tma_coord.b)// %4: b坐标（通常表示批次）
            : "memory"// 告知编译器内存被修改
        );
    }
    else {
        // 带缓存提示的预取指令（如流式加载、绕过缓存等优化策略）
        asm volatile (
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint"
            " [%0, {%1, %2, %3, %4}], %5;"// %5: 缓存策略参数
            :
            : "l"(tma_ptr), 
            "r"(tma_coord.c), 
            "r"(tma_coord.r), 
            "r"(tma_coord.d), 
            "r"(tma_coord.b), 
            "l"(make_cache_policy<policy>())// 生成缓存策略参数
            : "memory"
        );
    }
}
// 异步存储函数：将共享内存数据存储到全局内存
// src_i_ptr: 共享内存中的源数据指针
template<cache_policy policy> __device__ static inline void vec_store_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
        // 内存栅栏：确保之前的共享内存异步操作完成
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    
    if constexpr (policy == cache_policy::NORMAL) {
                // cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group:
        // 将共享内存中的张量数据块批量异步存储到全局内存
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"// [目标全局内存], [源共享内存]
            :
            : "l"(tma_ptr), // %0: 目标全局内存的TMA描述符
            "r"(src_i_ptr), // %1: 源共享内存地址
            "r"(tma_coord.c),  // %2: c坐标
            "r"(tma_coord.r), // %3: r坐标
            "r"(tma_coord.d), // %4: d坐标
            "r"(tma_coord.b)// %5: b坐标
            : "memory"
        );
    }
    else {
        // 带缓存提示的异步存储指令
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"// %6: 缓存策略参数
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
// 异步加法归约存储：将共享内存数据原子加操作存储到全局内存
// 用于实现原子加操作（如梯度累加）
template<cache_policy policy> __device__ static inline void vec_store_add_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        // cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group:
        // 批量异步归约存储，执行加法操作
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        // 带缓存提示的异步加法归约存储
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
// 异步最小值归约存储：将共享内存数据原子最小值操作存储到全局内存
// 用于实现原子最小值操作
template<cache_policy policy> __device__ static inline void vec_store_min_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        // cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group:
        // 批量异步归约存储，执行最小值操作
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        // 带缓存提示的异步最小值归约存储
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

// 异步最大值归约存储：将共享内存数据原子最大值操作存储到全局内存
// 用于实现原子最大值操作
template<cache_policy policy> __device__ static inline void vec_store_max_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        // cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group:
        // 批量异步归约存储，执行最大值操作
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        // 带缓存提示的异步最大值归约存储
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
// 异步加载函数：从全局内存加载数据到集群共享内存
// dst_i_ptr: 目标共享内存地址
// mbar_ptr: 内存屏障指针，用于同步数据传输完成
template<cache_policy policy> __device__ static inline void vec_load_async_tma_internal(uint64_t tma_ptr, uint32_t dst_i_ptr, uint32_t mbar_ptr, coord<> tma_coord) {
    if constexpr (policy == cache_policy::NORMAL) {
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes:
        // 使用内存屏障的批量异步张量加载
        // mbarrier::complete_tx::bytes: 内存屏障跟踪传输的字节数
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"// [目标], [源], [内存屏障]
            :
            : "r"(dst_i_ptr), // %0: 目标共享内存地址
            "l"(tma_ptr), // %1: 源全局内存TMA描述符
            "r"(mbar_ptr), // %2: 内存屏障地址
            "r"(tma_coord.c), // %3: c坐标
            "r"(tma_coord.r), // %4: r坐标
            "r"(tma_coord.d), // %5: d坐标
            "r"(tma_coord.b)// %6: b坐标
            : "memory"
        );
    }
    else {
        // 带缓存提示的异步加载指令
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"// %7: 缓存策略参数
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

// ==================== 集群操作命名空间 ====================
// 提供跨CTA（线程块）的集群级TMA操作
namespace cluster {

// 集群级异步加载函数：支持多播和跨CTA内存屏障
// cluster_mask: 集群掩码，指定哪些CTA参与多播
// dst_mbar_cta: 目标内存屏障所在的CTA索引（-1表示当前CTA）
template<cache_policy policy> __device__ static inline void vec_load_async_tma_internal(uint64_t tma_ptr, uint32_t dst_i_ptr, uint32_t mbar_ptr, coord<> tma_coord, uint16_t cluster_mask, int dst_mbar_cta=-1) {
#ifdef DF_BLACKWELL// 针对Blackwell架构的特殊处理
    if(dst_mbar_cta != -1) {
        // 映射目标CTA的内存屏障地址
        uint32_t neighbor_mbar_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"// 映射集群共享内存地址
            : "=r"(neighbor_mbar_ptr)
            : "r"(mbar_ptr), "r"(dst_mbar_cta)
        );
        if constexpr (policy == cache_policy::NORMAL) {
            // cta_group::2.multicast::cluster: 双CTA组多播加载
            asm volatile (
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"// %7: 集群掩码
                :
                : "r"(dst_i_ptr), // 目标共享内存
                "l"(tma_ptr), // 源全局内存TMA描述符
                "r"(neighbor_mbar_ptr),// 映射后的内存屏障地址
                "r"(tma_coord.c), // c坐标
                "r"(tma_coord.r),// r坐标
                 "r"(tma_coord.d), // d坐标
                 "r"(tma_coord.b),// b坐标
                 "h"(cluster_mask)// 集群掩码（低16位）
                : "memory"
            );
        }
        else {
            // 带缓存提示的双CTA组多播加载
            asm volatile (
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"// %8: 缓存策略参数
                :
                : "r"(dst_i_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else 
#endif
    // 默认情况：单CTA组或多播加载（当前CTA的内存屏障）
    if constexpr (policy == cache_policy::NORMAL) {
        // multicast::cluster: 集群多播加载
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_i_ptr), // 目标共享内存
            "l"(tma_ptr), // 源全局内存TMA描述符
            "r"(mbar_ptr),// 当前CTA的内存屏障
            "r"(tma_coord.c), // c坐标
            "r"(tma_coord.r), // r坐标
            "r"(tma_coord.d), // d坐标
            "r"(tma_coord.b), // b坐标
            "h"(cluster_mask)// 集群掩码
            : "memory"
        );
    }
    else {
        // 带缓存提示的集群多播加载
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
} // namespace cluster

} // namespace tma
} // namespace detail



namespace tma {
/**
 * @brief 预取数据从全局内存到共享内存（使用TMA）
 * @tparam policy 缓存策略（如L2缓存提示）
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型（默认为与SV关联的坐标类型）
 * @param dst 目标共享内存向量引用
 * @param src 源全局内存向量引用
 * @param idx 要访问的坐标位置
 * 
 * @note 使用Tensor Memory Accelerator (TMA)进行异步预取，提前将数据加载到缓存
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx) {
    // 将坐标转换为TMA所需的单位坐标格式（-1表示自动推导，3表示3D坐标）    
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    // 获取源全局内存的TMA指针
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    // 遍历SV的TMA维度2（通常是y维度或tile数量）    
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        // 为每个tile创建坐标        
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;// 沿C维度偏移
        // 调用内部TMA预取函数
        ::kittens::detail::tma::vec_prefetch_tma_internal<policy>(tma_ptr, tma_coord);
    }
}
// 生成默认的预取函数重载（带不同缓存策略）
__KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(prefetch)

/**
 * @brief 异步存储数据从共享内存到全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 * 
 * @note 使用TMA进行异步存储操作，存储后提交存储组
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    // 获取共享内存地址（转换为32位共享内存指针）
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        // 计算当前tile的共享内存指针
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        // 调用内部TMA异步存储函数
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    // 提交存储组，确保存储操作有序执行
    ::kittens::tma::store_commit_group();
}
// 生成默认的异步存储函数重载
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_async)


/**
 * @brief 异步存储数据从共享内存到PGL（Padded Global Load）全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam PGL 带填充的全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标PGL全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 * 
 * @note PGL类型支持带填充的全局内存访问，适用于不规则数据布局
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
// 生成PGL专用的异步存储函数重载
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_async)

/**
 * @brief 异步存储并累加（atomic add）数据从共享内存到全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 * 
 * @note 执行原子加操作，将源数据加到目标内存位置
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        // 调用内部TMA异步累加存储函数
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_add_async)

/**
 * @brief 异步存储并累加数据从共享内存到PGL全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam PGL 带填充的全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标PGL全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_add_async)

/**
 * @brief 异步存储并取最小值（atomic min）数据从共享内存到全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 * 
 * @note 静态断言确保不用于fp32类型（TMA不支持fp32的异步min/max归约）
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx) {
    // TMA不支持fp32类型的异步min/max归约操作
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_min_async)


/**
 * @brief 异步存储并取最小值数据从共享内存到PGL全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam PGL 带填充的全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标PGL全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_min_async)

/**
 * @brief 异步存储并取最大值（atomic max）数据从共享内存到全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_max_async)

/**
 * @brief 异步存储并取最大值数据从共享内存到PGL全局内存
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam PGL 带填充的全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标PGL全局内存向量引用
 * @param src 源共享内存向量引用
 * @param idx 坐标位置
 */
template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_max_async)

/**
 * @brief 异步加载数据从全局内存到共享内存（带信号量）
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标共享内存向量引用
 * @param src 源全局内存向量引用
 * @param idx 坐标位置
 * @param bar 信号量引用，用于同步
 * 
 * @note 使用信号量确保加载操作的完成，适用于需要同步的场景
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(load_async)


// ============================================================================
// 集群（Cluster）级别的TMA操作
// ============================================================================
namespace cluster {

/**
 * @brief 集群级别的异步加载（多个CTA协同）
 * @tparam policy 缓存策略
 * @tparam SV 共享内存向量类型
 * @tparam GL 全局内存向量类型
 * @tparam COORD 坐标类型
 * @param dst 目标共享内存向量引用
 * @param src 源全局内存向量引用
 * @param idx 坐标位置
 * @param bar 信号量引用
 * @param cluster_mask 集群掩码，指定参与协作的CTA
 * @param dst_mbar_cta 目标信号量的CTA ID（默认-1表示当前CTA）
 * 
 * @note 支持多个CTA协同加载，适用于大规模数据加载场景
 */    
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::cluster::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord, cluster_mask, dst_mbar_cta);
    }
}
__KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(load_async)
} // namespace cluster
} // namespace tma
} // namespace kittens














