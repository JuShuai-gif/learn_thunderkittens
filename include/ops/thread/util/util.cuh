#pragma once

#include "sync.cuh"///< 包含同步操作头文件
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
#include "tma.cuh"///< Hopper/Blackwell架构包含TMA操作头文件
#endif

namespace kittens {

/* ----------   防止通用寻址的PTX内联汇编包装 ---------- */

/**
 * @brief 内存移动操作模板结构体
 * @tparam T 数据类型
 * @note 提供不同类型的内存移动操作，使用PTX内联汇编以避免编译器生成通用寻址代码
 */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);///< 从共享内存加载到寄存器
    __device__ static inline void sts(uint32_t dst, const T& src); ///< 从寄存器存储到共享内存
    __device__ static inline void ldg(T& dst, T* src);///< 从全局内存加载到寄存器
    __device__ static inline void stg(T* dst, const T& src);///< 从寄存器存储到全局内存
};


// 非压缩类型（标量）的特化定义

/**
 * @brief bfloat16类型的内存移动特化
 * @note 使用16位二进制表示浮点数，常用于深度学习
 */
template<> struct move<bf16> {
    /// 从共享内存加载bfloat16到寄存器
    __device__ static inline void lds(bf16& dst, uint32_t src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
    }
    /// 从寄存器存储bfloat16到共享内存
    __device__ static inline void sts(uint32_t dst, const bf16& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
    }
    /// 从全局内存加载bfloat16到寄存器
    __device__ static inline void ldg(bf16& dst, bf16* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    /// 从寄存器存储bfloat16到全局内存
    __device__ static inline void stg(bf16* dst, const bf16& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};

/**
 * @brief half精度浮点类型的内存移动特化
 * @note 16位浮点数，常用于深度学习
 */
template<> struct move<half> {
    /// 从共享内存加载half到寄存器
    __device__ static inline void lds(half& dst, uint32_t src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
    }
    /// 从寄存器存储half到共享内存
    __device__ static inline void sts(uint32_t dst, const half& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
    }
    /// 从全局内存加载half到寄存器
    __device__ static inline void ldg(half& dst, half* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    /// 从寄存器存储half到全局内存
    __device__ static inline void stg(half* dst, const half& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};

/**
 * @brief 单精度浮点类型的内存移动特化
 */
template<> struct move<float> {
    /// 从共享内存加载float到寄存器
    __device__ static inline void lds(float& dst, uint32_t src) {
        asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "r"(src));
    }
    /// 从寄存器存储float到共享内存
    __device__ static inline void sts(uint32_t dst, const float& src) {
        asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "r"(dst));
    }
    /// 从全局内存加载float到寄存器
    __device__ static inline void ldg(float& dst, float* src) {
        asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(dst) : "l"(src));
    }
    /// 从寄存器存储float到全局内存
    __device__ static inline void stg(float* dst, const float& src) {
        asm volatile("st.global.f32 [%1], %0;\n" : : "f"(src), "l"(dst));
    }
};

/**
 * @brief 整数类型的内存移动特化
 */
template<> struct move<int> {
    /// 从共享内存加载int到寄存器
    __device__ static inline void lds(int& dst, uint32_t src) {
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(dst) : "r"(src));
    }
    /// 从寄存器存储int到共享内存
    __device__ static inline void sts(uint32_t dst, const int& src) {
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(src), "r"(dst));
    }
    /// 从全局内存加载int到寄存器
    __device__ static inline void ldg(int& dst, int* src) {
        asm volatile("ld.global.u32 %0, [%1];\n" : "=r"(dst) : "l"(src));
    }
    /// 从寄存器存储int到全局内存
    __device__ static inline void stg(int* dst, const int& src) {
        asm volatile("st.global.u32 [%1], %0;\n" : : "r"(src), "l"(dst));
    }
};

// 压缩类型（向量）的特化定义

/**
 * @brief 两个bfloat16组成的向量类型内存移动特化
 */
template<> struct move<bf16_2> {
    /// 从共享内存加载bf16_2到寄存器（32位加载）
    __device__ static inline void lds(bf16_2& dst, uint32_t src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
    }
    /// 从寄存器存储bf16_2到共享内存
    __device__ static inline void sts(uint32_t dst, const bf16_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    /// 从全局内存加载bf16_2到寄存器
    __device__ static inline void ldg(bf16_2& dst, bf16_2* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    /// 从寄存器存储bf16_2到全局内存
    __device__ static inline void stg(bf16_2* dst, const bf16_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
    /// 从共享内存加载4个bf16_2（8x8矩阵的4行）到寄存器，用于矩阵乘法
    __device__ static inline void ldsm4(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 从共享内存转置加载4个bf16_2（8x8矩阵的4列）到寄存器
    __device__ static inline void ldsm4t(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 将4个bf16_2存储到共享内存，用于矩阵存储
    __device__ static inline void stsm4(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
    /// 转置存储4个bf16_2到共享内存
    __device__ static inline void stsm4t(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};

/**
 * @brief 两个half精度浮点组成的向量类型内存移动特化
 */
template<> struct move<half_2> {
    /// 从共享内存加载half_2到寄存器
    __device__ static inline void lds(half_2& dst, uint32_t src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
    }
    /// 从寄存器存储half_2到共享内存
    __device__ static inline void sts(uint32_t dst, const half_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    /// 从全局内存加载half_2到寄存器
    __device__ static inline void ldg(half_2& dst, half_2* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    /// 从寄存器存储half_2到全局内存
    __device__ static inline void stg(half_2* dst, const half_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
    /// 从共享内存加载4个half_2（8x8矩阵的4行）到寄存器
    __device__ static inline void ldsm4(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 从共享内存转置加载4个half_2（8x8矩阵的4列）到寄存器
    __device__ static inline void ldsm4t(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 将4个half_2存储到共享内存
    __device__ static inline void stsm4(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
    /// 转置存储4个half_2到共享内存
    __device__ static inline void stsm4t(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};

/**
 * @brief 两个float组成的向量类型内存移动特化
 */
template<> struct move<float2> {
    /// 从共享内存加载float2到寄存器（向量加载）
    __device__ static inline void lds(float2& dst, uint32_t src) {
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "r"(src));
    }
    /// 从寄存器存储float2到共享内存（向量存储）
    __device__ static inline void sts(uint32_t dst, const float2& src) {
        asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "r"(dst));
    }
    /// 从全局内存加载float2到寄存器
    __device__ static inline void ldg(float2& dst, float2* src) {
        asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src));
    }
    /// 从寄存器存储float2到全局内存
    __device__ static inline void stg(float2* dst, const float2& src) {
        asm volatile("st.global.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst));
    }
};

/**
 * @brief 四个float组成的向量类型内存移动特化
 */
template<> struct move<float4> {
    /// 从共享内存加载float4到寄存器（向量加载）
    __device__ static inline void lds(float4& dst, uint32_t src) {
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "r"(src));
    }
    /// 从寄存器存储float4到共享内存（向量存储）
    __device__ static inline void sts(uint32_t dst, const float4& src) {
        asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "r"(dst));
    }
    /// 从全局内存加载float4到寄存器
    __device__ static inline void ldg(float4& dst, float4* src) {
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src));
    }
    /// 从寄存器存储float4到全局内存
    __device__ static inline void stg(float4* dst, const float4& src) {
        asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst));
    }
};
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief fp8e4m3格式（4个8位浮点）向量类型内存移动特化
 * @note Hopper/Blackwell架构支持的新浮点格式，用于高效深度学习推理
 */
template<> struct move<fp8e4m3_4> {
    /// 从共享内存加载4个fp8e4m3_4（8x8矩阵的4行）到寄存器
    __device__ static inline void ldsm4(fp8e4m3_4& dst1, fp8e4m3_4& dst2, fp8e4m3_4& dst3, fp8e4m3_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 将4个fp8e4m3_4存储到共享内存
    __device__ static inline void stsm4(uint32_t dst, fp8e4m3_4& src1, fp8e4m3_4& src2, fp8e4m3_4& src3, fp8e4m3_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }

};

/**
 * @brief fp8e5m2格式（4个8位浮点）向量类型内存移动特化
 * @note Hopper/Blackwell架构支持的新浮点格式，动态范围更大
 */
template<> struct move<fp8e5m2_4> {
    /// 从共享内存加载4个fp8e5m2_4（8x8矩阵的4行）到寄存器
    __device__ static inline void ldsm4(fp8e5m2_4& dst1, fp8e5m2_4& dst2, fp8e5m2_4& dst3, fp8e5m2_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    /// 将4个fp8e5m2_4存储到共享内存
    __device__ static inline void stsm4(uint32_t dst, fp8e5m2_4& src1, fp8e5m2_4& src2, fp8e5m2_4& src3, fp8e5m2_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};
#endif

/* ----------   缓存策略常量 ---------- */

/**
 * @brief 缓存策略枚举
 * @note 用于控制L2缓存行为，优化内存访问模式
 */
enum cache_policy {
    NORMAL = 0,      ///< 正常缓存行为
    EVICT_FIRST = 1, ///< 优先驱逐策略，用于流式访问模式
    EVICT_LAST = 2   ///< 最后驱逐策略，用于重用性高的数据
};

/**
 * @brief 创建缓存策略对象
 * @tparam policy 缓存策略枚举值
 * @return 64位缓存策略描述符
 * @note 使用PTX指令创建缓存策略，用于后续内存操作
 */
template<cache_policy policy> __device__ inline uint64_t make_cache_policy() {
    uint64_t cache_policy_val;
    constexpr float fraction = 1.0f; ///< 分数参数，控制驱逐比例


    static_assert(policy == cache_policy::EVICT_FIRST || policy == cache_policy::EVICT_LAST, "Unexpected cache policy");
    if constexpr (policy == cache_policy::EVICT_FIRST) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
    else {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
    return cache_policy_val;
}

/* ----------   CLC（Cluster Launch Control）调度器操作 ---------- */

#ifdef DF_BLACKWELL

namespace clc {

/**
 * @brief CLC句柄结构体
 * @note 这是一个不透明类型，不应直接访问其内部值
 *       用于管理线程块集群调度
 */
struct handle {
    uint4 internal_value; ///< 内部值，不应直接访问
};

/**
 * @brief CLC调度结果结构体
 */
struct result {
    uint32_t success; ///< 调度是否成功（1成功，0失败）
    uint32_t x;       ///< 调度的线程块ID的x分量
    uint32_t y;       ///< 调度的线程块ID的y分量
    uint32_t z;       ///< 调度的线程块ID的z分量
};

/**
 * @brief 调度一个新的线程块
 * 
 * 必须在整个CTA集群中由单个线程调用。
 * 调用者必须在信号量上等待（使用tma::cluster::expect_bytes后接tma::cluster::wait）。
 * 句柄被多播到集群中的所有CTA，并通知集群中所有CTA的信号量。
 * 
 * @param h CLC句柄引用
 * @param sem 调用者将等待的信号量
 * @note 使用clusterlaunchcontrol指令进行异步调度
 */
__device__ static inline void schedule(handle &h, semaphore &sem) {
    asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
        : "memory"
    );
}

/**
 * @brief 查询调度操作的结果
 * 
 * 在失败后再次调用此函数是未定义行为。
 * 
 * @param h CLC句柄引用
 * @return 调度结果结构体
 */
__device__ static inline result query(handle &h) {
    result r;
    asm volatile(
        "{\n"
        ".reg .pred SUCCESS;\n"
        ".reg .b128 CLC_HANDLE;\n"
        "ld.shared.b128 CLC_HANDLE, [%4];\n"// 加载CLC句柄
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"// 查询是否被取消
        "selp.u32 %0, 1, 0, SUCCESS;\n"// 根据结果设置success字段
        "@!SUCCESS bra.uni DONE;\n"// 如果失败则跳转到结束
        "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, CLC_HANDLE;\n"// 获取线程块ID
        "fence.proxy.async.shared::cta;\n"// 代理异步操作的栅栏
        "DONE:\n"
        "}"
        : "=r"(r.success), "=r"(r.x), "=r"(r.y), "=r"(r.z)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
        : "memory"
    );
    return r;
}

} // namespace clc

#endif
/* ----------   Warp选举操作 ---------- */
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 选举warp领导者
 * @return 如果当前线程被选为warp领导者则返回true，否则返回false
 * @note 使用elect.sync指令在warp内选举一个领导者
 *       用于协调warp内的操作，如集体内存访问
 */
__device__ static inline bool elect_warp_leader() {
    uint32_t elected = 0;
    asm volatile(
        "{.reg .pred P;\n"
        " elect.sync _|P, %1;\n"// 执行选举，掩码为0xFFFFFFFF（所有线程参与）
        " selp.u32 %0, 1, 0, P;}\n"// 根据选举结果设置返回值
        : "+r"(elected)
        : "r"(0xFFFFFFFF)
    );
    return static_cast<bool>(elected);
}
#endif

} // namespace kittens