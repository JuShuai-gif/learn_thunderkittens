/**
 * @file
 * @brief ThunderKittens定义的所有组级（一个或多个warp）操作的聚合头文件
 * 
 * @details 该文件定义了线程组级别的操作抽象，支持不同大小的线程组（warp/warpgroup）
 *          提供内存操作、同步、MMA等高级原语
 */

#pragma once

#include <cuda/pipeline>// CUDA流水线原语

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../thread/thread.cuh" // 一些组内存操作依赖于底层的单线程范围操作
// 宏定义：检查是否为warp组（1个warp）
#define KITTENS_CHECK_WARP static_assert(GROUP_WARPS==1, "Warp (GROUP_WARPS=1) function called from a non-warp group.");
// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
// 宏定义：检查是否为warpgroup组（4个连续的warp）
#define KITTENS_CHECK_WARPGROUP static_assert(GROUP_WARPS==4, "Warpgroup (GROUP_WARPS=4) function called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within the group struct, so we declare them in advance.

// WGMMA依赖于一些模板结构，这些结构无法在group结构内特化，因此提前声明
#if defined(KITTENS_HOPPER)
#include "mma/base/base.cuh"
#endif

namespace kittens {
/*
 * @brief 线程组模板类，用于抽象一组线程（warp或warpgroup）
 * 
 * @details 在每个内核开始时使用 `using group_N = kittens::group<NUM_WORKERS>;` 
 *          该模板提供了线程组级别的并行原语和操作
 * 
 * @tparam _GROUP_WARPS 线程组中包含的warp数量（1表示warp，4表示warpgroup）
 */
template<int _GROUP_WARPS>
struct group {
static constexpr int GROUP_WARPS = _GROUP_WARPS; // 线程组中的warp数量
static constexpr int GROUP_THREADS = GROUP_WARPS * kittens::WARP_THREADS; // 线程组中的线程总数

    /**
     * @brief 获取线程在线程组内的lane ID
     * @return 线程在线程组内的相对ID（0到GROUP_THREADS-1）
     */
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }

    /**
     * @brief 获取线程在线程组内的warp ID
     * @return 线程在线程组内的warp索引（0到GROUP_WARPS-1）
     */
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }

    /**
     * @brief 获取线程组的全局ID
     * @return 当前线程组在整个线程块中的索引
     */
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }
   
/**
     * @brief 线程组内同步（使用显式barrier ID）
     * @param id 同步屏障ID
     */
__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

    /**
     * @brief warp内同步（无屏障，使用掩码）
     * @tparam MASK warp同步掩码，默认为所有线程（0xFFFFFFFF）
     * 
     * @note 只能由单个warp（GROUP_WARPS==1）调用
     */
template<uint32_t MASK=0xFFFFFFFF> __device__ static inline void sync() {
    static_assert(GROUP_WARPS==1, "barrier-less sync() can only be called by a single warp!");
    asm volatile("bar.warp.sync %0;\n" :: "n"(MASK));
}

    /**
     * @brief 到达同步屏障但不等待
     * @param id 同步屏障ID
     */
__device__ static inline void arrive(int id) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}

    // 包含各种功能模块
    #include "memory/memory.cuh"     // 内存操作
    #include "shared/shared.cuh"     // 共享内存操作
    #include "register/register.cuh" // 寄存器操作
    #include "mma/mma.cuh"           // 矩阵乘加操作
    #include "util/util.cuh"         // 工具函数
    // Hopper和Blackwell架构特定功能
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    /**
     * @brief 增加寄存器数量限制
     * @tparam n_reg 要增加的寄存器数量（必须是8的倍数）
     */
template<int n_reg> __device__ static inline void increase_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}

    /**
     * @brief 减少寄存器数量限制
     * @tparam n_reg 要减少的寄存器数量（必须是8的倍数）
     */
template<int n_reg> __device__ static inline void decrease_registers() {
    static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n_reg));
}

    /**
     * @brief 设置生产者寄存器配置（减少寄存器使用）
     */
__device__ static inline void producer_registers() { decrease_registers<24>(); }

    /**
     * @brief 设置消费者寄存器配置（增加寄存器使用）
     * @tparam NCWG 消费者工作组数量
     */
template<int NCWG> __device__ static inline void consumer_registers() { increase_registers<480/NCWG - 8*(NCWG>3) - 224*(NCWG==1)>(); }

    // ---- TMA操作 ----
    // 这些必须包含在这里因为：
    //   1. 我们希望与单线程操作保持并行范围（即tma::和tma::cluster）
    //   2. 不能在多个地方声明结构体
    //   3. 无法在结构体外使用命名空间
struct tma {
        #include "memory/tile/tma.cuh"          // TMA tile操作
        #include "memory/vec/tma.cuh"           // TMA向量操作
        #include "util/tma.cuh"                 // TMA工具函数
        // 集群范围TMA操作
        struct cluster {
            #include "memory/tile/tma_cluster.cuh"   // 集群TMA tile操作
            #include "memory/vec/tma_cluster.cuh"    // 集群TMA向量操作
            #include "util/tma_cluster.cuh"          // 集群TMA工具函数
        };
};

#endif

};
// 所有线程级别的操作命名空间
namespace everyone {
    /**
     * @brief 线程块级别的同步
     * @param id 同步屏障ID
     */
    __device__ static inline void sync(int id) {
        asm volatile("bar.sync %0;\n" :: "r"(id));
    }

    // 集群级别的同步函数
    namespace tma {
        namespace cluster {
            /**
             * @brief 对齐的集群屏障到达操作（集群中所有线程必须调用）
             */
            __device__ static inline void arrive_aligned() {
                asm volatile ("barrier.cluster.arrive.release.aligned;\n");
            }
            
            /**
             * @brief 对齐的集群屏障等待操作
             */
            __device__ static inline void wait_aligned() {
                asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
            }
            
            /**
             * @brief 完整的集群同步操作（到达+等待）
             */
            __device__ static inline void sync() {
                arrive_aligned();
                wait_aligned();
            }
        }
    }
};

// 常用线程组类型别名
using warp = group<1>;      // warp范围（单个warp），用于大多数Hopper之前的GPU和大多数寄存器操作
using warpgroup = group<4>; // warpgroup范围（4个连续warp），用于Hopper架构的特殊操作

}  // namespace kittens