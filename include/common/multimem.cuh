/**
 * @file multimem.cuh
 * @brief Multi-memory operations for NVIDIA GPUs with weak/strong memory model support
 * 
 * =============================================================================
 * 背景说明
 * =============================================================================
 * 
 * 本文件封装了 NVIDIA GPU 的 multi-memory 指令，这些是专门用于在全局内存
 * (global memory) 上执行原子操作的硬件指令。与传统的 CUDA atomicAdd 等
 * 函数相比，multi-memory 指令具有更高的吞吐量和更低的延迟。
 * 
 * Multi-memory 指令的核心优势:
 * - 可以在单个指令中同时完成"读取-修改-写入"操作
 * - 支持多种数据类型 (int, uint, float, float2, bf16, half 等)
 * - 支持三种归约操作: ADD (加法), MIN (最小值), MAX (最大值)
 * - 通过 weak/strong 内存模型提供灵活的一致性控制
 * 
 * =============================================================================
 * Weak vs Strong 内存模型详解
 * =============================================================================
 * 
 * GPU 程序设计中一个核心挑战是如何在高性能和内存一致性之间取得平衡。
 * CUDA 提供了多种内存模型，本文件实现了其中的 weak 和 strong 两种。
 * 
 * 
 * ------------------
 * WEAK (弱内存模型)
 * ------------------
 * 
 * 核心理念: "相信程序员" - 硬件不强制排序，最大限度提升性能
 * 
 * 指令格式:
 *   - 加载: multimem.ld_reduce.weak.global.add/min/max.<type>
 *   - 存储: multimem.st.weak.global.<type>
 * 
 * 特点:
 *   1. 不插入任何内存栅栏 (memory fence)
 *   2. 允许编译器/硬件自由重排内存访问顺序
 *   3. 不保证其他线程能看到最新的数据
 *   4. 性能最高，但存在数据竞争风险
 * 
 * 适用场景:
 *   - 多个线程完全独立地访问不同的内存位置
 *   - 无跨线程数据依赖的并行计算
 *   - 简单的并行归约 (每个线程处理自己专属的输出位置)
 *   - 对数据一致性没有要求的场景
 * 
 * 性能优势:
 *   - 减少内存延迟等待
 *   - 增加指令级并行度 (ILP)
 *   - 降低 GPU 核心的空闲时间
 * 
 * 使用示例:
 *   // 多个线程各自累加到不同的计数器
 *   multimem::ld_reduce<reduce_op::ADD, memory_model::WEAK>(sum[i], &global_buffer[i]);
 * 
 * 
 * ------------------
 * STRONG (强内存模型)
 * ------------------
 * 
 * 核心理念: "硬件保证一致性" - 通过 acquire/release 语义确保正确性
 * 
 * 指令格式:
 *   - 加载: multimem.ld_reduce.acquire.sys.global.add/min/max.<type>
 *   - 存储: multimem.st.release.sys.global.<type>
 *   - 归约: multimem.red.release.sys.global.add/min/max.<type>
 * 
 * 内存序语义解释:
 *   - acquire (获取): 读取时确保能看到所有之前的写操作
 *   - release (释放): 写入时确保所有之前的操作都完成
 *   - sys (系统级别): 跨越整个 GPU 和 CPU 内存系统
 * 
 * 特点:
 *   1. 插入适当的内存栅栏保证可见性
 *   2. 确保跨线程/跨 kernel 的数据正确传递
 *   3. 提供happens-before关系保证
 *   4. 有性能开销，但保证正确性
 * 
 * 适用场景:
 *   - 生产者-消费者模式 (一个线程写，另一个线程读)
 *   - 线程间有依赖关系的同步场景
 *   - 需要跨 thread block 传递数据
 *   - 需要与 CPU 端进行数据同步
 * 
 * 使用示例:
 *   // 生产者写入数据后，消费者使用 STRONG 读取确保看到最新值
 *   multimem::st<memory_model::STRONG>(&shared_data, value);  // 生产者
 *   // ... 一些同步操作 ...
 *   multimem::ld_reduce<reduce_op::ADD, memory_model::STRONG>(result, &shared_data);  // 消费者
 * 
 * 
 * =============================================================================
 * Weak vs Strong 选择指南
 * =============================================================================
 * 
 * 选择建议:
 * 
 * 1. 默认选择 WEAK:
 *    - 因为性能差异可能高达 2-10 倍
 *    - 简单并行代码通常不需要同步
 * 
 * 2. 考虑使用 STRONG 当:
 *    - 多个线程读写同一内存位置
 *    - 存在跨线程的数据依赖
 *    - 需要在 kernel 之间传递数据
 *    - 与 CPU 端有数据交互
 *    - 使用 shared memory + global memory 混合时
 * 
 * 3. 性能对比 (典型值):
 *    - WEAK: 100% 性能 (基准)
 *    - STRONG: 20-80% 性能 (取决于硬件和访问模式)
 * 
 * 4. 调试建议:
 *    - 先用 WEAK 实现功能
 *    - 如果结果不正确，再考虑改用 STRONG
 *    - 最终代码中只有必要的地方使用 STRONG
 * 
 * =============================================================================
 * 函数说明
 * =============================================================================
 * 
 * 本文件为每种数据类型提供三个核心操作:
 * 
 * 1. ld_reduce<Op, M>(dst, src)
 *    - 功能: 原子地读取内存值，执行归约操作，结果写入 dst
 *    - 语义: dst = dst OP *src (但实际上是 atomic_fetch_op(dst, *src))
 *    - 参数:
 *        - Op: reduce_op::ADD, reduce_op::MIN, reduce_op::MAX
 *        - M: memory_model::WEAK 或 memory_model::STRONG
 *    - 注意: 浮点数 (float, bf16, half) 仅支持 ADD 操作
 * 
 * 2. st<M>(dst, src)
 *    - 功能: 将 src 的值存储到 dst 指向的内存
 *    - 参数:
 *        - M: memory_model::WEAK 或 memory_model::STRONG
 *    - 注意: STRONG 模式下会执行 release 语义
 * 
 * 3. red<Op>(dst, src)
 *    - 功能: 原子地将 src 加到 *dst 上 (仅支持 release 模式)
 *    - 语义: *dst = *dst OP src
 *    - 参数:
 *        - Op: reduce_op::ADD, reduce_op::MIN, reduce_op::MAX
 *    - 注意: 只有 release 模式，没有 weak 版本
 * 
 * =============================================================================
 * 使用示例
 * =============================================================================
 * 
 * // 1. 简单的并行累加 (使用 WEAK，性能最佳)
 * __global__ void parallel_sum(const float* input, float* output, int n) {
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (tid < n) {
 *         multimem::ld_reduce<reduce_op::ADD, memory_model::WEAK>(output[0], &input[tid]);
 *     }
 * }
 * 
 * // 2. 生产者-消费者模式 (使用 STRONG 确保正确性)
 * __global__ void producer_consumer(float* shared_data, bool* ready) {
 *     // 生产者
 *     shared_data[0] = compute_value();
 *     multimem::st<memory_model::STRONG>(ready, true);  // release
 *     
 *     // 消费者
 *     bool is_ready;
 *     multimem::ld_reduce<reduce_op::ADD, memory_model::STRONG>(
 *         *(int*)&is_ready, ready);  // acquire
 *     if (is_ready) {
 *         process(shared_data[0]);
 *     }
 * }
 * 
 * =============================================================================
 * 
 * @see NVIDIA CUDA Programming Guide - Multi-Memory Operations
 * @see PTX ISA Documentation - Multi-Memory Instructions
 * @see CUDA C++ Programming Guide - Memory Model
 */

#pragma once

namespace kittens{

/**
 * @brief Reduce operations supported by multi-memory instructions
 * 
 * 定义了 multi-memory 指令支持的归约操作类型。
 * 
 * 归约操作说明:
 * - ADD (加法): 最常用，主要用于累加计数、求和等场景
 *   * 整数: dst = dst + *src (原子操作)
 *   * 浮点数: dst = dst + *src (使用 f32 累加器避免精度损失)
 * 
 * - MIN (最小值): 用于维护最小值，如找到最优解、统计最小延迟等
 *   * dst = min(dst, *src)
 *   * 注意: 浮点数 MIN/MAX 可能存在 NaN 处理问题
 * 
 * - MAX (最大值): 用于维护最大值，如找到最大概率、峰值等
 *   * dst = max(dst, *src)
 *   * 注意: 浮点数 MIN/MAX 可能存在 NaN 处理问题
 * 
 * @note 浮点数类型 (float, float2) 仅支持 ADD，因为 MIN/MAX 对浮点数
 *       的语义不够明确 (涉及 NaN 和 -0.0 的处理)
 */
enum class reduce_op{
    ADD = 0,  // 加法/累加操作 (累加计数器, 求和等)
    MIN = 1, // 最小值操作 (原子最小值)
    MAX = 2  // 最大值操作 (原子最大值)
};

/**
 * @brief Memory model for multi-memory operations
 * 
 * 控制 multi-memory 指令的内存一致性保证级别。这是 GPU 程序设计中
 * 最重要的选择之一，直接影响正确性和性能。
 * 
 * =============================================================================
 * 深入理解 Weak vs Strong
 * =============================================================================
 * 
 * CUDA/GPU 内存模型基于以下概念:
 * 
 * 1. 内存一致性 (Memory Coherence)
 *    - 强一致性: 任何线程看到的内存值都是相同的
 *    - 弱一致性: 只在同步点保证一致性
 * 
 * 2. 内存序 (Memory Ordering)
 *    - 程序顺序: 代码中语句的执行顺序
 *    - 观察顺序: 其他线程能看到这个线程写操作的顺序
 *    - 在弱一致性模型下，这两者可能不同
 * 
 * 3. 内存栅栏 (Memory Fence)
 *    - 用于强制内存访问的顺序性
 *    - strong 模式自动插入栅栏
 *    - weak 模式需要程序员手动控制
 * 
 * =============================================================================
 * WEAK (弱内存模型) 详解
 * =============================================================================
 * 
 * PTX 指令: multimem.ld_reduce.weak / multimem.st.weak
 * 
 * 行为:
 *   - 不插入任何内存栅栏
 *   - 允许指令重排
 *   - 允许合并读写操作
 *   - 不保证跨线程的可见性
 * 
 * 优势:
 *   - 性能最佳 (无额外同步开销)
 *   - 延迟最低
 *   - 指令吞吐量最高
 * 
 * 风险:
 *   - 如果存在数据竞争，结果不确定
 *   - 可能读到过期数据
 *   - 多线程写入同一位置可能丢失更新
 * 
 * 正确使用场景:
 *   - 每个线程写入/读取完全独立的内存位置
 *   - 使用归约操作时，每个线程操作不同的目标位置
 *   - 已通过其他机制 (如 warp vote, shuffle) 保证同步
 * 
 * 代码示例:
 *   // 多个线程各自累加到不同的位置 - 安全
 *   __global__ void safe_reduce(const float* input, float* output, int n) {
 *       int i = blockIdx.x * blockDim.x + threadIdx.x;
 *       if (i < n) {
 *           // 每个线程操作 output[i]，互不干扰
 *           multimem::ld_reduce<reduce_op::ADD, memory_model::WEAK>(
 *               output[i], &input[i]);
 *       }
 *   }
 * 
 * =============================================================================
 * STRONG (强内存模型) 详解
 * =============================================================================
 * 
 * PTX 指令: 
 *   - 加载: multimem.ld_reduce.acquire.sys
 *   - 存储: multimem.st.release.sys
 *   - 归约: multimem.red.release.sys
 * 
 * 语义解释:
 *   - acquire: 获取语义，确保读取前能看到所有之前的写操作
 *     * 相当于在读取前插入一个内存栅栏
 *     * 保证 happens-before 关系
 *   - release: 释放语义，确保写入后所有操作都完成
 *     * 相当于在写入后插入一个内存栅栏
 *     * 保证 happens-after 关系
 *   - sys: 系统级别作用域
 *     * 跨越 GPU 和 CPU 内存空间
 *     * 影响所有线程和所有内存地址
 * 
 * 优势:
 *   - 数据一致性保证
 *   - 跨线程/跨 kernel 正确性
 *   - 与 CPU 端内存同步
 * 
 * 代价:
 *   - 额外内存栅栏开销
 *   - 降低指令级并行度
 *   - 可能增加内存延迟
 * 
 * 正确使用场景:
 *   - 生产者-消费者模式
 *   - 线程间有共享数据依赖
 *   - 需要 CPU 端参与同步
 * 
 * 代码示例:
 *   // 生产者写入完成后，消费者读取确保看到最新数据
 *   __global__ void producer(float* data, int* ready) {
 *       if (threadIdx.x == 0) {
 *           data[0] = compute();                    // 计算
 *           multimem::st<memory_model::STRONG>(ready, 1); // release 通知
 *       }
 *   }
 *   __global__ void consumer(float* data, int* ready, float* result) {
 *       int r;
 *       multimem::ld_reduce<reduce_op::ADD, memory_model::STRONG>(
 *           r, ready);  // acquire 检查
 *       if (r) {
 *           result[0] = data[0];  // 读取数据
 *       }
 *   }
 * 
 * =============================================================================
 * 选择决策表
 * =============================================================================
 * 
 * 问自己以下问题:
 * 
 * Q1: 多个线程是否访问同一个内存位置?
 *    是 → STRONG
 *    否 → 继续
 * 
 * Q2: 是否需要跨 thread block 或 kernel 传递数据?
 *    是 → STRONG
 *    否 → 继续
 * 
 * Q3: 是否与 CPU 端有数据交互?
 *    是 → STRONG
 *    否 → 继续
 * 
 * Q4: 是否使用其他同步机制 (barrier, lock)?
 *    是 → 可能 WEAK (取决于具体场景)
 *    否 → WEAK
 * 
 * 如果不确定，默认使用 WEAK，性能优先。
 * 如果结果不正确，再尝试 STRONG。
 * 
 * @note 默认使用 WEAK 以获得最佳性能
 * @note STRONG 模式只在需要数据同步时使用，避免不必要的一致性开销
 */
enum class memory_model{
    WEAK = 0,   // 弱内存模型：高绩效，无自动同步，但需程序员保证无数据竞争
    STRONG = 1  // 强内存模型：提供 acquire/release 语义，保证数据一致性
};

 /**
 * @brief Multi-memory operations for 32-bit unsigned integer (uint)
 * 
 * 提供对 unsigned int 类型的 multi-memory 操作支持。
 * 功能与 multimem<int> 完全相同，参见 multimem<int> 的详细说明。
 * 
 * 数据类型说明:
 * - u32: 32位无符号整数
 * - 范围: 0 到 4,294,967,295
 * - 溢出行为: 取决于硬件，通常为 wrap-around
 * 
 * @note 与 int 版本相同，参见 multimem<int> 的说明
 */
template <>
struct multimem<uint>
{
    /**
     * @brief 原子加载并归约操作 (ld_reduce)
     * 
     * 功能与 multimem<int>::ld_reduce 相同，参见其详细说明。
     * 
     * @tparam Op 归约操作类型: ADD, MIN, MAX
     * @tparam M 内存模型: WEAK 或 STRONG
     * @param dst 目标地址
     * @param src 源地址
     */
    template <reduce_op Op,memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(int& dst,const int* src){
        if constexpr(Op == reduce_op::ADD)
        {
            if constexpr(M == memory_model::WEAK)
            {
                asm volatile("multimem.ld_reduce.weak.global.add.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
}
    }

    /**
     * @brief 原子归约操作 (red)
     * 
     * 将 half_2 值与全局内存执行原子归约操作。
     * 注意: 这个函数只支持 ADD 操作，且只有 release 模式。
     * 
     * @tparam Op 归约操作类型: 只能是 ADD
     * @param dst 目标地址
     * @param src 源值
     * 
     * PTX 指令:
     *   multimem.red.release.sys.global.add.f16x2
     */
    template <reduce_op Op>
    __device__ static inline void red(half *dst, const half &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f16 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.f16 [%0], %1;"
                : : "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        }
    }
};

/**
 * @brief Multi-memory operations for half x2 (两个 half/f16 向量)
 * 
 * half_2 是一个包含两个 half 的结构体，用于 SIMD 操作。
 * 
 * 通过在单个指令中处理两个 half，可以:
 * - 提高内存带宽利用率
 * - 增加计算吞吐率
 * - 适合向量化的深度学习操作
 * 
 * @note 用于同时处理两个 half，提高内存带宽利用率
 * @note 支持 ADD, MIN, MAX 操作
 * @see multimem<half>::ld_reduce 的说明
 */
template <>
struct multimem<half_2> {
    /**
     * @brief 原子加载并归约操作 (ld_reduce)
     * 
     * 同时对 half_2 的两个元素执行原子加载并归约操作。
     * 
     * @tparam Op 归约操作类型: ADD, MIN, MAX
     * @tparam M 内存模型: WEAK 或 STRONG
     * @param dst 目标地址
     * @param src 源地址
     */
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(half_2 &dst, const half_2 *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.acc::f32.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(half_2 *dst, const half_2 &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.f16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(half_2 *dst, const half_2 &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f16_2 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                : : "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
};

}
