#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens
{
namespace tma{
/**
 * @brief 设置信号量期望接收的字节数（线程块范围）
 * @details 用于异步内存传输，通知信号量期望接收指定字节数的数据
 * 
 * @param bar 信号量引用
 * @param bytes 期望的字节数
 */
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 转换为共享内存指针
    // 使用PTX指令设置期望的传输字节数
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}

/**
 * @brief 设置信号量期望接收的字节数（模板版本，自动计算字节数）
 * @tparam T 数据类型
 * @tparam args 其他数据类型（变长模板参数）
 * @param bar 信号量引用
 * @param _1 第一个数据引用
 * @param _2 其他数据引用（变长参数包）
 */
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);// 计算所有参数的总字节数
}
/**
 * @brief 提交异步存储组
 * @details 提交当前所有的异步存储操作，使其成为等待组的一部分
 */
__device__ static inline void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}
/**
 * @brief 等待异步存储操作完成
 * @tparam N 等待的组数（默认为0，表示等待所有未完成的组）
 */
template <int N=0>
__device__ static inline void store_async_wait() {
    asm volatile (
        "cp.async.bulk.wait_group %0;"
        :
        : "n"(N)
        : "memory"
    );
}
/**
 * @brief 等待异步存储操作完成并确保数据可读
 * @tparam N 等待的组数（默认为0，表示等待所有未完成的组）
 */
template <int N=0>
__device__ static inline void store_async_read_wait() {
    asm volatile (
        "cp.async.bulk.wait_group.read %0;"
        :
        : "n"(N)
        : "memory"
    );
}

/* ----------   集群范围操作  ---------- */

namespace cluster {

/**
 * @brief 在集群范围内等待信号量达到指定阶段
 * @details 使用忙等待循环，直到指定阶段的信号量条件满足
 * 
 * @param bar 信号量引用
 * @param kPhaseBit 用于信号量的阶段位
 */
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 转换为共享内存指针
    // 内联汇编实现忙等待
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"// 声明谓词寄存器P1
        "LAB_WAIT:\n"// 等待标签
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"// 尝试等待mbarrier
        "@P1                       bra.uni DONE;\n"// 如果成功(P1=true)，跳转到DONE
        "bra.uni                   LAB_WAIT;\n"// 否则继续等待
        "DONE:\n"// 完成标签
        "}\n"
        :: "r"(mbar_ptr),// 输入：mbarrier指针
        "r"(kPhaseBit)// 输入：阶段位
    );
}
/**
 * @brief 非阻塞尝试等待信号量达到指定阶段（集群范围）
 * @details 尝试等待一次，立即返回结果，不会阻塞线程
 * 
 * @param bar 信号量引用
 * @param kPhaseBit 用于信号量的阶段位
 * @return bool 是否成功等待（true:成功, false:失败）
 */
__device__ static inline bool try_wait(semaphore &bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 转换为共享内存指针
    uint32_t success;// 存储尝试结果
    // 内联汇编实现单次尝试等待
    asm volatile(
        "{\n"
        ".reg .pred P1; \n"// 声明谓词寄存器P1
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2; \n"// 尝试等待mbarrier
        "selp.b32 %0, 1, 0, P1; \n"// 根据谓词选择返回值(1或0)
        "}\n"
        : "=r"(success)// 输出：成功标志
        : "r"(mbar_ptr), "r"(kPhaseBit)// 输入：mbarrier指针和阶段位
        : "memory"// 告诉编译器内存可能被修改
    );

    return static_cast<bool>(success);// 将整数结果转换为bool类型
}

/**
 * @brief 带有超时检测的谨慎等待函数
 * @details 在等待信号量时检查时钟周期，超过阈值(1000000个周期)则触发trap
 *          用于检测可能出现的死锁或超时情况
 * 
 * @param bar 信号量引用
 * @param kPhaseBit 用于信号量的阶段位
 */
__device__ static inline void careful_wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 转换为共享内存指针

    // 内联汇编实现带超时检测的等待
    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"// 时钟寄存器
        "mov.b64                   start_clock, %clock64;\n"// 记录开始时钟
        ".reg .pred                P_CLOCK;\n"// 时钟谓词寄存器
        ".reg .pred                P1;\n"// 等待谓词寄存器
        "LAB_WAIT:\n"// 等待标签
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"// 尝试等待mbarrier
        "@P1                       bra.uni DONE;\n"// 如果成功，跳转到DONE
        "mov.b64                   current_clock, %clock64;\n"// 获取当前时钟
        "sub.u64                   current_clock, current_clock, start_clock;\n"// 计算耗时
        "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n" // 检查是否超时（1000000周期）
        "@P_CLOCK                  trap;\n"// 如果超时，触发trap
        "bra.uni                   LAB_WAIT;\n"// 否则继续等待
        "DONE:\n"// 完成标签
        "}\n"
        :: "r"(mbar_ptr),// 输入：mbarrier指针
        "r"(kPhaseBit)// 输入：阶段位
    );
}

/**
 * @brief 设置信号量期望接收的字节数（集群范围）
 * @details 用于异步内存传输，通知信号量期望接收指定字节数的数据
 *          特别适用于组播传输
 * 
 * @param bar 信号量引用
 * @param bytes 期望的字节数
 */
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));// 转换为共享内存指针
    // 使用PTX指令设置期望的传输字节数（集群范围）
    asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(bytes));
}

/**
 * @brief 设置信号量期望接收的字节数（指定目标线程块）
 * @details 针对特定目标线程块设置期望字节数，使用mapa指令计算目标线程块的信号量地址
 * 
 * @param bar 信号量引用
 * @param bytes 期望的字节数
 * @param dst_cta 目标线程块ID
 */
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); // 转换为共享内存指针
    uint32_t neighbor_mbar_addr;// 邻居线程块的信号量地址
    
    // 使用mapa指令计算目标线程块的共享内存地址
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    // 设置目标线程块信号量的期望字节数
    asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(neighbor_mbar_addr), "r"(bytes));
}
/**
 * @brief 设置信号量期望接收的字节数（模板版本，集群范围）
 * @details 自动计算所有参数的总字节数并设置
 * 
 * @tparam T 数据类型
 * @tparam args 其他数据类型（变长模板参数）
 * @param bar 信号量引用
 * @param _1 第一个数据引用
 * @param _2 其他数据引用（变长参数包）
 */
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);// 计算所有参数的总字节数
}


/**
 * @brief 在集群范围内到达信号量
 * @details 标记线程已到达mbarrier，用于减少信号量计数器
 * 
 * @param bar 信号量引用
 * @param dst_cta 目标线程块ID
 * @param count 到达计数（默认为1）
 */
__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); // 转换为共享内存指针
    uint32_t neighbor_mbar_addr;// 邻居线程块的信号量地址
    // 使用mapa指令计算目标线程块的共享内存地址    
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    // 到达目标线程块的信号量
    asm volatile (
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :
        : "r"(neighbor_mbar_addr), "r" (count)
        : "memory"
    );
}

/**
 * @brief 集群范围的异步存储操作（通用版本）
 * @details 使用TMA执行跨线程块的异步内存存储操作
 * 
 * @param dst 目标地址指针（在目标线程块的共享内存中）
 * @param src 源地址指针（在当前线程块的共享内存中）
 * @param dst_cta 目标线程块ID
 * @param size_bytes 传输字节数
 * @param bar 信号量引用（用于同步）
 */
__device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes, semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t mbarrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 转换信号量指针

    // 转换源地址和目标地址为共享内存指针

    // **************************************************
    // load from src to dst in different threadblocks
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
    // find dst addr in neighbor's cta

    // 使用mapa指令计算目标线程块中的目标地址
    // mapa指令文档：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa
    uint32_t neighbor_addr_dst;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_addr_dst)
        : "r"(dst_ptr), "r"(dst_cta)
    );
    // 使用mapa指令计算目标线程块中的信号量地址    
    uint32_t neighbor_addr_mbarrier = mbarrier_ptr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_addr_mbarrier)
        : "r"(mbarrier_ptr), "r"(dst_cta)
    );
    
    // 执行异步批量存储操作
    // cp.async指令文档：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");// 内存栅栏，确保之前的代理操作完成
    
    // 执行异步批量复制，从源地址到目标线程块的目标地址，使用信号量同步
    asm volatile (
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbarrier)
        : "memory"
    );
}
/**
 * @brief 集群范围的异步存储操作（模板版本）
 * @details 自动计算数据类型的大小，简化调用
 * 
 * @tparam T 数据类型
 * @param dst_ 目标数据引用（在目标线程块的共享内存中）
 * @param src_ 源数据引用（在当前线程块的共享内存中）
 * @param dst_cta 目标线程块ID
 * @param bar 信号量引用（用于同步）
 */
template<typename T>
__device__ static inline void store_async(T &dst_, T &src_, int dst_cta, semaphore& bar) {
    store_async((void*)&dst_, (void*)&src_, dst_cta, size_bytes<T>, bar);
}

} // namespace cluster
} // namespace tma
} // namespace kittens



























