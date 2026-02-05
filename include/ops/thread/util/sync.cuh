/**
 * @file
 * @brief 单线程同步操作的各种工具函数。
 *        提供单线程上下文中的同步原语，包括信号量、屏障和异步操作等待。
 */

#pragma once

namespace kittens {
/**
 * @brief 信号量结构体（mbarrier封装）
 * @note 这是一个不透明类型，不应直接访问其内部值
 *       实际使用mbarrier硬件原语进行同步
 */
struct semaphore {
private:
    uint64_t value;///< 内部值，不应直接访问
}; // note that this is an opaque type, so the value should not be accessed directly.

/**
 * @brief 屏障结构体
 * @tparam num_warps warp数量
 * @note 包装CUDA硬件屏障，提供屏障同步功能
 */
template<int num_warps> struct barrier {
    int barrier_id; ///< 屏障标识符
    
    /**
     * @brief 构造函数
     * @param _id 屏障ID
     */
    __device__ __forceinline__ barrier(int _id) : barrier_id(_id) {}
    
    /**
     * @brief 数组访问操作符，用于获取偏移屏障
     * @param i 偏移量
     * @return 新的屏障对象
     */
    __device__ __forceinline__ barrier operator[](int i) {
        return barrier(barrier_id + i);
    }
};

/**
 * @brief 初始化同步信号量（mbarrier）
 * 
 * 此函数设置用于块内线程在异步操作期间同步的信号量。
 * 使用线程计数初始化信号量。
 * 
 * @param bar 信号量变量引用
 * @param thread_count 线程计数器
 * @param transaction_count 事务计数器，默认为0
 * @note 由每个线程执行，使用mbarrier.init指令初始化信号量
 */
__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}
/**
 * @brief 使mbarrier失效
 * 
 * @param bar 信号量变量引用
 * @note 由每个线程执行，使用mbarrier.inval指令使信号量失效
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    asm volatile (
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

/**
 * @brief 到达信号量
 * 
 * 标记warp到达mbarrier
 * 
 * @param sem 信号量变量引用
 * @note 由每个线程执行，使用mbarrier.arrive.release指令标记到达
 */
__device__ static inline void arrive(semaphore& sem) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_ptr)
        : "memory"
    );
}

/**
 * @brief 到达屏障
 * 
 * @tparam num_warps warp数量
 * @param bar 屏障对象
 * @note 使用bar.arrive指令标记线程到达屏障
 */
template<int num_warps> __device__ static inline void arrive(barrier<num_warps> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 到达信号量并指定计数
 * 
 * 标记warp到达mbarrier并指定到达计数
 * 
 * @param sem 信号量变量引用
 * @param count 到达计数
 * @note Hopper/Blackwell架构专用，支持计数到达
 */
__device__ static inline void arrive(semaphore& sem, uint32_t count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}
#endif


/**
 * @brief 等待信号量达到指定阶段
 * 
 * @param sem 信号量变量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @note 使用忙等待循环等待信号量，直到指定阶段完成
 */
__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    // Hopper/Blackwell架构使用try_wait指令
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#else
    // 早期架构使用test_wait指令
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // 在Hopper之前的架构上等待几纳秒，以节省指令发射槽
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 尝试等待信号量（非阻塞版本）
 * 
 * @param sem 信号量变量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @return 如果信号量已达到指定阶段返回true，否则返回false
 * @note Hopper/Blackwell架构专用，使用try_wait指令进行非阻塞检查
 */
__device__ static inline bool try_wait(semaphore &sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    uint32_t success;

    asm volatile(
        "{\n"
        ".reg .pred P1; \n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
        "selp.b32 %0, 1, 0, P1; \n"
        "}\n"
        : "=r"(success)
        : "r"(mbar_ptr), "r"(kPhaseBit)
        : "memory"
    );

    return static_cast<bool>(success);
}
#endif

/**
 * @brief 谨慎等待信号量（带超时检测）
 * 
 * @param sem 信号量变量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @note Hopper/Blackwell架构上使用时钟检测避免死锁，超时触发trap
 *       早期架构使用普通等待
 */
__device__ static inline void careful_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"
        "mov.b64                   start_clock, %clock64;\n"
        ".reg .pred                P_CLOCK;\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "mov.b64                   current_clock, %clock64;\n"
        "sub.u64                   current_clock, current_clock, start_clock;\n"
        "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n"
        "@P_CLOCK                  trap;\n"// 如果等待超过100万个时钟周期，触发trap
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // 在Hopper之前的架构上等待几纳秒，以节省指令发射槽
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

/**
 * @brief 测试信号量是否达到指定阶段
 * 
 * @param sem 信号量变量引用
 * @param kPhaseBit 阶段位，标识要测试的信号量阶段
 * @return 如果信号量已达到指定阶段返回1，否则返回0
 * @note 使用mbarrier.test_wait指令测试信号量状态，不进行等待
 */
__device__ static inline int test_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    int result;
    asm volatile (
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0,1,0,P1;"
        "}\n"
        : "=r"(result)
        : "r"(mbar_ptr), "r"(kPhaseBit)
    );
    return result;
}

/**
 * @brief 到达信号量并等待
 * 
 * @param sem 信号量变量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @note 组合操作：先到达信号量，然后等待指定阶段
 */
__device__ static inline void arrive_and_wait(semaphore& sem, int kPhaseBit) {
    arrive(sem);
    wait(sem, kPhaseBit);
}

/**
 * @brief 到达屏障并等待所有线程
 * 
 * @tparam num_warps warp数量
 * @param bar 屏障对象
 * @note 使用bar.sync指令执行完整的屏障同步
 */
template<int num_warps> __device__ static inline void arrive_and_wait(barrier<num_warps> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

/**
 * @brief 等待异步加载完成
 * 
 * @tparam N 等待的异步操作组数量，默认0表示等待所有
 * @note 用于完成（非TMA）异步加载，然后执行warp同步
 */
template<int N=0> __device__ static inline void load_async_wait() { // 用于完成（非TMA）异步加载
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);// 等待所有异步操作
    } else {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));// 等待指定组
    }
    __syncwarp();// warp内同步
}


// 以下代码用于计算类型大小，主要用于共享内存块和向量
namespace detail {
/**
 * @brief 类型大小信息模板
 * @tparam T 类型
 * @note 提供类型的大小信息，针对共享内存块和向量有特化
 */    
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);///< 字节大小
};

/**
 * @brief 共享内存块的大小信息特化
 * @tparam ST 共享内存块类型
 */
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements; ///< 元素数量
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);///< 总字节数
};

/**
 * @brief 共享向量的大小信息特化
 * @tparam SV 共享向量类型
 */
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;///< 向量长度
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);///< 总字节数
};
}

/**
 * @brief 计算多个类型的总字节数（递归终止条件）
 */
template<typename... Args>             inline constexpr uint32_t size_bytes             = 0; // base case

/**
 * @brief 计算多个类型的总字节数（递归计算）
 * @tparam T 第一个类型
 * @tparam Args 其余类型
 */
template<typename T, typename... Args> inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

/* ----------   TCGEN05同步操作（Blackwell架构专用） ---------- */

#if defined(DF_BLACKWELL)
/**
 * @brief 线程同步前的张量操作栅栏
 * @note Blackwell架构专用，在thread_sync之前插入张量操作的栅栏
 */
__device__ static inline void tensor_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}

/**
 * @brief 线程同步后的张量操作栅栏
 * @note Blackwell架构专用，在thread_sync之后插入张量操作的栅栏
 */
__device__ static inline void tensor_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}


/**
 * @brief 等待张量加载操作完成
 * @note Blackwell架构专用，等待张量加载操作对齐完成
 */
__device__ inline static void tensor_load_wait() {
   asm volatile("tcgen05.wait::ld.sync.aligned;");
}


/**
 * @brief 等待张量存储操作完成
 * @note Blackwell架构专用，等待张量存储操作对齐完成
 */
__device__ inline static void tensor_store_wait() {
   asm volatile("tcgen05.wait::st.sync.aligned;"); 
}

#endif

/* ----------   多GPU同步操作 ---------- */

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief 向指定设备发送信号
 * @tparam NUM_DEVICES 设备数量
 * @param barrier 屏障数组引用
 * @param idx 坐标索引
 * @param dst_dev_idx 目标设备索引
 * @param val 信号值
 * @note 使用原子加法向指定设备的屏障位置发送信号
 */
template <int NUM_DEVICES>
__device__ static inline void signal(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dst_dev_idx, const int val
) {
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dst_dev_idx][idx]), "r"(val) : "memory");
}

/**
 * @brief 向所有设备发送信号
 * @tparam NUM_DEVICES 设备数量
 * @param barrier 屏障数组引用
 * @param idx 坐标索引
 * @param val 信号值
 * @note 使用多设备原子加法向所有设备的屏障位置发送信号
 */
template <int NUM_DEVICES>
__device__ static inline void signal_all(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int val
) {
    asm volatile("{multimem.red.release.sys.global.add.s32 [%0], %1;}" :: "l"(barrier.mc_ptr_at(idx)), "r"(val) : "memory");
}

/**
 * @brief 等待指定设备的信号达到期望值
 * @tparam NUM_DEVICES 设备数量
 * @param barrier 屏障数组引用
 * @param idx 坐标索引
 * @param dev_idx 设备索引
 * @param expected 期望值
 * @note 使用忙等待循环检查指定设备屏障位置的值
 */
template <int NUM_DEVICES>
__device__ static inline void wait(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dev_idx, const int expected
) {
    int val;
    do {
        asm volatile("{ld.relaxed.sys.global.s32 %0, [%1];}" : "=r"(val) : "l"(&barrier[dev_idx][idx]) : "memory");
    } while (val != expected);
}


/**
 * @brief 与所有设备进行屏障同步
 * @tparam NUM_DEVICES 设备数量
 * @param barrier 屏障数组引用
 * @param idx 坐标索引
 * @param dev_idx 当前设备索引
 * @note 1. 向所有设备发送信号
 *       2. 等待当前设备接收所有信号
 *       3. 重置当前设备屏障值
 */
template <int NUM_DEVICES>
__device__ static inline void barrier_all(
    const barrier_t<NUM_DEVICES> &barrier, const coord<ducks::default_type> &idx, const int dev_idx
) {
    signal_all(barrier, idx, 1);// 向所有设备发送信号
    wait(barrier, idx, dev_idx, NUM_DEVICES);// 等待当前设备接收所有信号
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(&barrier[dev_idx][idx]), "r"(-NUM_DEVICES) : "memory");// 重置当前设备屏障值
}

#endif

} // namespace kittens
