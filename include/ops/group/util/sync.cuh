/**
 * @file
 * @brief 组同步操作的各种工具函数。
 *        提供基于CUDA硬件原语的同步操作，包括异步加载等待、屏障同步和信号量操作。
 */

/**
 * @brief 等待异步加载完成（非TMA加载）
 * @tparam N 等待的异步操作组数量，默认0表示等待所有
 * @param bar_id 屏障ID，用于等待后执行同步
 * @note 该函数使用cp.async.wait_group指令等待异步加载完成，然后执行屏障同步
 */
template<int N=0> __device__ static inline void load_async_wait(int bar_id) { // 用于完成（非TMA）异步加载
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    sync(bar_id);
}

/**
 * @brief 等待异步加载完成（非TMA加载），使用warp级同步
 * @tparam N 等待的异步操作组数量，默认0表示等待所有
 * @note 该函数使用cp.async.wait_group指令等待异步加载完成，然后执行warp同步
 */
template<int N=0> __device__ static inline void load_async_wait() { // 用于完成（非TMA）异步加载
    KITTENS_CHECK_WARP
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    __syncwarp();
}

/**
 * @brief 到达屏障但不等待
 * @param bar 屏障对象，模板参数GROUP_WARPS指定warp数量
 * @note 使用bar.arrive指令标记线程到达屏障
 */
__device__ static inline void arrive(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(GROUP_WARPS*WARP_THREADS) : "memory");
}

/**
 * @brief 到达屏障并等待所有线程
 * @param bar 屏障对象，模板参数GROUP_WARPS指定warp数量
 * @note 使用bar.sync指令执行完整的屏障同步
 */
__device__ static inline void arrive_and_wait(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(GROUP_WARPS*WARP_THREADS) : "memory");
}

/**
 * @brief 初始化同步信号量（mbarrier）
 * @param bar 信号量引用
 * @param thread_count 线程计数器
 * @param transaction_count 事务计数器，默认为0
 * @note 由lane 0线程执行，使用mbarrier.init指令初始化信号量
 *       总计数为thread_count + transaction_count
 */
__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }
}

/**
 * @brief 使mbarrier失效
 * @param bar 信号量引用
 * @note 由lane 0线程执行，使用mbarrier.inval指令使信号量失效
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
        asm volatile (
            "mbarrier.inval.shared::cta.b64 [%0];\n"
            :: "r"(bar_ptr)
        );
    }
}

/**
 * @brief 到达信号量（mbarrier）
 * @param sem 信号量引用
 * @note 由lane 0线程执行，使用mbarrier.arrive.release指令标记到达
 *       使用release内存序保证之前的存储操作对后续等待线程可见
 */
__device__ static inline void arrive(semaphore& sem) {
    if(laneid() == 0) {
            uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
            asm volatile (
                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
            :
            : "r"(mbar_ptr)
            : "memory"
        );
    }
}

/**
 * @brief 到达屏障（通用版本）
 * @tparam num_warps warp数量
 * @param bar 屏障对象
 * @note 使用bar.arrive指令标记线程到达屏障
 */
template<int num_warps> __device__ static inline void arrive(barrier<num_warps> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
/**
 * @brief 到达信号量并指定计数（Hopper/Blackwell架构专用）
 * @param sem 信号量引用
 * @param count 到达计数
 * @note 由lane 0线程执行，使用mbarrier.arrive.release指令标记到达
 *       支持指定到达计数，用于更灵活的同步控制
 */
__device__ static inline void arrive(semaphore& sem, uint32_t count) {
    if(laneid() == 0) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile (
            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
            :
            : "r"(mbar_ptr), "r"(count)
            : "memory"
        );
    }
}
#endif

/**
 * @brief 等待信号量达到指定阶段
 * @param sem 信号量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @note 使用忙等待循环等待信号量，直到指定阶段完成
 *       在Hopper/Blackwell架构上使用mbarrier.try_wait指令
 *       在更早的架构上使用mbarrier.test_wait指令，并包含nanosleep以减少指令发射
 */
__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
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
 * @brief 尝试等待信号量达到指定阶段（非阻塞版本）
 * @param sem 信号量引用
 * @param kPhaseBit 阶段位，标识要等待的信号量阶段
 * @return 如果信号量已达到指定阶段返回true，否则返回false
 * @note Hopper/Blackwell架构专用，使用mbarrier.try_wait指令进行非阻塞检查
 */
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
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
 * @brief 测试信号量是否达到指定阶段
 * @param sem 信号量引用
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