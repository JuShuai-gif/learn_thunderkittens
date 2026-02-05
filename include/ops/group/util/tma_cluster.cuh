/**
 * @file
 * @brief 集群范围内的TMA内存操作工具函数
 * @note TMA: Tensor Memory Accelerator, NVIDIA Hopper架构的硬件内存搬运引擎
 */

/**
* @brief 在集群范围内等待信号量达到指定阶段
* 
* @details 使用忙等待循环，直到指定阶段的信号量条件满足
* 
* @param bar 信号量引用
* @param kPhaseBit 用于信号量的阶段位
* @warning 这是一个阻塞调用，可能导致线程挂起
*/
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 将通用指针转换为共享内存指针

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
* @brief 非阻塞尝试等待信号量达到指定阶段
* 
* @details 尝试等待一次，立即返回结果，不会阻塞线程
* 
* @param bar 信号量引用
* @param kPhaseBit 用于信号量的阶段位
* @return bool 是否成功等待（true:成功, false:失败）
*/
__device__ static inline bool try_wait(semaphore &bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); // 将通用指针转换为共享内存指针
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
        : "memory" // 告诉编译器内存可能被修改
    );

    return static_cast<bool>(success);// 将整数结果转换为bool类型
}

/**
* @brief 设置信号量期望接收的字节数（组播指令）
* 
* @details 仅warp中的第一个线程(laneid=0)执行此操作。对于组播TMA传输，
*          需要在接收方设置期望的字节数，以便信号量能够正确计数。
* 
* @note 分布式共享内存操作可以使用普通的tma::expect后接wait，详见dsmem单元测试
* 
* @param bar 信号量引用
* @param bytes 期望的字节数
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {// 仅warp中的第一个线程执行
        ::kittens::tma::cluster::expect_bytes(bar, bytes);// 调用底层库函数
    }
}

/**
* @brief 设置信号量期望接收的字节数（指定目标CTA）
* 
* @param bar 信号量引用
* @param bytes 期望的字节数
* @param dst_cta 目标线程块ID
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    if(laneid() == 0) {// 仅warp中的第一个线程执行
        ::kittens::tma::cluster::expect_bytes(bar, bytes, dst_cta);// 调用底层库函数
    }
}
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @tparam T The type of the data to be stored at the semaphore.
* @param semaphore Reference to the semaphore variable.
*/
/**
* @brief 设置信号量期望接收的字节数（模板版本）
* 
* @tparam T 数据类型
* @tparam args 其他数据类型（变长模板参数）
* @param bar 信号量引用
* @param _1 第一个数据引用
* @param _2 其他数据引用（变长参数包）
* 
* @note 自动计算所有参数的总字节数
*/
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);// 计算模板参数的总字节数并设置
}

/**
* @brief 在集群范围内到达信号量
* 
* @details 标记线程已到达mbarrier，用于减少信号量计数器
*          仅warp中的第一个线程(laneid=0)执行此操作
* 
* @param bar 信号量引用
* @param dst_cta 目标线程块ID
* @param count 到达计数（默认为1）
*/
__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    if(laneid() == 0) {// 仅warp中的第一个线程执行
        ::kittens::tma::cluster::arrive(bar, dst_cta, count);// 调用底层库函数
    }
}

/**
* @brief 异步存储操作（通用版本）
* 
* @details 使用TMA执行异步内存存储操作
*          仅warp中的第一个线程(laneid=0)执行此操作
* 
* @param dst 目标地址指针
* @param src 源地址指针
* @param dst_cta 目标线程块ID
* @param size_bytes 传输字节数
* @param bar 信号量引用（用于同步）
*/
__device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes, semaphore& bar) {
    if(laneid() == 0) {// 仅warp中的第一个线程执行
        ::kittens::tma::cluster::store_async(dst, src, dst_cta, size_bytes, bar);// 调用底层库函数
    }
}

/**
* @brief 异步存储操作（模板版本）
* 
* @tparam T 数据类型
* @param dst_ 目标数据引用
* @param src_ 源数据引用
* @param dst_cta 目标线程块ID
* @param bar 信号量引用（用于同步）
* 
* @note 自动计算数据类型T的字节大小
*/
template<typename T>
__device__ static inline void store_async(T &dst_, T &src_, int dst_cta, semaphore& bar) {
    // 调用通用版本，自动计算类型大小
    store_async((void*)&dst_, (void*)&src_, dst_cta, size_bytes<T>, bar);
}
