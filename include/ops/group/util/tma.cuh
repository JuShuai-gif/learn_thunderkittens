/**
 * @file
 * @brief 组TMA内存操作的各种工具函数。
 *        提供张量内存访问（TMA）操作的同步和内存管理功能。
 */

/* ----------   异步加载的屏障函数 ---------- */

/**
 * @brief 设置信号量期望的字节数
 * 
 * 此函数设置在信号量处期望的字节数，由warp中的第一个线程执行。
 * 它将信号量指针转换为共享内存通用指针，并使用内联汇编指令设置期望的字节数。
 * 
 * @param bar 信号量引用
 * @param bytes 期望的字节数
 * @note 由lane 0线程执行，使用TMA期望字节数设置
 */
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    if(laneid() == 0) {
        ::kittens::tma::expect_bytes(bar, bytes);
    }
}
/**
 * @brief 设置信号量期望的字节数（基于类型大小）
 * 
 * 此函数在事务到达前设置mbarrier期望的字节数。
 * 使用模板参数自动计算总字节数。
 * 
 * @tparam T 第一个参数的类型
 * @tparam args 其他参数的类型
 * @param bar 信号量引用
 * @param _1 第一个参数引用
 * @param _2 其他参数引用
 * @note 自动计算所有类型参数的总字节数并设置期望值
 */
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}

/* ----------   异步存储的同步函数 ---------- */

/**
 * @brief 提交之前异步TMA存储到组并执行它们
 * 
 * 使用cp.async.bulk.commit_group指令提交所有挂起的异步TMA存储操作
 * 使其成为当前组的一部分并开始执行
 */
__device__ static inline void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}
/**
 * @brief 等待之前提交的TMA存储组完成
 * 
 * @tparam N 剩余TMA存储组的最大数量，默认为0
 * @note 使用cp.async.bulk.wait_group指令等待异步存储操作完成
 *       通常用于确保存储操作完成后才能访问存储的数据
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
 * @brief 等待之前提交的TMA存储组完成从共享内存的读取
 * 
 * @tparam N 剩余TMA存储组的最大数量，默认为0
 * @note 使用cp.async.bulk.wait_group.read指令等待异步存储操作完成从共享内存的读取
 *       这确保存储操作从共享内存读取数据完成后，共享内存可以安全重用
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
