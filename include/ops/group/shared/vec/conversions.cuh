/**
 * @file
 * @brief 共享向量（shared memory）的组转换操作
 * 
 * 提供在共享内存中不同向量之间进行数据复制和类型转换的功能。
 * 与寄存器向量转换不同，共享向量转换基于共享内存的全局访问模式。
 */

/**
 * @brief 将一个共享向量的数据复制到另一个共享向量，必要时进行类型转换
 * 
 * 此函数从源共享向量 `src` 复制数据到目标共享向量 `dst`。
 * 如果 `src` 和 `dst` 的数据类型相同，则执行直接内存复制。
 * 否则，使用适当的类型转换器将每个元素从源数据类型转换为目标数据类型后再复制。
 * 
 * @tparam SV1 目标共享向量类型，必须满足 ducks::sv::all 概念
 * @tparam SV2 源共享向量类型，必须满足 ducks::sv::all 概念
 * @param[out] dst 目标共享向量，接收复制数据
 * @param[in]  src 源共享向量，提供数据
 * 
 * @note `src` 和 `dst` 的长度必须相等，这在编译时强制执行
 * @note 使用协作线程组（thread group）模式，每个线程处理向量的一部分元素
 *       以提高内存访问效率并避免bank冲突
 */
template<ducks::sv::all SV1, ducks::sv::all SV2>
__device__ static inline void copy(SV1 &dst, const SV2 &src) {
    // 编译时检查：确保源和目标向量长度相同
    static_assert(SV1::length == SV2::length, "源和目标向量必须具有相同长度");
    
    // 获取当前线程在协作线程组内的ID（通常0-31）
    int thread_id = laneid();
    
    /**
     * @brief 循环展开优化：每个线程处理向量的一部分元素
     * 
     * 采用跨步循环（stride loop）模式：
     * - 起始索引：每个线程的lane ID
     * - 步长：GROUP_THREADS（协作线程组的大小，通常为32）
     * - 结束条件：索引小于向量长度
     * 
     * 这种访问模式：
     * 1. 确保连续的内存地址由连续的线程访问，实现合并内存访问
     * 2. 减少共享内存的bank冲突
     * 3. 提高内存带宽利用率
     * 
     * #pragma unroll 提示编译器展开循环以减少循环开销
     */
    #pragma unroll
    for(int i = thread_id; i < dst.length; i += GROUP_THREADS) {
        /**
         * @brief 执行类型安全的元素转换和复制
         * 
         * 使用 base_types::convertor 模板进行类型转换：
         * - 如果 SV1::dtype 和 SV2::dtype 相同：执行简单复制（编译器优化）
         * - 如果类型不同：执行适当的数据类型转换（如half到float，float到half等）
         * 
         * 这种方式确保：
         * 1. 类型安全：避免隐式转换的精度损失
         * 2. 性能优化：使用专用硬件转换指令（如__half2float, __float2half）
         * 3. 可扩展性：支持自定义数据类型的转换
         */
        dst[i] = base_types::convertor<typename SV1::dtype, typename SV2::dtype>::convert(src[i]);
    }
    
    /**
     * @note 函数结束后不需要显式同步
     * 
     * 调用者有责任在需要时插入适当的同步原语：
     * - __syncthreads()：确保所有线程完成对共享内存的写入
     * - 这允许调用者根据具体使用场景优化同步次数
     */
}