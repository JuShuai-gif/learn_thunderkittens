/**
 * @file
 * @brief 用于组（协作warp）操作的函数，用于从全局内存加载数据到共享内存向量和从共享内存向量存储到全局内存。
 */

/**
 * @brief 从全局内存加载数据到共享内存向量。
 * 
 * 此函数从全局内存位置（由`src`指向）加载数据到共享内存向量`dst`。
 * 基于`float4`与`SV`数据类型的大小比率，计算一次操作可以传输的元素数量。
 * 通过将工作分配给warp中的线程，确保合并内存访问和高效使用带宽。
 * 
 * @tparam SV 共享向量类型，必须满足ducks::sv::all概念。
 * @param dst 引用到共享向量，数据将加载到此处。
 * @param src 指向全局内存位置的指针，数据将从该处加载。
 * @param idx 坐标索引，指定在全局内存中的位置。
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load(SV &dst, const GL &src, const COORD &idx) {
    // 计算每次传输可以处理的元素数量（基于float4大小）
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    // 计算总共需要的传输次数（保证整除）
    constexpr uint32_t total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    // 获取全局内存源指针
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    // 获取共享内存目标指针的32位表示
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 循环展开：每个线程处理部分数据
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        // 边界检查
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            // 从全局内存加载float4数据
            move<float4>::ldg(tmp, (float4*)&src_ptr[i*elem_per_transfer]);
            // 将数据存储到共享内存
            move<float4>::sts(dst_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer, tmp);
        }
    }
}

/**
 * @brief 从共享内存向量存储数据到全局内存。
 * 
 * 此函数从共享内存向量`src`存储数据到全局内存位置（由`dst`指向）。
 * 与加载函数类似，基于`float4`与`SV`数据类型的大小比率计算一次操作可以传输的元素数量。
 * 通过将工作分配给warp中的线程，确保合并内存访问和高效使用带宽。
 * 
 * @tparam SV 共享向量类型，必须满足ducks::sv::all概念。
 * @param dst 指向全局内存位置的指针，数据将存储到该处。
 * @param src 引用到共享向量，数据将从该处存储。
 * @param idx 坐标索引，指定在全局内存中的位置。
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store(GL &dst, const SV &src, const COORD &idx) {
    // 计算每次传输可以处理的元素数量（基于float4大小）
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    // 计算总共需要的传输次数（保证整除）
    constexpr uint32_t total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    // 获取全局内存目标指针
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
    // 获取共享内存源指针的32位表示
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    // 循环展开：每个线程处理部分数据
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        // 边界检查
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            // 从共享内存加载float4数据
            move<float4>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*i*elem_per_transfer);
            // 将数据存储到全局内存
            move<float4>::stg((float4*)&dst_ptr[i*elem_per_transfer], tmp);
        }
    }
}

/**
 * @brief 异步从全局内存加载数据到共享内存向量。
 *
 * 使用异步复制指令从全局内存加载数据到共享内存，适用于CUDA的异步内存复制功能。
 * 
 * @tparam SV 共享内存向量类型。
 * @param[out] dst 目标共享内存向量。
 * @param[in] src 源全局内存数组。
 * @param[in] idx 全局内存数组的坐标。
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx) {
    // 计算每次传输可以处理的元素数量（基于float4大小）
    constexpr uint32_t elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    // 计算总共需要的传输次数（保证整除）
    constexpr uint32_t total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    // 获取全局内存源指针
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    // 获取共享内存目标指针的32位表示
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 循环展开：每个线程处理部分数据
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        // 边界检查
        if(i * elem_per_transfer < dst.length) {
            // 使用异步复制指令：从全局内存复制16字节到共享内存，使用L2缓存128B缓存行
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst_ptr + (uint32_t)sizeof(typename SV::dtype)*i*elem_per_transfer), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    // 提交异步复制组，确保所有异步操作完成
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}


/**
 * @brief 使用较小传输大小从全局内存异步加载数据到共享内存向量。
 *
 * 与load_async类似，但使用较小的传输大小（4字节而非16字节），适用于小数据类型。
 * 
 * @tparam SV 共享内存向量类型。
 * @param[out] dst 目标共享内存向量。
 * @param[in] src 源全局内存数组。
 * @param[in] idx 全局内存数组的坐标。
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async_small(SV &dst, const GL &src, const COORD &idx) {
    // 计算每次传输可以处理的元素数量（基于float大小）
    constexpr uint32_t elem_per_transfer = sizeof(float) / sizeof(typename SV::dtype);
    // 计算总共需要的传输次数（保证整除）
    constexpr uint32_t total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    // 获取全局内存源指针
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    // 获取共享内存目标指针的32位表示
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 循环展开：每个线程处理部分数据
    #pragma unroll
    for(uint32_t i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        // 边界检查
        if(i * elem_per_transfer < SV::length) {
            // 使用异步复制指令：从全局内存复制4字节到共享内存，使用L2缓存128B缓存行            
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
                :: "r"(dst_ptr + (uint32_t)sizeof(typename SV::dtype)*i*elem_per_transfer), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    // 提交异步复制组，确保所有异步操作完成
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
