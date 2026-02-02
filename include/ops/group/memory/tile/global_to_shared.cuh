/**
 * @file
 * @brief 工作组（协作warp）操作 - 在全局内存和共享内存之间加载/存储共享tile
 */

/**
 * @brief 从全局内存加载数据到共享内存tile（可配置轴和对齐）
 *
 * @tparam axis 加载操作的轴方向（控制数据访问模式）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标共享内存tile
 * @param[in] src 源全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 该函数将全局内存中的数据协作加载到共享内存tile中，支持轴方向配置和对齐优化。
 * 使用float4向量化加载以提高内存带宽利用率。
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    // 获取共享内存tile的数据类型
    using T = typename ST::dtype;
    // 获取源数组在指定轴方向的行步长（元素数）
    const int row_stride = src.template stride<axis>();
    // 每次内存拷贝可以处理的元素数量（float4 = 16字节）
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    // 每行需要的内存拷贝次数
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    // 总的内存拷贝次数（向上取整）
    constexpr int total_calls = (dst.rows*dst.cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up
    // 获取tile在源数组中的起始坐标
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // 计算源数据指针
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    // 将共享内存地址转换为通用地址（用于内联PTX指令）
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 获取线程在组内的ID
    int laneid = threadIdx.x % GROUP_THREADS;
    
    // 主循环：每个线程处理一部分数据
    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        // 计算当前线程处理的全局索引
        int load_idx = i * GROUP_THREADS + laneid;
        // 计算在tile中的行和列位置        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;
        // 根据对齐假设选择不同的加载策略
        if constexpr (assume_aligned) {
            // 假设内存对齐：使用更高效的加载方式
            float4 tmp;  // 临时存储从全局内存加载的数据
            // 从全局内存加载float4数据（向量化加载）
            move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
            // 将数据存储到共享内存
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else {
            // 不假设内存对齐：需要检查边界条件
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                // 在边界内：正常加载
                float4 tmp;
                move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
            }
            else {
                // 超出边界：填充零值
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
}

/**
 * @brief 从全局内存加载数据到共享内存tile（默认设置版本）
 *
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标共享内存tile
 * @param[in] src 源全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是load函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 从共享内存tile存储数据到全局内存（可配置轴和对齐）
 *
 * @tparam axis 存储操作的轴方向（控制数据访问模式）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标全局内存数组
 * @param[in] src 源共享内存tile
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 该函数将共享内存tile中的数据协作存储到全局内存中，支持轴方向配置和对齐优化。
 * 使用float4向量化存储以提高内存带宽利用率。
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    // 获取共享内存tile的数据类型
    using T = typename ST::dtype;
    // 获取目标数组在指定轴方向的行步长（元素数）
    const int row_stride = dst.template stride<axis>();
    // 每次内存拷贝可以处理的元素数量（float4 = 16字节）
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    // 每行需要的内存拷贝次数
    constexpr int memcpy_per_row = src.cols / elem_per_memcpy;
    // 总的内存拷贝次数（向上取整）
    constexpr int total_calls = (src.rows*src.cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up
    // 获取tile在目标数组中的起始坐标
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // 计算目标数据指针
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];
    // 将共享内存地址转换为通用地址
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    // 获取线程在组内的ID
    int laneid = threadIdx.x % GROUP_THREADS;
    // 主循环：每个线程处理一部分数据
    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        // 计算当前线程处理的全局索引
        int load_idx = i * GROUP_THREADS + laneid;
        // 计算在tile中的行和列位置
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;
        // 根据对齐假设选择不同的存储策略
        if constexpr (assume_aligned) {
            // 假设内存对齐：使用更高效的存储方式
            float4 tmp;  // 临时存储从共享内存加载的数据
            // 从共享内存加载float4数据
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            // 将数据存储到全局内存
            move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
        }
        else {
            // 不假设内存对齐：需要检查边界条件
            if (row + unit_coord.template dim<axis>() < dst.template shape<axis>()) {
                // 在边界内：正常存储
                float4 tmp;
                move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
                move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
            }
            // 超出边界：不执行存储操作
        }
    }
}

/**
 * @brief 从共享内存tile存储数据到全局内存（默认设置版本）
 *
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标全局内存数组
 * @param[in] src 源共享内存tile
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是store函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief 异步从全局内存加载数据到共享内存tile（可配置轴和对齐）
 *
 * @tparam axis 加载操作的轴方向（控制数据访问模式）
 * @tparam assume_aligned 是否假设内存对齐（优化标志）
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标共享内存tile
 * @param[in] src 源全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 *
 * @note 此函数期望16字节对齐，否则行为未定义。
 * 
 * 该函数使用异步拷贝指令(cp.async)从全局内存加载数据到共享内存，
 * 可以在数据加载的同时执行其他计算，隐藏内存访问延迟。
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    // 获取共享内存tile的数据类型
    using T = typename ST::dtype;
    // 获取源数组在指定轴方向的行步长（元素数）
    const int row_stride = src.template stride<axis>();
    // 每次异步拷贝可以处理的元素数量（float4 = 16字节）
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    // 每行需要的异步拷贝次数
    constexpr int memcpy_per_row = dst.cols / elem_per_memcpy;
    // 总的异步拷贝次数（向上取整）
    constexpr int total_calls = (dst.rows*dst.cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy); // round up
    // 获取tile在源数组中的起始坐标
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // 计算源数据指针
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    // 将共享内存地址转换为通用地址
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 获取线程在组内的ID
    int laneid = threadIdx.x % GROUP_THREADS;
    
    // 主循环：每个线程处理一部分异步拷贝
    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        // 计算当前线程处理的全局索引
        int load_idx = i * GROUP_THREADS + laneid;
        // 计算在tile中的行和列位置        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;
        // 根据对齐假设选择不同的异步加载策略
        if constexpr (assume_aligned) {
            // 假设内存对齐：使用异步拷贝指令
            // cp.async.cg.shared.global.L2::128B: 从全局内存异步拷贝到共享内存，使用L2缓存，128字节粒度
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
        else {
            // 不假设内存对齐：需要检查边界条件
            if (row + unit_coord.template dim<axis>() < src.template shape<axis>()) {
                // 在边界内：正常异步拷贝
                asm volatile(
                    "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                    :: "r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row*row_stride + col])
                    : "memory"
                );
            }
            else {
                // 超出边界：填充零值（使用同步方式）
                // printf("thread %d skipping async load on row %d, col %d\n", threadIdx.x, row + unit_coord.template dim<axis>(), col);
                float4 zeros = {0.f,0.f,0.f,0.f};
                move<float4>::sts(dst.idx(dst_ptr, {row, col}), zeros); // use the default value
            }
        }
    }
    // 提交异步拷贝组，确保所有异步拷贝操作都已发起
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/**
 * @brief 异步从全局内存加载数据到共享内存tile（默认设置版本）
 *
 * @tparam ST 共享内存tile类型（必须满足ducks::st::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<ST>
 * @param[out] dst 目标共享内存tile
 * @param[in] src 源全局内存数组
 * @param[in] idx 在全局内存数组中的tile坐标
 *
 * @note 此函数期望16字节对齐，否则行为未定义。
 * 
 * 这是load_async函数的简化版本，使用默认设置：
 * - axis = 2（默认平面方向）
 * - assume_aligned = false（不假设内存对齐）
 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx) {
    load_async<2, false, ST, GL, COORD>(dst, src, idx);
}