/**
 * @file
 * @brief 工作组（协作warp）操作 - 用于处理并行全局内存中的tile
 *
 * 警告：此API处于实验阶段。
 */



/*
关键概念说明：

并行全局内存(Parallel Global Memory, PGL):

一种特殊的内存类型，支持并行访问和归约操作

相比普通全局内存，PGL可以直接执行原子操作和归约

归约操作类型:

kittens::reduce_op 定义了可能的归约操作类型

可能包括加法、最大值、最小值、平均值等

bf16_2数据类型:

表示两个bf16数打包在一起的数据类型

提高内存访问效率，支持向量化操作

多线程归约模式:

每个warp处理64个bf16元素（32个线程 * 2个bf16）

warp内的线程协作执行归约操作

使用multimem<bf16_2>::ld_reduce<OP>执行加载+归约的复合操作

地址计算:

pgl.mc_ptr_at({b, d, row, col}) 计算并行全局内存中的地址

dst[{b, d, row, col}] 计算普通全局内存中的地址

坐标包含批次(b)、深度(d)、行(r)和列(c)维度

性能优化:

tile列数必须是64的倍数，确保每个warp处理完整的数据块

循环使用GROUP_WARPS步长，实现warp间的负载均衡

使用#pragma unroll可能的手动循环展开（虽然代码中未显示）

实验性API警告:

代码明确标注为实验阶段，接口可能发生变化

目前仅支持bf16数据类型，未来可能支持更多数据类型

使用场景:

分布式训练中的梯度归约

多GPU间的数据同步

大规模并行计算中的结果聚合

*/








/**
 * @brief 在并行全局内存(PGL)中对tile执行全归约操作
 *
 * @tparam TILE_ROWS tile的行数
 * @tparam TILE_COLS tile的列数
 * @tparam OP 归约操作类型（加法、最大值、最小值等）
 * @tparam PGL 并行全局内存类型（必须满足ducks::pgl::all概念）
 * @param[inout] pgl 并行全局内存对象，将对其进行原地归约
 * @param[in] idx tile在内存中的坐标
 * 
 * 此函数在指定的tile内执行全归约操作，所有数据归约到同一位置。
 * 目前仅支持bf16数据类型，且tile列数必须是64的倍数以获得最佳性能。
 */
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP, kittens::ducks::pgl::all PGL>
__device__ inline static void all_reduce(PGL &pgl, const coord<ducks::default_type> &idx) {
    // 静态断言：目前仅支持bf16数据类型
    static_assert(std::is_same_v<typename PGL::dtype, bf16>, "Currently only bf16 is supported.");
    // 静态断言：tile列数必须是64的倍数（WARP_THREADS*2 = 32*2 = 64）
    static_assert(TILE_COLS > 0 && TILE_COLS % (WARP_THREADS * 2) == 0, "TILE_COLS must be a multiple of 64 for best performance.");
    // 计算每行需要的warp数量（每个warp处理64个元素，bf16x2）
    const int warps_per_row = TILE_COLS / (WARP_THREADS * 2);
    // 计算tile的起始行和列
    const int row_base = idx.r * TILE_ROWS;

    const int col_base = idx.c * TILE_COLS;
    // 获取warp内的线程ID
    const int warp_laneid = threadIdx.x % WARP_THREADS;
    // 主循环：每个warp处理一部分数据
    // i从当前warp ID开始，步长为GROUP_WARPS，实现warp间的负载均衡
    for (int i = warpid(); i < TILE_ROWS * warps_per_row; i += GROUP_WARPS) {
        // 计算当前处理的全局行索引和列索引        
        int row_idx = row_base + i / warps_per_row;
        int col_idx = col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        // 从并行全局内存加载并执行归约操作
        bf16_2 tmp;
        bf16_2 *ptr = reinterpret_cast<bf16_2 *>(pgl.mc_ptr_at({idx.b, idx.d, row_idx, col_idx}));
        multimem<bf16_2>::ld_reduce<OP>(tmp, ptr);
        
        // 将归约结果存回原位置
        multimem<bf16_2>::st(ptr, tmp);
    }
}

/**
 * @brief 从并行全局内存(PGL)归约到普通全局内存(GL)
 *
 * @tparam TILE_ROWS tile的行数
 * @tparam TILE_COLS tile的列数
 * @tparam OP 归约操作类型（加法、最大值、最小值等）
 * @tparam PGL 并行全局内存类型（必须满足ducks::pgl::all概念）
 * @tparam GL 普通全局内存类型（必须满足ducks::gl::all概念）
 * @param[out] dst 目标普通全局内存数组
 * @param[in] dst_idx 目标tile在dst中的坐标
 * @param[in] src 源并行全局内存对象
 * @param[in] src_idx 源tile在src中的坐标
 * 
 * 此函数将并行全局内存中的tile归约到普通全局内存中。
 * 与all_reduce不同，此函数将结果写入不同的内存位置。
 * 目前仅支持bf16数据类型，且tile列数必须是64的倍数以获得最佳性能。
 */
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP, kittens::ducks::pgl::all PGL, kittens::ducks::gl::all GL>
__device__ inline static void reduce(GL &dst, const coord<ducks::default_type> &dst_idx, PGL &src, const coord<ducks::default_type> &src_idx) {
    // 静态断言：目前仅支持bf16数据类型
    static_assert(std::is_same_v<typename PGL::dtype, bf16>, "Currently only bf16 is supported.");
    // 静态断言：tile列数必须是64的倍数（WARP_THREADS*2 = 32*2 = 64）
    static_assert(TILE_COLS > 0 && TILE_COLS % (WARP_THREADS * 2) == 0, "TILE_COLS must be a multiple of 64 for best performance.");
    // 计算每行需要的warp数量（每个warp处理64个元素，bf16x2）
    const int warps_per_row = TILE_COLS / (WARP_THREADS * 2);
    // 计算源和目标tile的起始行和列    
    const int src_row_base = src_idx.r * TILE_ROWS;
    const int src_col_base = src_idx.c * TILE_COLS;
    const int dst_row_base = dst_idx.r * TILE_ROWS;
    const int dst_col_base = dst_idx.c * TILE_COLS;
    // 获取warp内的线程ID
    const int warp_laneid = threadIdx.x % WARP_THREADS;
    // 主循环：每个warp处理一部分数据
    // i从当前warp ID开始，步长为GROUP_WARPS，实现warp间的负载均衡
    for (int i = warpid(); i < TILE_ROWS * warps_per_row; i += GROUP_WARPS) {
        // 计算源和目标的全局行索引和列索引
        int src_row_idx = src_row_base + i / warps_per_row;
        int src_col_idx = src_col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        int dst_row_idx = dst_row_base + i / warps_per_row;
        int dst_col_idx = dst_col_base + (i % warps_per_row) * WARP_THREADS * 2 + warp_laneid * 2; // bf16x2
        // 抑制未使用变量的警告（在某些编译配置下可能会发出警告）
        (void)dst_row_idx; // Suppress false positive unused warning
        (void)dst_col_idx; // Suppress false positive unused warning
        // 从并行全局内存加载并执行归约操作
        bf16_2 tmp;
        multimem<bf16_2>::ld_reduce<OP>(tmp, reinterpret_cast<bf16_2 *>(src.mc_ptr_at({src_idx.b, src_idx.d, src_row_idx, src_col_idx})));
        // 将归约结果存储到普通全局内存
        move<bf16_2>::stg(reinterpret_cast<bf16_2 *>(&dst[{dst_idx.b, dst_idx.d, dst_row_idx, dst_col_idx}]), tmp);
    }
}
