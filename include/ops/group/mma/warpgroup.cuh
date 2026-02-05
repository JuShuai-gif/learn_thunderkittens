/**
 * @file
 * @brief Warpgroup矩阵乘加操作。这些操作对于在H100 GPU上实现完全利用是必要的。
 */

//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  ------------------------------------------------------ 内存栅栏 ------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------

/**
 * @brief 同步warp group并确保对共享内存的所有写入对warp group中的所有线程都可见。
 *
 * 此函数作为共享内存操作的栅栏，确保在进行之前所有先前的写入都可见。
 * 在运行wgmma::mma或wgmma::dot指令之前应调用此函数。
 *
 * @tparam height 矩阵`dst`的高度。
 * @tparam width 矩阵`dst`的宽度。
 * @param dst[in,out] 要同步的目标寄存器-瓦片矩阵。
 */
template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP // 检查是否在正确的warp group中执行
    
    // 使用内联汇编强制编译器将所有寄存器值刷新到内存/寄存器中
    // 这是一个编译器屏障，确保所有对dst的修改在此点之前完成
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                // 根据数据类型选择不同的处理方式
                if constexpr(std::is_same_v<typename D::T, float>) {
                    // 对于float类型，使用"+f"约束确保浮点寄存器被正确同步
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].y) :: "memory");
                } else {
                    // 对于其他类型（如half, bf16），使用通用寄存器
                    asm volatile("" : "+r"(*(uint32_t*)&dst.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    // 执行WGMMA（Warp Group Matrix Multiply Accumulate）栅栏操作
    // 确保所有先前的WGMMA操作在继续之前完成
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");

    // 执行共享内存代理操作的栅栏
    // 确保所有异步的共享内存操作对同一CTA（线程块）中的所有线程都可见
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/**
 * @brief 复数矩阵版本的warp group内存栅栏
 *
 * 此函数与实数版本类似，但处理复数矩阵（包含实部和虚部）
 */
template<ducks::crt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP // 检查是否在正确的warp group中执行
    
    // 遍历复数矩阵的实部和虚部，确保所有数据被同步
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.real.packed_per_tile; k++) {
                // 根据数据类型选择不同的处理方式
                if constexpr(std::is_same_v<typename D::T, float>) {
                    // 同步实部数据的两个部分（假设复数被拆分为两个浮点数）
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].y) :: "memory");
                    // 同步虚部数据的两个部分
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].y) :: "memory");
                } else {
                    // 对于非float类型，使用通用寄存器同步
                    asm volatile("" : "+r"(*(uint32_t*)&dst.real.tiles[i][j].data[k]) :: "memory");
                    asm volatile("" : "+r"(*(uint32_t*)&dst.imag.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    // 执行WGMMA栅栏
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    // 执行共享内存代理操作的栅栏
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/**
 * @brief 无数据依赖的warp group内存栅栏
 *
 * 此函数仅执行栅栏操作，不操作任何具体数据。
 * 用于只需要同步而不需要数据刷新的场景。
 *
 * @tparam T 模板参数，仅用于防止静态断言被实例化，除非被调用
 */
template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP // 检查是否在正确的warp group中执行
    // 执行WGMMA栅栏
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    // 执行共享内存代理操作的栅栏
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/**
 * @brief 提交当前的一组warp group矩阵乘加调用
 *
 * 此函数将当前累积的WGMMA操作提交给硬件执行。
 * 通常在配置好所有WGMMA操作后调用，以启动计算。
 *
 * @tparam T 模板参数，仅用于防止静态断言被实例化，除非被调用
 */
template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP // 检查是否在正确的warp group中执行
    
    // 提交当前组的所有WGMMA操作，同步并确保对齐
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief 等待warp group到达同步点
 *
 * 此函数使当前warp group等待，直到足够多的已提交WGMMA组完成。
 * 用于控制并发WGMMA操作的数量，防止资源竞争。
 *
 * @tparam N 允许的剩余活动WGMMA提交组的数量。此函数将等待，直到活动组的数量小于或等于N。默认为0。
 */
template<int N=0>
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP // 检查是否在正确的warp group中执行
    
    // 等待WGMMA操作完成，直到活动组的数量不超过N
    // %0是内联汇编中的占位符，将被N的值替换
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}

//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  ------------------------------------------------------ NORMAL ------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------

/*
 ### 可选的操作（OPTIONS）:

 寄存器+共享内存 -> 寄存器（REG+SMEM -> REG）
 - mma_AB   (accum) [DONE]  // 矩阵乘累加：A × B
 - mm_AB    (reset) [DONE]  // 矩阵乘重置：A × B（不累加）
 - mma_ABt  (accum) [DONE]  // 矩阵乘累加：A × Bᵀ
 - mm_ABt   (reset) [DONE]  // 矩阵乘重置：A × Bᵀ（不累加）
 
 共享内存+共享内存 -> 寄存器（SMEM+SMEM -> REG）
 - mma_AB   (accum) [DONE]  // 矩阵乘累加：A × B
 - mm_AB    (reset) [DONE]  // 矩阵乘重置：A × B（不累加）
 - mma_ABt  (accum) [DONE]  // 矩阵乘累加：A × Bᵀ
 - mm_ABt   (reset) [DONE]  // 矩阵乘重置：A × Bᵀ（不累加）
 - mma_AtB  (accum) [DONE]  // 矩阵乘累加：Aᵀ × B
 - mm_AtB   (reset) [DONE]  // 矩阵乘重置：Aᵀ × B（不累加）
 - mma_AtBt (accum) [DONE]  // 矩阵乘累加：Aᵀ × Bᵀ
 - mm_AtBt  (reset) [DONE]  // 矩阵乘重置：Aᵀ × Bᵀ（不累加）
 
注意：mma 是 mma_AB 的别名，dot 是 mma_ABt 的别名
*/

// ================================================================
// 矩阵乘操作：寄存器A × 共享内存B -> 寄存器D
// ================================================================

/**
 * @brief 使用Warp Group矩阵乘累加（WGMMA）原语执行矩阵乘累加操作。
 *        计算 A × B 并累加到 D 中。
 *
 * @tparam accumulate 是否将结果累加到D中（1为累加，0为覆盖）
 * @tparam fence 是否在执行前插入内存栅栏（1为插入，0为不插入）
 * @param d[out] 目标寄存器瓦片，结果将累加或写入到此
 * @param a[in] 源寄存器瓦片，参与乘法
 * @param b[in] 源共享内存瓦片，参与乘法
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // 检查：确保在Warp Group中执行
    KITTENS_CHECK_WARPGROUP

    // 静态断言：检查维度匹配    
    constexpr int M_DIV_4 = A::height;      // A的高度（以瓦片计），对应输出D的行数
    static_assert(D::height == M_DIV_4);    // 输出寄存器D的高度必须与A的高度匹配
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;// 从B的列数计算N维度
    static_assert(D::width == N);           // D的宽度必须等于N
    constexpr int K = A::width;             // A的宽度，约减维度
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K);   // B的行数必须等于K（约减维度匹配）
    static_assert(std::is_same_v<typename A::T, typename B::T>);        // A和B的数据类型必须相同

    // 类型别名
    using T_AB = A::T;  // A和B的数据类型（如bf16、half等）
    using T_D  = D::T;  // D的数据类型（如float、half等）

    // 检查不支持的数据类型（如fp8）    
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");

    // 使用WGMMA基类模板，参数：输出类型，输入类型，跨步，是否转置A，是否转置B    
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>;
    // 创建共享内存描述符，用于访问共享内存中的B矩阵
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b); // 参数1表示B矩阵的布局
    // 可选：插入内存栅栏，确保之前的存储操作完成
    if constexpr (fence) { mma_fence(d); }

    // 执行矩阵乘法
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        // 获取D中对应行的子瓦片引用
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);
        // 第一步：使用k=0的瓦片初始化累加或覆盖
        base::rt_st(
            d_ref,                          // 目标寄存器
            a.tiles[m][0],                  // A的第m行第0列瓦片
            b_desc.chunk_descriptor(0),     // B的第0个瓦片描述符
            accumulate                      // 是否累加
        );
        // 后续步骤：K维度累加，从k=1到K-1
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,                      // 目标寄存器（作为累加器）
                a.tiles[m][k],              // A的第m行第k列瓦片
                b_desc.chunk_descriptor(k), // B的第k个瓦片描述符
                1                           // 总是累加模式
            );
        }
    }
    mma_commit_group(); // 提交WGMMA调用组，确保所有操作完成
}

/**
 * @brief 矩阵乘重置操作：寄存器A × 共享内存B -> 寄存器D（覆盖模式）
 *        计算 A × B 并覆盖写入 D（不累加）。
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    // 调用累加版本的mma_AB，但设置accumulate=0，fence=1
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

// ================================================================
// 矩阵乘操作：共享内存A × 共享内存B -> 寄存器D
// ================================================================

/**
 * @brief 使用Warp Group矩阵乘累加（WGMMA）原语执行矩阵乘累加操作。
 *        计算共享内存A × 共享内存B 并累加到寄存器D中。
 *
 * @tparam accumulate 是否将结果累加到D中（1为累加，0为覆盖）
 * @tparam fence 是否在执行前插入内存栅栏（1为插入，0为不插入）
 * @param d[out] 目标寄存器瓦片
 * @param a[in] 源共享内存瓦片A
 * @param b[in] 源共享内存瓦片B
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // 检查：确保在Warp Group中执行
    KITTENS_CHECK_WARPGROUP


    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;   // 从A的行数计算M维度
    static_assert(M == 4);// 共享内存版本要求M=4（这是WGMMA指令的限制）
    static_assert(D::height == 1); // 输出寄存器D的高度必须为1（因为M=4）
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;// 从B的列数计算N维度
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;// 从A的列数计算K维度（约减维度）
    static_assert(B::rows / kittens::TILE_ROW_DIM<typename B::T> == K); // B的行数必须等于K（约减维度匹配）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A和B的数据类型必须相同

    // 类型别名
    using T_AB = A::T;  // A和B的数据类型
    using T_D  = D::T;  // D的数据类型

    // 检查不支持的数据类型（如fp8）    
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");

    // 使用WGMMA基类模板，参数：输出类型，输入类型，跨步，是否转置A，是否转置B
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>;

    // 创建共享内存描述符，用于访问共享内存中的A和B矩阵
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);// 参数0表示A矩阵的布局
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);// 参数1表示B矩阵的布局
    
    // 可选：插入内存栅栏
    if constexpr (fence) { mma_fence(d); }

    // 执行矩阵乘法
    base::st_st(
        d,// 目标寄存器
        a_desc.chunk_descriptor(0),// A的第0个瓦片描述符
        b_desc.chunk_descriptor(0),// B的第0个瓦片描述符
        accumulate// 是否累加
    );

    // 后续步骤：K维度累加，从k=1到K-1
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,                              // 目标寄存器（作为累加器）
            a_desc.chunk_descriptor(k),     // A的第k个瓦片描述符
            b_desc.chunk_descriptor(k),     // B的第k个瓦片描述符
            1                               // 总是累加模式
        );
    }
    mma_commit_group(); // 提交WGMMA调用组
}

/**
 * @brief 矩阵乘重置操作：共享内存A × 共享内存B -> 寄存器D（覆盖模式）
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    // 调用累加版本的mma_AB，但设置accumulate=0，fence=1
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

// ================================================================
// 点积操作：寄存器A × 共享内存Bᵀ -> 寄存器D
// ================================================================

/**
 * @brief 使用Warp Group矩阵乘累加（WGMMA）原语执行点积累加操作。
 *        计算 A × Bᵀ 并累加到 D 中。
 *
 * @tparam accumulate 是否将结果累加到D中（1为累加，0为覆盖）
 * @tparam fence 是否在执行前插入内存栅栏（1为插入，0为不插入）
 * @param d[out] 目标寄存器瓦片
 * @param a[in] 源寄存器瓦片
 * @param b[in] 源共享内存瓦片（计算时转置）
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // 检查：确保在Warp Group中执行
    KITTENS_CHECK_WARPGROUP
    // 静态断言：检查维度匹配
    constexpr int M_DIV_4 = A::height;      // A的高度（以瓦片计）
    static_assert(D::height == M_DIV_4);    // 输出寄存器D的高度必须与A的高度匹配
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;   // 从B的行数计算N维度（因为B要转置）
    constexpr int K = A::width;             // A的宽度，约减维度
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);   // B的列数必须等于K（约减维度匹配）
    static_assert(std::is_same_v<typename A::T, typename B::T>);        // A和B的数据类型必须相同

    // 类型别名
    using T_AB = A::T;  // A和B的数据类型
    using T_D  = D::T;  // D的数据类型

    // 使用WGMMA基类模板，参数：输出类型，输入类型，跨步，是否转置A，是否转置B
    // 注意：这里转置B，所以最后一个参数是0
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;

    // 创建共享内存描述符，用于访问共享内存中的B矩阵
    // 参数0表示B矩阵的布局（因为B要转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);
    
    // 可选：插入内存栅栏
    if constexpr (fence) { mma_fence(d); }

    // 执行矩阵乘法
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        // 获取D中对应行的子瓦片引用
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);

        // 第一步：使用k=0的瓦片初始化累加或覆盖
        base::rt_st(
            d_ref,              // 目标寄存器
            a.tiles[m][0],      // A的第m行第0列瓦片
            b_desc.chunk_descriptor(0),     // B的第0个瓦片描述符
            accumulate          // 是否累加
        );

        // 后续步骤：K维度累加，从k=1到K-1
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,          // 目标寄存器（作为累加器）
                a.tiles[m][k],  // A的第m行第k列瓦片
                b_desc.chunk_descriptor(k), // B的第k个瓦片描述符
                1               // 总是累加模式
            );
        }
    }
    mma_commit_group(); // 提交WGMMA调用组
}

/**
 * @brief 点积重置操作：寄存器A × 共享内存Bᵀ -> 寄存器D（覆盖模式）
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    // 调用累加版本的mma_ABt，但设置accumulate=0，fence=1
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// ================================================================
// 点积操作：共享内存A × 共享内存Bᵀ -> 寄存器D
// ================================================================

/**
 * @brief 使用Warp Group矩阵乘累加（WGMMA）原语执行点积累加操作。
 *        计算共享内存A × 共享内存Bᵀ 并累加到寄存器D中。
 *
 * @tparam accumulate 是否将结果累加到D中（1为累加，0为覆盖）
 * @tparam fence 是否在执行前插入内存栅栏（1为插入，0为不插入）
 * @param d[out] 目标寄存器瓦片
 * @param a[in] 源共享内存瓦片A
 * @param b[in] 源共享内存瓦片B（计算时转置）
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // 检查：确保在Warp Group中执行
    KITTENS_CHECK_WARPGROUP
    // 静态断言：检查维度匹配
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;   // 从A的行数计算M维度
    static_assert(M == 4);  // 共享内存版本要求M=4
    static_assert(D::height == 1); // 输出寄存器D的高度必须为1
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;// 从B的行数计算N维度（因为B要转置）
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;// 从A的列数计算K维度（约减维度）
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K); // B的列数必须等于K（约减维度匹配）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A和B的数据类型必须相同

    // 类型别名
    using T_AB = A::T;// A和B的数据类型
    using T_D  = D::T;// D的数据类型

    // 使用WGMMA基类模板，参数：输出类型，输入类型，跨步，是否转置A，是否转置B
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;

    // 创建共享内存描述符，用于访问共享内存中的A和B矩阵
    // 参数0表示A矩阵的布局，参数0表示B矩阵的布局（因为B要转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);
    // 可选：插入内存栅栏
    if constexpr (fence) { mma_fence(d); }

    // 执行矩阵乘法
    base::st_st(
        d,// 目标寄存器
        a_desc.chunk_descriptor(0),// A的第0个瓦片描述符
        b_desc.chunk_descriptor(0),// B的第0个瓦片描述符
        accumulate// 是否累加
    );

    // 后续步骤：K维度累加，从k=1到K-1
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,                              // 目标寄存器（作为累加器）
            a_desc.chunk_descriptor(k),     // A的第k个瓦片描述符
            b_desc.chunk_descriptor(k),     // B的第k个瓦片描述符
            1                               // 总是累加模式
        );
    }
    mma_commit_group(); // 提交WGMMA调用组
}

/**
 * @brief 点积重置操作：共享内存A × 共享内存Bᵀ -> 寄存器D（覆盖模式）
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    // 调用累加版本的mma_ABt，但设置accumulate=0，fence=1
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// ================================================================
// 矩阵乘操作：共享内存Aᵀ × 共享内存B -> 寄存器D
// ================================================================

/**
 * @brief 使用Warp Group矩阵乘累加（WGMMA）原语执行矩阵乘累加操作。
 *        计算共享内存Aᵀ × 共享内存B 并累加到寄存器D中。
 *
 * @tparam accumulate 是否将结果累加到D中（1为累加，0为覆盖）
 * @tparam fence 是否在执行前插入内存栅栏（1为插入，0为不插入）
 * @param d[out] 目标寄存器瓦片
 * @param a[in] 源共享内存瓦片A（计算时转置）
 * @param b[in] 源共享内存瓦片B
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    // 检查：确保在Warp Group中执行
    KITTENS_CHECK_WARPGROUP

    // 静态断言：检查维度匹配
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;// 从A的列数计算M维度（因为A要转置）
    static_assert(M == 4);// 共享内存版本要求M=4
    static_assert(D::height == 1); // 输出寄存器D的高度必须为1
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;// 从B的列数计算N维度
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;// 从A的行数计算K维度（约减维度，因为A要转置）
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K); // B的行数必须等于K（约减维度匹配）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A和B的数据类型必须相同

    // 类型别名
    using T_AB = A::T;  // A和B的数据类型
    using T_D  = D::T;  // D的数据类型

    // 检查不支持的数据类型（如fp8）
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");

    // 使用WGMMA基类模板，参数：输出类型，输入类型，跨步，是否转置A，是否转置B
    // 注意：这里转置A，所以倒数第二个参数是1
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;

    // 创建共享内存描述符，用于访问共享内存中的A和B矩阵
    // 参数1表示A矩阵的布局（因为A要转置），参数1表示B矩阵的布局
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);
    // 可选：插入内存栅栏
    if constexpr (fence) { mma_fence(d); }

    // 执行矩阵乘法
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    // 后续步骤：K维度累加，从k=1到K-1
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}

/**
 * @brief 矩阵乘重置操作：共享内存Aᵀ × 共享内存B -> 寄存器D（覆盖模式）
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    // 调用累加版本的mma_AtB，但设置accumulate=0，fence=1
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] 版本
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行矩阵乘法，其中A和B已转置。
 *
 * 此函数计算共享图块`a`和共享图块`b`的外积，并将结果写入寄存器图块`d`。
 *
 * @tparam D 目标寄存器图块类型。
 * @tparam A 源共享图块类型。
 * @tparam B 源共享图块类型。
 * @tparam fence 是否在执行前添加memory fence屏障(默认1:是)。
 * @tparam accumulate 是否将结果累加到`d`中(默认1:累加)或覆盖`d`(0:覆盖)。
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {
    // 静态断言检查，确保当前执行环境支持warp group操作
    KITTENS_CHECK_WARPGROUP

    // 计算M维度：A的列数除以每个tile的列维度    
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);  // 强制要求M维度为4

    // 检查输出寄存器图块高度是否为1（正确的行布局）    
    static_assert(D::height == 1); // output register is correctly sized
    // 计算N维度：B的行数除以每个tile的行维度
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    // 计算K维度：A的行数除以每个tile的行维度
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    // 验证K维度匹配：B的列数对应的tile数必须等于K
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K); // K dimension must match
    // 验证A和B的数据类型必须相同
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // 类型别名定义
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出寄存器数据类型

    // 静态断言：目前不支持fp8e4m3和fp8e5m2数据类型
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    // 定义WGMMA操作的基础类型，配置为从共享内存到寄存器的操作
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    // 创建共享内存描述符，用于访问共享内存中的矩阵数据
    // 参数1表示列主序访问（假设A需要转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    // 参数0表示行主序访问（假设B需要转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);
    // 如果需要，在执行前添加memory fence，确保内存操作顺序
    if constexpr (fence) { mma_fence(d); }

    // 执行WGMMA操作：第一次迭代
    // 使用第0个数据块进行计算，根据accumulate参数决定是否累加
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );

    // 循环处理剩余的K-1个数据块
    #pragma unroll  // 编译器指令：完全展开循环以优化性能
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group();     // 提交这一组WGMMA操作，确保所有操作完成
}

/**
 * @brief 矩阵乘法函数（覆盖模式），A和B已转置
 *
 * 此函数是mma_AtBt的包装，专用于覆盖模式（不累加到现有结果）
 *
 * @tparam D 目标寄存器图块类型
 * @tparam A 源共享图块类型
 * @tparam B 源共享图块类型
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    // 调用基础函数，设置fence=1（启用屏障），accumulate=0（覆盖模式）
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}



//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  -------------------------------------------------- COMPLEX INPUTS --------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------

/*
 ### OPTIONS:

 REG+SMEM -> REG
 - mma_AB   (accum) [TODO]   // 寄存器+共享内存->寄存器，累加模式
 - mm_AB    (reset) [TODO]   // 寄存器+共享内存->寄存器，重置模式
 - mma_ABt  (accum) [TODO]   // 寄存器+共享内存转置->寄存器，累加模式
 - mm_ABt   (reset) [TODO]   // 寄存器+共享内存转置->寄存器，重置模式
 
 SMEM+SMEM -> REG
 - mma_AB   (accum) [TODO]   // 共享内存+共享内存->寄存器，累加模式
 - mm_AB    (reset) [TODO]   // 共享内存+共享内存->寄存器，重置模式
 - mma_AtB  (accum) [TODO]   // 共享内存转置+共享内存->寄存器，累加模式
 - mm_AtB   (reset) [TODO]   // 共享内存转置+共享内存->寄存器，重置模式
 - mma_AtBt (accum) [TODO]   // 共享内存转置+共享内存转置->寄存器，累加模式
 - mm_AtBt  (reset) [TODO]   // 共享内存转置+共享内存转置->寄存器，重置模式
 
Note: mma is an alias for mma_AB and dot is an alias for mma_ABt  // 注：mma是mma_AB的别名，dot是mma_ABt的别名
*/

// [(register, shared) -> register] edition  // 版本：[寄存器，共享内存] -> 寄存器
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵乘加操作。
 *        该函数将寄存器tile `a` 与共享tile `b` 相乘，并将结果写入寄存器tile `d`。
 *
 * @tparam accumulate 是否将结果累加到 `d` 中还是覆盖 `d`。
 * @tparam N_DIV_4 矩阵 `a` 的高度除以4。
 * @tparam K 矩阵 `a` 和 `b` 的公共维度。
 * @tparam M 矩阵 `b` 和 `d` 的宽度。
 * @tparam L_B 矩阵 `b` 的布局。
 * @param d[out] 目标寄存器tile，结果将累加或写入到此。
 * @param a[in] 源寄存器tile，用于乘法。
 * @param b[in] 源共享tile，用于乘法。
 */
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // 检查
    KITTENS_CHECK_WARPGROUP     // 检查warp group配置
    constexpr int M_DIV_4 = A::height;  // 获取A的高度（除以4后的值）
    static_assert(D::height == M_DIV_4); // 验证输出寄存器大小正确
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;// 计算N维度
    constexpr int K = A::width;// 获取K维度（A的宽度）
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K); // 验证K维度必须匹配
    static_assert(std::is_same_v<typename A::T, typename B::T>); // 验证A和B必须类型相同

    // 类型别名
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出矩阵数据类型


    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");// 检查不支持的类型
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");// 检查不支持的类型
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>; // 使用WGMMA基础类
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);// 创建b实部共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);// 创建b虚部共享内存描述符

    if constexpr (fence) { mma_fence(d); }// 如果需要，插入内存栅栏

    // 执行计算
    #pragma unroll // 实部计算循环展开
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m);// 获取d的实部子tile引用
        base::rt_st(// 执行寄存器到共享内存的乘加操作（实部*实部）
            d_ref,
            a.real.tiles[m][0],// a的实部
            b_desc_real.chunk_descriptor(0), // b的实部描述符
            accumulate// 是否累加
        );
        #pragma unroll      // K维度的累加循环展开
        for(int k = 1; k < K; k++) {
            base::rt_st(    // 继续累加实部*实部
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1           // 总是累加
            );
        }
        #pragma unroll      // 虚部*虚部循环展开（注意符号取反）
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>( // 虚部*虚部，符号取反（因为i² = -1）
                d_ref,
                a.imag.tiles[m][k], // a的虚部
                b_desc_imag.chunk_descriptor(k),    // b的虚部描述符
                1           // 总是累加
            );
        }
    }
    #pragma unroll // 虚部计算循环展开
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_COL_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);// 获取d的虚部子tile引用
        base::rt_st(// 执行实部*虚部
            d_ref,
            a.real.tiles[m][0],// a的实部
            b_desc_imag.chunk_descriptor(0),// b的虚部描述符
            accumulate// 是否累加
        );
        #pragma unroll// K维度的累加循环展开
        for(int k = 1; k < K; k++) {
            base::rt_st(// 继续累加实部*虚部
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1// 总是累加
            );
        }
        #pragma unroll// 虚部*实部循环展开
        for(int k = 0; k < K; k++) {
            base::rt_st(// 执行虚部*实部
                d_ref,
                a.imag.tiles[m][k],// a的虚部
                b_desc_real.chunk_descriptor(k),// b的实部描述符
                1// 总是累加
            );
        }
    }
    mma_commit_group(); // 提交这一组WGMMA调用
}

// 非累加版本的mm_AB（重置模式）
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);// 调用累加版本，但设置accumulate=0（重置模式）
}

// [(shared, shared) -> register] edition  // 版本：[共享内存，共享内存] -> 寄存器
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵乘加操作。
 *        该函数将共享tile `a` 与共享tile `b` 相乘，并将结果写入寄存器tile `d`。
 *
 * @tparam accumulate 是否将结果累加到 `d` 中还是覆盖 `d`。
 * @tparam fence 是否插入内存栅栏。
 * @param d[out] 目标寄存器tile，结果将累加或写入到此。
 * @param a[in] 源共享tile，用于乘法。
 * @param b[in] 源共享tile，用于乘法。
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // 检查
    KITTENS_CHECK_WARPGROUP  // 检查warp group配置
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;// 计算M维度
    static_assert(M == 4);// 验证M必须为4
    static_assert(D::height == 1); // 验证输出寄存器大小正确
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;// 计算N维度
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;// 计算K维度
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K); // 验证K维度必须匹配
    static_assert(std::is_same_v<typename A::T, typename B::T>); // 验证A和B必须类型相同

    // 类型别名
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出矩阵数据类型
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");// 检查不支持的类型
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");// 检查不支持的类型
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>; // 使用WGMMA基础类
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);// 创建a实部共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);// 创建a虚部共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);// 创建b实部共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);// 创建b虚部共享内存描述符

    if constexpr (fence) { mma_fence(d); }// 如果需要，插入内存栅栏

    // 执行计算
    base::st_st(  // 执行共享内存到共享内存的乘加操作（实部*实部）
        d.real,  // 结果实部
        a_desc_real.chunk_descriptor(0),  // a的实部描述符
        b_desc_real.chunk_descriptor(0),  // b的实部描述符
        accumulate  // 是否累加
    );
    #pragma unroll  // K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*实部
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll// 虚部*虚部循环展开（注意符号取反）
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // 虚部*虚部，符号取反（因为i² = -1）
            d.real,
            a_desc_imag.chunk_descriptor(k),// a的虚部描述符
            b_desc_imag.chunk_descriptor(k),// b的虚部描述符
            1 // 总是累加
        );
    }
    base::st_st(// 执行实部*虚部
        d.imag,// 结果虚部
        a_desc_real.chunk_descriptor(0),// a的实部描述符
        b_desc_imag.chunk_descriptor(0),// b的虚部描述符
        accumulate// 是否累加
    );
    #pragma unroll// K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*虚部
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll  // 虚部*实部循环展开
    for(int k = 0; k < K; k++) {
        base::st_st(  // 执行虚部*实部
            d.imag,
            a_desc_imag.chunk_descriptor(k),  // a的虚部描述符
            b_desc_real.chunk_descriptor(k),  // b的实部描述符
            1  // 总是累加
        );
    }
    mma_commit_group(); // 提交这一组WGMMA调用
}

// 非累加版本的mm_AB（重置模式）
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b); // 调用累加版本，但设置accumulate=0（重置模式）
}

// [(register, shared) -> register] edition  // 版本：[寄存器，共享内存] -> 寄存器
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵外积操作。
 *        该函数计算寄存器tile `a` 与共享tile `b` 的外积，并将结果写入寄存器tile `d`。
 *
 * @tparam accumulate 是否将结果累加到 `d` 中还是覆盖 `d`。
 * @tparam N_DIV_4 矩阵 `a` 的高度除以4。
 * @tparam K 矩阵 `a` 和 `b` 的公共维度。
 * @tparam M 矩阵 `b` 和 `d` 的高度。
 * @tparam L_B 矩阵 `b` 的布局。
 * @param d[out] 目标寄存器tile，结果将累加或写入到此。
 * @param a[in] 源寄存器tile，用于乘法。
 * @param b[in] 源共享tile，用于乘法。
 */
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // 检查
    KITTENS_CHECK_WARPGROUP  // 检查warp group配置
    constexpr int M_DIV_4 = A::height;  // 获取A的高度（除以4后的值）
    static_assert(D::height == M_DIV_4); // 验证输出寄存器大小正确
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;// 计算N维度（注意：这里使用的是行数，因为B被转置）
    constexpr int K = A::width;// 获取K维度（A的宽度）
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K); // 验证K维度必须匹配（B的列数）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // 验证A和B必须类型相同

    // 类型别名
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出矩阵数据类型
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;// 使用WGMMA基础类，注意最后参数为0表示B未转置
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);// 创建b实部共享内存描述符，布局为行优先
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);// 创建b虚部共享内存描述符，布局为行优先

    if constexpr (fence) { mma_fence(d); }// 如果需要，插入内存栅栏

    // 执行计算（与mma_AB类似，但使用了不同的base和描述符布局）
    #pragma unroll  // 实部计算循环展开
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_ROW_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m); // 获取d的实部子tile引用
        base::rt_st(// 执行寄存器到共享内存的乘加操作（实部*实部）
            d_ref,
            a.real.tiles[m][0],// a的实部
            b_desc_real.chunk_descriptor(0),// b的实部描述符
            accumulate// 是否累加
        );
        #pragma unroll  // K维度的累加循环展开
        for(int k = 1; k < K; k++) {
            base::rt_st(// 继续累加实部*实部
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1   // 总是累加
            );
        }
        #pragma unroll// 虚部*虚部循环展开（注意符号取反）
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>( // 虚部*虚部，符号取反（因为i² = -1）
                d_ref,
                a.imag.tiles[m][k],// a的虚部
                b_desc_imag.chunk_descriptor(k),// b的虚部描述符
                1// 总是累加
            );
        }
    }
    #pragma unroll// 虚部计算循环展开
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_ROW_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);// 获取d的虚部子tile引用
        base::rt_st(// 执行实部*虚部
            d_ref,
            a.real.tiles[m][0],// a的实部
            b_desc_imag.chunk_descriptor(0),// b的虚部描述符
            accumulate// 是否累加
        );
        #pragma unroll// K维度的累加循环展开
        for(int k = 1; k < K; k++) {
            base::rt_st(// 继续累加实部*虚部
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1// 总是累加
            );
        }
        #pragma unroll // 虚部*实部循环展开
        for(int k = 0; k < K; k++) {
            base::rt_st(// 执行虚部*实部
                d_ref,
                a.imag.tiles[m][k],// a的虚部
                b_desc_real.chunk_descriptor(k),// b的实部描述符
                1// 总是累加
            );
        }
    }
    mma_commit_group(); // 提交这一组WGMMA调用
}

// 非累加版本的mm_ABt（重置模式）
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);// 调用累加版本，但设置accumulate=0（重置模式）
}

// [(shared, shared) -> register] edition  // 版本：[共享内存，共享内存] -> 寄存器
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵外积操作。
 *        该函数计算共享tile `a` 与共享tile `b` 的外积，并将结果写入寄存器tile `d`。
 *
 * @tparam accumulate 是否将结果累加到 `d` 中还是覆盖 `d`。
 * @tparam K 矩阵 `a` 和 `b` 的公共维度。
 * @tparam M 矩阵 `b` 和 `d` 的高度。
 * @tparam L_A 矩阵 `a` 的布局。
 * @tparam L_B 矩阵 `b` 的布局。
 * @param d[out] 目标寄存器tile，结果将累加或写入到此。
 * @param a[in] 源共享tile，用于乘法。
 * @param b[in] 源共享tile，用于乘法。
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // 检查
    KITTENS_CHECK_WARPGROUP  // 检查warp group配置
    constexpr int M = A::rows / kittens::TILE_ROW_DIM<typename A::T>;  // 计算M维度
    static_assert(M == 4);  // 验证M必须为4
    static_assert(D::height == 1); // 验证输出寄存器大小正确
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;  // 计算N维度（注意：这里使用的是行数，因为B被转置）
    constexpr int K = A::cols / kittens::TILE_COL_DIM<typename A::T>;  // 计算K维度
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K); // 验证K维度必须匹配（B的列数）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // 验证A和B必须类型相同

    // 类型别名
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出矩阵数据类型
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 0>;  // 使用WGMMA基础类，注意最后参数为0表示B未转置
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);  // 创建a实部共享内存描述符，布局为行优先
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);  // 创建a虚部共享内存描述符，布局为行优先
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);  // 创建b实部共享内存描述符，布局为行优先
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);  // 创建b虚部共享内存描述符，布局为行优先

    if constexpr (fence) { mma_fence(d); }// 如果需要，插入内存栅栏

    // 执行计算（与共享内存版本的mma_AB类似，但使用了不同的base和描述符布局）
    base::st_st(  // 执行共享内存到共享内存的乘加操作（实部*实部）
        d.real,  // 结果实部
        a_desc_real.chunk_descriptor(0),  // a的实部描述符
        b_desc_real.chunk_descriptor(0),  // b的实部描述符
        accumulate  // 是否累加
    );
    #pragma unroll// K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*实部
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll// 虚部*虚部循环展开（注意符号取反）
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // 虚部*虚部，符号取反（因为i² = -1）
            d.real,
            a_desc_imag.chunk_descriptor(k),// a的虚部描述符
            b_desc_imag.chunk_descriptor(k),// b的虚部描述符
            1// 总是累加
        );
    }
    base::st_st( // 执行实部*虚部
        d.imag,// 结果虚部
        a_desc_real.chunk_descriptor(0),// a的实部描述符
        b_desc_imag.chunk_descriptor(0),// b的虚部描述符
        accumulate// 是否累加
    );
    #pragma unroll// K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*虚部
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll// 虚部*实部循环展开
    for(int k = 0; k < K; k++) {
        base::st_st(// 执行虚部*实部
            d.imag,
            a_desc_imag.chunk_descriptor(k),// a的虚部描述符
            b_desc_real.chunk_descriptor(k), // b的实部描述符
            1// 总是累加
        );
    }
    mma_commit_group(); // 提交这一组WGMMA调用
}

// 非累加版本的mm_ABt（重置模式）
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);// 调用累加版本，但设置accumulate=0（重置模式）
}

// [(shared, shared) -> register] edition  // 版本：[共享内存，共享内存] -> 寄存器
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵乘法，其中A被转置。
 *        该函数计算共享tile `a` 与共享tile `b` 的乘法（A转置），并将结果写入寄存器tile `d`。
 *
 * @tparam accumulate 是否将结果累加到 `d` 中还是覆盖 `d`。
 * @tparam K 矩阵 `a` 和 `b` 的公共维度。
 * @tparam M 矩阵 `b` 和 `d` 的高度。
 * @tparam L_A 矩阵 `a` 的布局。
 * @tparam L_B 矩阵 `b` 的布局。
 * @param d[out] 目标寄存器tile，结果将累加或写入到此。
 * @param a[in] 源共享tile，用于乘法（将被转置）。
 * @param b[in] 源共享tile，用于乘法。
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    // 检查
    KITTENS_CHECK_WARPGROUP  // 检查warp group配置
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;  // 计算M维度（A的列数，因为A被转置）
    static_assert(M == 4);  // 验证M必须为4
    static_assert(D::height == 1); // 验证输出寄存器大小正确
    constexpr int N = B::cols / kittens::TILE_COL_DIM<typename B::T>;  // 计算N维度
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;  // 计算K维度（A的行数）
    static_assert(B::rows/kittens::TILE_ROW_DIM<typename B::T> == K); // 验证K维度必须匹配（B的行数）
    static_assert(std::is_same_v<typename A::T, typename B::T>); // 验证A和B必须类型相同

    // 类型别名
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出矩阵数据类型
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");  // 检查不支持的类型
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");  // 检查不支持的类型
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;  // 使用WGMMA基础类，注意第4个参数为1表示A转置，第5个参数为1表示B未转置
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);  // 创建a实部共享内存描述符，布局为列优先（因为A转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);  // 创建a虚部共享内存描述符，布局为列优先（因为A转置）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);  // 创建b实部共享内存描述符，布局为列优先
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);  // 创建b虚部共享内存描述符，布局为列优先

    if constexpr (fence) { mma_fence(d); }// 如果需要，插入内存栅栏

    // 执行计算（与共享内存版本的mma_AB类似，但使用了不同的base和描述符布局）
    base::st_st(  // 执行共享内存到共享内存的乘加操作（实部*实部）
        d.real,  // 结果实部
        a_desc_real.chunk_descriptor(0),  // a的实部描述符
        b_desc_real.chunk_descriptor(0),  // b的实部描述符
        accumulate  // 是否累加
    );
    #pragma unroll  // K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*实部
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll// 虚部*虚部循环展开（注意符号取反）
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // 虚部*虚部，符号取反（因为i² = -1）
            d.real,
            a_desc_imag.chunk_descriptor(k),// a的虚部描述符
            b_desc_imag.chunk_descriptor(k),// b的虚部描述符
            1// 总是累加
        );
    }
    base::st_st(  // 执行实部*虚部
        d.imag,  // 结果虚部
        a_desc_real.chunk_descriptor(0),  // a的实部描述符
        b_desc_imag.chunk_descriptor(0),  // b的虚部描述符
        accumulate  // 是否累加
    );
    #pragma unroll  // K维度的累加循环展开
    for(int k = 1; k < K; k++) {
        base::st_st(// 继续累加实部*虚部
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1// 总是累加
        );
    }
    #pragma unroll // 虚部*实部循环展开
    for(int k = 0; k < K; k++) {
        base::st_st(// 执行虚部*实部
            d.imag,
            a_desc_imag.chunk_descriptor(k),// a的虚部描述符
            b_desc_real.chunk_descriptor(k),// b的实部描述符
            1// 总是累加
        );
    }
    mma_commit_group(); // 提交这一组WGMMA调用
}

// 非累加版本的mm_AtB（重置模式）
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);// 调用累加版本，但设置accumulate=0（重置模式）
}

// [(shared, shared) -> register] 版本 - 复数矩阵乘法
/**
 * @brief 使用warp group矩阵乘加(WGMMA)原语执行复数矩阵乘法，其中A和B已转置。
 *
 * 此函数计算复数共享图块`a`和`b`的外积，并将结果写入复数寄存器图块`d`。
 * 复数矩阵乘法公式：C = A^T * B^T，其中A和B是复数矩阵。
 * 计算方式：real = real_A * real_B - imag_A * imag_B
 *         imag = real_A * imag_B + imag_A * real_B
 *
 * @tparam D 目标复数寄存器图块类型。
 * @tparam A 源复数共享图块类型。
 * @tparam B 源复数共享图块类型。
 * @tparam fence 是否在执行前添加memory fence屏障(默认1:是)。
 * @tparam accumulate 是否将结果累加到`d`中(默认1:累加)或覆盖`d`(0:覆盖)。
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {
    // 静态断言检查，确保当前执行环境支持warp group操作
    KITTENS_CHECK_WARPGROUP
    
    // 计算M维度：A的列数除以每个tile的列维度
    constexpr int M = A::cols / kittens::TILE_COL_DIM<typename A::T>;
    static_assert(M == 4);  // 强制要求M维度为4
    
    // 检查输出寄存器图块高度是否为1（正确的行布局）
    static_assert(D::height == 1);
    
    // 计算N维度：B的行数除以每个tile的行维度
    constexpr int N = B::rows / kittens::TILE_ROW_DIM<typename B::T>;
    // 计算K维度：A的行数除以每个tile的行维度
    constexpr int K = A::rows / kittens::TILE_ROW_DIM<typename A::T>;
    
    // 验证K维度匹配：B的列数对应的tile数必须等于K
    static_assert(B::cols/kittens::TILE_COL_DIM<typename B::T> == K);
    
    // 验证A和B的数据类型必须相同
    static_assert(std::is_same_v<typename A::T, typename B::T>);

    // 类型别名定义
    using T_AB = A::T;  // 输入矩阵数据类型
    using T_D  = D::T;  // 输出寄存器数据类型
    
    // 静态断言：目前不支持fp8e4m3和fp8e5m2数据类型
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, 
                  "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, 
                  "Currently unsupported type");
    // 定义WGMMA操作的基础类型，配置为从共享内存到寄存器的操作
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    
    // 创建共享内存描述符，分别处理实部和虚部
    // 对于A矩阵（转置），使用列主序访问（参数1）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);
    // 对于B矩阵（转置），使用行主序访问（参数0）
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    // 如果需要，在执行前添加memory fence，确保内存操作顺序
    if constexpr (fence) { mma_fence(d); }

    // ================== 计算实部部分 ==================
    // 实部计算公式：real = real_A * real_B - imag_A * imag_B
    
    // 第一部分：计算 real_A * real_B（正贡献）
    base::st_st(
        d.real,                          // 目标寄存器（实部）
        a_desc_real.chunk_descriptor(0), // A矩阵实部的第0个数据块
        b_desc_real.chunk_descriptor(0), // B矩阵实部的第0个数据块
        accumulate                       // 是否累加到现有结果
    );

    // 循环处理剩余的K-1个数据块    
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }

    // 第二部分：计算 - imag_A * imag_B（负贡献）
    // 使用模板参数-1来反转符号（负数累加）
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>(  // INVERT THE SIGN OF THE IMAGINARY PART
            d.real,                          // 目标寄存器（实部）
            a_desc_imag.chunk_descriptor(k), // A矩阵虚部的第k个数据块
            b_desc_imag.chunk_descriptor(k), // B矩阵虚部的第k个数据块
            1                                // 累加（由于符号反转，实际上是减去）
        );
    }


    // ================== 计算虚部部分 ==================
    // 虚部计算公式：imag = real_A * imag_B + imag_A * real_B
    
    // 第一部分：计算 real_A * imag_B（正贡献）
    base::st_st(
        d.imag,                          // 目标寄存器（虚部）
        a_desc_real.chunk_descriptor(0), // A矩阵实部的第0个数据块
        b_desc_imag.chunk_descriptor(0), // B矩阵虚部的第0个数据块
        accumulate                       // 是否累加到现有结果
    );
    
    // 循环处理剩余的K-1个数据块
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1// 始终累加
        );
    }

    // 第二部分：计算 imag_A * real_B（正贡献）
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,     // 目标寄存器（虚部）
            a_desc_imag.chunk_descriptor(k),        // A矩阵虚部的第k个数据块
            b_desc_real.chunk_descriptor(k),        // B矩阵实部的第k个数据块
            1           // 累加
        );
    }
    mma_commit_group();     // 提交这一组WGMMA操作，确保所有操作完成
}

/**
 * @brief 复数矩阵乘法函数（覆盖模式），A和B已转置
 *
 * 此函数是mma_AtBt的包装，专用于覆盖模式（不累加到现有结果）
 *
 * @tparam D 目标复数寄存器图块类型
 * @tparam A 源复数共享图块类型
 * @tparam B 源复数共享图块类型
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    // 调用基础函数，设置fence=1（启用屏障），accumulate=0（覆盖模式）
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}

// ================== 额外的包装函数，提供更优雅的接口 ==================

/**
 * @brief 通用的矩阵乘法函数，支持转置选项
 *
 * 根据转置标志选择正确的矩阵乘法变体
 *
 * @tparam trans_A A矩阵的转置标志（transpose::T或transpose::N）
 * @tparam trans_B B矩阵的转置标志（transpose::T或transpose::N）
 * @tparam D 目标寄存器图块类型
 * @tparam A 源共享图块类型
 * @tparam B 源共享图块类型
 */
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b) {
    // 根据转置标志选择正确的矩阵乘法变体
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b);// A转置，B转置
        } else {
            mma_AtB(d, a, b);// A转置，B不转置
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b);// A不转置，B转置
        } else {
            mma_AB(d, a, b);// A不转置，B不转置
        }
    }
}

/**
 * @brief 通用的矩阵乘法函数（覆盖模式），支持转置选项
 *
 * 根据转置标志选择正确的矩阵乘法变体（覆盖模式）
 *
 * @tparam trans_A A矩阵的转置标志（transpose::T或transpose::N）
 * @tparam trans_B B矩阵的转置标志（transpose::T或transpose::N）
 * @tparam D 目标寄存器图块类型
 * @tparam A 源共享图块类型
 * @tparam B 源共享图块类型
 */
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mm(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mm_AtBt(d, a, b);
        } else {
            mm_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mm_ABt(d, a, b);
        } else {
            mm_AB(d, a, b);
        }
    }
}