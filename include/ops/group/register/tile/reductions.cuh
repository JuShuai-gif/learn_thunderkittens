/**
 * @file
 * @brief 归约操作：将tile映射为向量的操作
 * 
 * 这个文件包含了对tile进行归约操作的函数，包括行归约和列归约。
 * 归约操作将多维数据聚合为一维向量，常用于计算和、最大值、最小值等统计操作。
 */

/**
 * @brief 对行主序布局的矩阵执行行归约操作
 *
 * 这个函数模板使用指定的操作对矩阵的行进行并行归约。
 * 它利用warp shuffle函数实现高效的warp内部通信。
 *
 * @tparam op 用于归约的操作类型
 * @tparam V 行累加器的向量类型
 * @tparam T 具有行布局的矩阵类型
 * @tparam reset 布尔标志，指示是否重置累加器（忽略src_accum）
 * @param[out] row_accum 存储归约结果的累加器
 * @param[in] src 执行归约的源矩阵
 * @param[in] src_accum 累加器的初始值，当reset为false时使用
 * 
 * 行归约将矩阵的每一行压缩为一个值，生成一个列向量。
 * 对于行主序布局，数据在内存中按行存储，因此行归约更自然。
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // 我喜欢这些静态断言，因为当出错时它们能提供更详细的错误信息
    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    // 静态断言：确保向量数据类型与tile的数据类型兼容
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量的外部维度（行数）与tile的高度匹配
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x1C; // 二进制11100，即28，用于warp内的数据收集
    #pragma unroll
    for(int i = 0; i < src.height; i++) {  // 遍历每一行
        // 初始化累加器：处理每个tile的前两个打包数据
        dtype accum_top_row    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[2]);
        dtype accum_bottom_row = op::template op<dtype>(src.tiles[i][0].data[1], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {  // 遍历同一行的其他tile
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k+=2) {  // 每次处理两个打包数据
                // 累加奇数索引的打包数据（上半行）
                accum_top_row    = op::template op<dtype>(accum_top_row,    src.tiles[i][j].data[k+0]);
                // 累加偶数索引的打包数据（下半行）                
                accum_bottom_row = op::template op<dtype>(accum_bottom_row, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        // 对上半行的两个分量进行归约
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_top_row.x,    accum_top_row.y);
        // 对下半行的两个分量进行归约
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_bottom_row.x, accum_bottom_row.y);

        // 现在我们需要进行一些shuffle操作，让每个线程都得到正确的结果

        // 第一步shuffle：跨2个线程进行归约
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        // 第二步shuffle：跨1个线程进行归约
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));
        // 将结果广播到leader线程
        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {            
            // 重置累加器：直接存储归约结果
            row_accum[i][0] = accum_packed;
        }
        else {
            // 累加到现有累加器：将归约结果与初始值结合            
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
}
/**
 * @brief 对列主序布局的矩阵执行行归约操作
 *
 * 这个函数模板使用指定的操作对矩阵的行进行并行归约。
 * 它利用warp shuffle函数实现高效的warp内部通信，并针对列主序矩阵进行了优化。
 *
 * @tparam op 用于归约的操作类型
 * @tparam V 行累加器的向量类型
 * @tparam T 具有列布局的矩阵类型
 * @tparam reset 布尔标志，指示是否重置累加器（忽略src_accum）
 * @param[out] row_accum 存储归约结果的累加器
 * @param[in] src 执行归约的源矩阵
 * @param[in] src_accum 累加器的初始值，当reset为false时使用
 * 
 * 对于列主序布局，数据在内存中按列存储，行归略需要跨列收集数据。
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // 我喜欢这些静态断言，因为当出错时它们能提供更详细的错误信息
    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    // 静态断言：确保向量数据类型与tile的数据类型兼容
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量的外部维度（行数）与tile的高度匹配
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x3; // 二进制00011，即3，用于warp内的数据收集
    #pragma unroll
    for(int i = 0; i < src.height; i++) {  // 遍历每一行
        // 初始化累加器：处理每个tile的前四个打包数据
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {  // 遍历同一行的其他tile
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile/2; k++) {  // 每次处理两个打包数据
                // 累加左侧的打包数据（上半行）
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k+0]);
                // 累加右侧的打包数据（下半行）
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k+2]);
            }
        }

        // 现在我们需要进行一些shuffle操作，让每个线程都得到正确的结果

        // 对上半行数据进行多级shuffle归约
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 16));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 4));

        // 对下半行数据进行多级shuffle归约
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 16));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 4));

        // 将结果广播到leader线程
        accum_top_rows    = packed_shfl_sync(MASK_ALL, accum_top_rows,    leader);
        accum_bottom_rows = packed_shfl_sync(MASK_ALL, accum_bottom_rows, leader);

        if(reset) {
            // 重置累加器：直接存储归约结果
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        }
        else {
            // 累加到现有累加器：将归约结果与初始值结合
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
}
/* ----------  列归约操作  ---------- */

/**
 * @brief 对行主序布局的矩阵执行列归约操作
 *
 * 这个函数模板使用指定的操作对矩阵的列进行并行归约。
 * 它利用warp shuffle函数实现高效的warp内部通信，并针对行主序矩阵进行了优化。
 *
 * @tparam op 用于归约的操作类型
 * @tparam V 列累加器的向量类型
 * @tparam T 具有行布局的矩阵类型
 * @tparam reset 布尔标志，指示是否重置累加器（忽略src_accum）
 * @param[out] col_accum 存储归约结果的累加器
 * @param[in] src 执行归约的源矩阵
 * @param[in] src_accum 累加器的初始值，当reset为false时使用
 * 
 * 列归约将矩阵的每一列压缩为一个值，生成一个行向量。
 * 对于行主序布局，列归约需要跨行收集数据。
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // 我喜欢这些静态断言，因为当出错时它们能提供更详细的错误信息
    KITTENS_CHECK_WARP  // 检查warp配置的宏
    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    // 静态断言：确保向量数据类型与tile的数据类型兼容
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量的外部维度（列数）与tile的宽度匹配
    static_assert(V::outer_dim == T::width);

    using dtype = V::dtype;

    const int leader = threadIdx.x & 0x3; // 二进制00011，即3，用于warp内的数据收集
    #pragma unroll
    for(int j = 0; j < src.width; j++) {  // 遍历每一列
        // 初始化累加器：处理每个tile的前四个打包数据
        dtype accum_left_cols  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[1]);
        dtype accum_right_cols = op::template op<dtype>(src.tiles[0][j].data[2], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < src.height; i++) {  // 遍历同一列的其他tile
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile/2; k++) {  // 每次处理两个打包数据
                // 累加左侧的打包数据（左列）
                accum_left_cols  = op::template op<dtype>(accum_left_cols,  src.tiles[i][j].data[k+0]);
                // 累加右侧的打包数据（右列）
                accum_right_cols = op::template op<dtype>(accum_right_cols, src.tiles[i][j].data[k+2]);
            }
        }

        // 现在我们需要进行一些shuffle操作，让每个线程都得到正确的结果

        // 对左列数据进行多级shuffle归约
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 16));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 8));
        accum_left_cols = op::template op<dtype>(accum_left_cols, packed_shfl_down_sync(MASK_ALL, accum_left_cols, 4));

        // 对右列数据进行多级shuffle归约
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 16));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 8));
        accum_right_cols = op::template op<dtype>(accum_right_cols, packed_shfl_down_sync(MASK_ALL, accum_right_cols, 4));

        // 将结果广播到leader线程
        accum_left_cols  = packed_shfl_sync(MASK_ALL, accum_left_cols,  leader);
        accum_right_cols = packed_shfl_sync(MASK_ALL, accum_right_cols, leader);

        if(reset) {
            // 重置累加器：直接存储归约结果
            col_accum[j][0] = accum_left_cols;
            col_accum[j][1] = accum_right_cols;
        }
        else {
            // 累加到现有累加器：将归约结果与初始值结合
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
        }
    }
}
/**
 * @brief 对列主序布局的矩阵执行列归约操作
 *
 * 这个函数模板使用指定的操作对矩阵的列进行并行归约。
 * 它利用warp shuffle函数实现高效的warp内部通信，并针对列主序矩阵进行了优化。
 *
 * @tparam op 用于归约的操作类型
 * @tparam V 列累加器的向量类型
 * @tparam T 具有列布局的矩阵类型
 * @tparam reset 布尔标志，指示是否重置累加器（忽略src_accum）
 * @param[out] col_accum 存储归约结果的累加器
 * @param[in] src 执行归约的源矩阵
 * @param[in] src_accum 累加器的初始值，当reset为false时使用
 * 
 * 对于列主序布局，数据在内存中按列存储，因此列归约更自然。
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // 我喜欢这些静态断言，因为当出错时它们能提供更详细的错误信息
    KITTENS_CHECK_WARP
    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    // 静态断言：确保向量数据类型与tile的数据类型兼容
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量的外部维度（列数）与tile的宽度匹配
    static_assert(V::outer_dim == T::width);

    using dtype = V::dtype;
    const int leader = threadIdx.x & 0x1C; // 二进制11100，即28，用于warp内的数据收集
    #pragma unroll
    for(int j = 0; j < src.width; j++) {  // 遍历每一列（注意：现在宽度是外层循环）
        // 初始化累加器：处理每个tile的前两个打包数据
        dtype accum_left_col  = op::template op<dtype>(src.tiles[0][j].data[0], src.tiles[0][j].data[2]);
        dtype accum_right_col = op::template op<dtype>(src.tiles[0][j].data[1], src.tiles[0][j].data[3]);
        #pragma unroll
        for(int i = 1; i < src.height; i++) {  // 遍历同一列的其他tile（高度是内层循环）
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k+=2) {  // 每次处理两个打包数据
                // 累加奇数索引的打包数据（左列）
                accum_left_col  = op::template op<dtype>(accum_left_col,  src.tiles[i][j].data[k+0]);
                // 累加偶数索引的打包数据（右列）
                accum_right_col = op::template op<dtype>(accum_right_col, src.tiles[i][j].data[k+1]);
            }
        }
        dtype accum_packed;
        // 对左列的两个分量进行归约
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_left_col.x,  accum_left_col.y);
        // 对右列的两个分量进行归约
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_right_col.x, accum_right_col.y);

        // 现在我们需要进行一些shuffle操作，让每个线程都得到正确的结果

        // 第一步shuffle：跨2个线程进行归约
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        // 第二步shuffle：跨1个线程进行归约
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));

        // 将结果广播到leader线程
        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);

        if(reset) {
            // 重置累加器：直接存储归约结果
            col_accum[j][0] = accum_packed;
        }
        else {
            // 累加到现有累加器：将归约结果与初始值结合
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_packed);
        }
    }
}

/* ----------  为美观性而设的包装函数  ---------- */

// 两操作数行归约函数。（累加并替换原有值。）

/**
 * @brief 计算源寄存器tile每行的最大值，并存储到行累加器列向量中（替换模式）。
 * 此函数使用行归约操作，将每行元素与当前累加器值（作为初始值）进行比较取最大值，并替换累加器原有值。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);// true表示替换模式
}
/**
 * @brief 计算源寄存器tile每行的最小值，并存储到行累加器列向量中（替换模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief 计算源寄存器tile每行的和，并存储到行累加器列向量中（替换模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief 计算源寄存器tile每行的乘积，并存储到行累加器列向量中（替换模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}

// 三操作数行归约函数。（累加到已有值上。）

/**
 * @brief 计算源寄存器tile每行的最大值，并与初始累加器值合并，存储到行累加器列向量中（累加模式）。
 * 此函数使用行归约操作，将每行元素与初始累加器值src_accum进行比较取最大值，结果存入row_accum。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief 计算源寄存器tile每行的最小值，并与初始累加器值合并，存储到行累加器列向量中（累加模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}

/**
 * @brief 计算源寄存器tile每行的和，并与初始累加器值相加，存储到行累加器列向量中（累加模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}

/**
 * @brief 计算源寄存器tile每行的乘积，并与初始累加器值相乘，存储到行累加器列向量中（累加模式）。
 *
 * @tparam V 行累加器的向量类型。
 * @tparam T 矩阵类型。
 * @param[out] row_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// 两操作数列归约函数。（累加并替换原有值。）

/**
 * @brief 计算源寄存器tile每列的最大值，并存储到列累加器行向量中（替换模式）。
 * 此函数使用列归约操作，将每列元素与当前累加器值（作为初始值）进行比较取最大值，并替换累加器原有值。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief 计算源寄存器tile每列的最小值，并存储到列累加器行向量中（替换模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief 计算源寄存器tile每列的和，并存储到列累加器行向量中（替换模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief 计算源寄存器tile每列的乘积，并存储到列累加器行向量中（替换模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// 三操作数列归约函数。（累加到已有值上。）

/**
 * @brief 计算源寄存器tile每列的最大值，并与初始累加器值合并，存储到列累加器行向量中（累加模式）。
 * 此函数使用列归约操作，将每列元素与初始累加器值src_accum进行比较取最大值，结果存入col_accum。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief 计算源寄存器tile每列的最小值，并与初始累加器值合并，存储到列累加器行向量中（累加模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief 计算源寄存器tile每列的和，并与初始累加器值相加，存储到列累加器行向量中（累加模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief 计算源寄存器tile每列的乘积，并与初始累加器值相乘，存储到列累加器行向量中（累加模式）。
 *
 * @tparam V 列累加器的向量类型（此处为行向量）。
 * @tparam T 矩阵类型。
 * @param[out] col_accum 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值，用于累加到已有值上的模式。
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

// 模板化版本的归约函数（根据轴选择行归约或列归约）

/**
 * @brief 根据指定轴执行最大值归约，使用累加模式（即保留原有累加器值）。
 * 
 * @tparam ax 归约轴（axis::COL表示按列归约，其他表示按行归约）。
 * @tparam RV 累加器向量类型。
 * @tparam T 矩阵类型。
 * @param[out] dst 存储归约结果的累加器。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值。
 */

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);// 按列归约（对应行最大值）
    else col_max(dst, src, src_accum); // 按行归约（对应列最大值）
}

/**
 * @brief 根据指定轴执行最大值归约，使用累加模式，返回新的累加器。
 * 
 * @tparam ax 归约轴。
 * @tparam RV 累加器向量类型。
 * @tparam T 矩阵类型。
 * @param[in] src 执行归约操作的源矩阵。
 * @param[in] src_accum 累加器的初始值。
 * @return RV 包含归约结果的新累加器。
 */
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto max(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}

/**
 * @brief 根据指定轴执行最大值归约，使用替换模式（即忽略原有累加器值）。
 * 
 * @tparam ax 归约轴。
 * @tparam RV 累加器向量类型。
 * @tparam T 矩阵类型。
 * @param[out] dst 存储归约结果的累加器（将替换原有值）。
 * @param[in] src 执行归约操作的源矩阵。
 */
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void max(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}

/**
 * @brief 根据指定轴执行最大值归约，使用替换模式，返回新的累加器。
 * 
 * @tparam ax 归约轴。
 * @tparam T 矩阵类型。
 * @param[in] src 执行归约操作的源矩阵。
 * @return 自动推导的累加器类型（如果是按列归约则使用列向量，否则使用行向量）。
 */
template<int ax, ducks::rt::all T>
__device__ static inline auto max(const T &src) {
    // 根据归约轴选择累加器类型：按列归约使用列向量，否则使用行向量
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

// 最小值归约函数模板（结构与最大值类似）
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto min(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void min(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto min(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto sum(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void sum(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto sum(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src, const RV &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline auto prod(const T &src, const RV &src_accum) {
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::rv::all RV, ducks::rt::all T>
__device__ static inline void prod(RV &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::rt::all T>
__device__ static inline auto prod(const T &src) {
    using RV = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    RV dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}