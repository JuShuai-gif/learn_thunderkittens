/**
 * @file
 * @brief 地图操作：图块之间的操作，以及将向量应用于图块的操作
 * 
 * 这个文件包含了对tile（瓦片/图块）进行各种映射操作的函数，
 * 包括一元操作、二元操作和自定义lambda操作。
 * 这些操作是独立于布局的，适用于所有tile类型。
 */

/* ----------  统一的tile映射操作（独立于布局）  ---------- */

/**
 * @brief 对tile中的每个元素应用一元操作
 *
 * @tparam op 要应用的一元操作类型
 * @tparam T tile类型，必须满足ducks::rt::all概念
 * @param dst[out] 存储操作结果的目标tile
 * @param src[in] 应用操作的源tile
 * 
 * 这个函数遍历tile中的每个元素，对每个元素应用指定的一元操作，
 * 并将结果存储到目标tile中。使用#pragma unroll进行循环展开优化。
 */
template<typename op, ducks::rt::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历tile的行
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历tile的列
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {// 遍历每个位置打包的数据
                // 对每个打包的数据元素应用一元操作
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief 对tile中的每个元素应用带标量参数的二元操作
 *
 * @tparam op 要应用的二元操作类型
 * @tparam T tile类型，必须满足ducks::rt::all概念
 * @param dst[out] 存储操作结果的目标tile
 * @param src[in] 应用操作的源tile
 * @param param[in] 二元操作的标量参数
 * 
 * 这个函数将标量参数与tile中的每个元素进行二元操作。
 * 适用于需要将每个元素与同一个常数进行运算的场景。
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                // 对每个元素应用二元操作，使用相同的标量参数
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}

/**
 * @brief 对tile中的每个元素应用带解包标量参数的二元操作
 *
 * @tparam op 要应用的二元操作类型
 * @tparam T tile类型，必须满足ducks::rt::all概念
 * @param dst[out] 存储操作结果的目标tile
 * @param src[in] 应用操作的源tile
 * @param param[in] 二元操作的解包标量参数
 * 
 * 这个函数接受解包类型的参数，并在内部将其打包后应用操作。
 * 主要用于处理16位等需要打包/解包的数据类型。
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // 优化编译器应该能在32位情况下消除这个打包操作，但在16位情况下不会
    // 将解包参数打包后调用上面的bin_map函数
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/**
 * @brief 对两个tile逐元素应用二元操作
 *
 * @tparam op 要应用的二元操作类型
 * @tparam T tile类型，必须满足ducks::rt::all概念
 * @param dst[out] 存储操作结果的目标tile
 * @param lhs[in] 二元操作的左侧源tile
 * @param rhs[in] 二元操作的右侧源tile
 * 
 * 这个函数对两个相同形状的tile进行逐元素的二元操作。
 * 适用于两个tile之间的元素级运算。
 */
template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                // 对两个tile对应位置的元素应用二元操作
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief 使用自定义lambda函数对tile中的每个元素应用操作
 *
 * @tparam RT tile类型，必须满足ducks::rt::all概念
 * @tparam Lambda lambda函数类型
 * @param dst[out] 存储操作结果的目标tile
 * @param src[in] 应用操作的源tile
 * @param lambda 要应用的lambda函数，接受行、列和值参数
 * 
 * 这个函数提供了最大的灵活性，允许用户自定义对每个元素的操作。
 * lambda函数接收当前元素的行索引、列索引和值，返回新的值。
 * 根据tile的布局类型（行布局或列布局）计算实际的行列索引。
 */
template<ducks::rt::all RT, typename Lambda>
__device__ static inline void apply(RT &dst, const RT &src, Lambda &&lambda) {
    // 计算行偏移：在多warp分组的情况下，每个warp处理不同的行
    int row_offset = 0;
    if constexpr(GROUP_WARPS > 1) {
        row_offset = warpid()*RT::height;
    }

    // 静态断言：不允许对8位类型应用lambda操作
    static_assert(sizeof(RT::T) != 1, "Cannot apply lambda to 8-bit types");

    // 根据布局类型计算实际的行列索引
    if constexpr (ducks::rt::row_layout<RT>) {// 行主序布局
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    // 计算实际的行索引（考虑打包数据的布局）
                    int row = row_offset + i*TILE_ROW_DIM<typename RT::T> + (k%2) * (TILE_ROW_DIM<typename RT::T>/2) + ::kittens::laneid()/4;
                    // 计算实际的列索引
                    int col = j*TILE_COL_DIM<typename RT::T> + (k/2) * (TILE_COL_DIM<typename RT::T>/2) + (::kittens::laneid()%4)*2;
                    // 对打包数据中的x分量应用lambda函数
                    dst.tiles[i][j].data[k].x = lambda(row, col+0, src.tiles[i][j].data[k].x);
                    // 对打包数据中的y分量应用lambda函数
                    dst.tiles[i][j].data[k].y = lambda(row, col+1, src.tiles[i][j].data[k].y);
                }
            }
        }
    }
    else {// 列主序布局（或其他布局）
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    // 列主序布局下的行索引计算
                    int row = row_offset + i*TILE_ROW_DIM<typename RT::T> + (k/2) * (TILE_ROW_DIM<typename RT::T>/2) + (::kittens::laneid()%4)*2;
                    // 列主序布局下的列索引计算
                    int col = j*TILE_COL_DIM<typename RT::T> + (k%2) * (TILE_COL_DIM<typename RT::T>/2) + ::kittens::laneid()/4;
                    // 对打包数据中的x分量应用lambda函数
                    dst.tiles[i][j].data[k].x = lambda(row+0, col, src.tiles[i][j].data[k].x);
                    // 对打包数据中的y分量应用lambda函数
                    dst.tiles[i][j].data[k].y = lambda(row+1, col, src.tiles[i][j].data[k].y);
                }
            }
        }
    }
}

/**
 * @brief apply函数的返回值版本
 *
 * @tparam RT tile类型，必须满足ducks::rt::all概念
 * @tparam Lambda lambda函数类型
 * @param src[in] 应用操作的源tile
 * @param lambda 要应用的lambda函数，接受行、列和值参数
 * @return RT 包含操作结果的新tile
 * 
 * 这个函数是apply函数的功能性版本，它创建一个新的tile来存储结果，
 * 然后调用上述的apply函数，最后返回结果tile。
 * 提供更函数式的编程接口。
 */
template<ducks::rt::all RT, typename Lambda>
__device__ static inline RT apply(const RT &src, Lambda &&lambda) {
    RT dst;// 创建目标tile
    apply<RT, Lambda>(dst, src, std::forward<Lambda>(lambda));// 调用apply函数
    return dst;// 返回结果
}

/* ----------  Row tile maps  ----------*/

/**
 * @brief 对行主序布局的tile按行应用操作。
 * 将行向量row_values中的值广播到tile的每一行，与src tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的操作类型（如加法、乘法等）。
 * @tparam T 行主序布局的tile类型。
 * @tparam V 列向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param src[in] 输入tile，操作将应用于此tile。
 * @param row_values[in] 包含每行应用值的列向量。
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    // 静态断言：确保向量高度与tile高度匹配
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;
    
    // 循环遍历tile的每一行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 将行向量中的两个值打包（针对eager模式）
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); // 第一个值（顶行）
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // 第二个值（底行）

        // 循环遍历tile的每一列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            // 循环处理每个tile内部的打包数据，每次处理2个元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                // 对打包数据中的偶数索引元素应用操作，使用顶行值
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_top_row);
                // 对打包数据中的奇数索引元素应用操作，使用底行值
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}


/**
 * @brief 对列主序布局的tile按行应用操作。
 * 将行向量row_values中的值广播到tile的每一行，与src tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的操作类型（如加法、乘法等）。
 * @tparam T 列主序布局的tile类型。
 * @tparam V 列向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param src[in] 输入tile，操作将应用于此tile。
 * @param row_values[in] 包含每行应用值的列向量。
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {

    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    // 静态断言：确保向量高度与tile高度匹配
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    // 循环遍历tile的每一行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历tile的每一列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环处理每个tile内部的打包数据，每次处理一半元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                // 对前一半数据应用操作，使用行向量的第一个值
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], row_values[i][0]);
                // 对后一半数据应用操作，使用行向量的第二个值（跨步访问）
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}


// 三元操作行映射函数。主要用于FMA（乘加）指令。

/**
 * @brief 对行主序布局的两个tile按行应用三元操作。
 * 将行向量row_values中的值作为第三个操作数，与a和b tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的三元操作类型（如乘加）。
 * @tparam T 行主序布局的tile类型。
 * @tparam V 列向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param a[in] 第一个输入tile。
 * @param b[in] 第二个输入tile。
 * @param row_values[in] 包含每行应用值的列向量（作为第三个操作数）。
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量高度与tile高度匹配
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;

    // 循环遍历tile的每一行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 将行向量中的两个值打包（针对eager模式）
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x); // 第一个值（顶行）
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y); // 第二个值（底行）

        // 循环遍历tile的每一列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环处理每个tile内部的打包数据，每次处理2个元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                // 对打包数据中的偶数索引元素应用三元操作，使用顶行值作为第三个操作数
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_top_row);
                // 对打包数据中的奇数索引元素应用三元操作，使用底行值作为第三个操作数
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_bottom_row);
            }
        }
    }
}


/**
 * @brief 对列主序布局的两个tile按行应用三元操作。
 * 将行向量row_values中的值作为第三个操作数，与a和b tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的三元操作类型（如乘加）。
 * @tparam T 列主序布局的tile类型。
 * @tparam V 列向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param a[in] 第一个输入tile。
 * @param b[in] 第二个输入tile。
 * @param row_values[in] 包含每行应用值的列向量（作为第三个操作数）。
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    // 静态断言：确保向量布局与tile的列向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量高度与tile高度匹配
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;
    // 循环遍历tile的每一行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历tile的每一列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环处理每个tile内部的打包数据，每次处理一半元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                // 对前一半数据应用三元操作，使用行向量的第一个值作为第三个操作数
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], row_values[i][0]);
                // 对后一半数据应用三元操作，使用行向量的第二个值作为第三个操作数（跨步访问）
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], row_values[i][1]);
            }
        }
    }
}

/* ----------  列主序tile映射操作  ----------*/

/**
 * @brief 对行主序布局的tile按列应用操作。
 * 将列向量col_values中的值广播到tile的每一列，与src tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的操作类型（如加法、乘法等）。
 * @tparam T 行主序布局的tile类型。
 * @tparam V 行向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param src[in] 输入tile，操作将应用于此tile。
 * @param col_values[in] 包含每列应用值的行向量。
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP// 检查当前是否在warp内执行的宏

    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量宽度与tile宽度匹配
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    // 循环遍历tile的每一列
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        // 循环遍历tile的每一行
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            // 循环处理每个tile内部的打包数据，每次处理一半元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                // 对前一半数据应用操作，使用列向量的第一个值
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], col_values[j][0]);
                // 对后一半数据应用操作，使用列向量的第二个值（跨步访问）
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(src.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief 对列主序布局的tile按列应用操作。
 * 将列向量col_values中的值广播到tile的每一列，与src tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的操作类型（如加法、乘法等）。
 * @tparam T 列主序布局的tile类型。
 * @tparam V 行向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param src[in] 输入tile，操作将应用于此tile。
 * @param col_values[in] 包含每列应用值的行向量。
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &src, const V &col_values) {
    KITTENS_CHECK_WARP  // 检查当前是否在warp内执行的宏

    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量宽度与tile宽度匹配
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    // 循环遍历tile的每一列
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        // 将列向量中的两个值打包（针对eager模式）
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x);  // 第一个值（左列）
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y);  // 第二个值（右列）
        
        // 循环遍历tile的每一行
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            // 循环处理每个tile内部的打包数据，每次处理2个元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                // 对打包数据中的偶数索引元素应用操作，使用左列值
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(src.tiles[i][j].data[k+0], packed_left_col);
                // 对打包数据中的奇数索引元素应用操作，使用右列值
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(src.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}

// 三元操作列映射函数
/**
 * @brief 对行主序布局的两个tile按列应用三元操作。
 * 将列向量col_values中的值作为第三个操作数，与a和b tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的三元操作类型（如乘加）。
 * @tparam T 行主序布局的tile类型。
 * @tparam V 行向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param a[in] 第一个输入tile。
 * @param b[in] 第二个输入tile。
 * @param col_values[in] 包含每列应用值的行向量（作为第三个操作数）。
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP  // 检查当前是否在warp内执行的宏

    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量宽度与tile宽度匹配
    static_assert(V::outer_dim == T::width);

    using dtype = T::dtype;

    // 循环遍历tile的每一列
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        // 循环遍历tile的每一行
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            // 循环处理每个tile内部的打包数据，每次处理一半元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile/2; k++) {
                // 对前一半数据应用三元操作，使用列向量的第一个值作为第三个操作数
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], col_values[j][0]);
                // 对后一半数据应用三元操作，使用列向量的第二个值作为第三个操作数（跨步访问）
                dst.tiles[i][j].data[k+2] = op::template op<dtype>(a.tiles[i][j].data[k+2], b.tiles[i][j].data[k+2], col_values[j][1]);
            }
        }
    }
}
/**
 * @brief 对列主序布局的两个tile按列应用三元操作。
 * 将列向量col_values中的值作为第三个操作数，与a和b tile执行逐元素操作，结果存入dst tile。
 *
 * @tparam op 要应用的三元操作类型（如乘加）。
 * @tparam T 列主序布局的tile类型。
 * @tparam V 行向量类型。
 * @param dst[out] 存储操作结果的输出tile。
 * @param a[in] 第一个输入tile。
 * @param b[in] 第二个输入tile。
 * @param col_values[in] 包含每列应用值的行向量（作为第三个操作数）。
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {
    KITTENS_CHECK_WARP  // 检查当前是否在warp内执行的宏

    // 静态断言：确保数据类型一致
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    // 静态断言：确保向量布局与tile的行向量布局兼容
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    // 静态断言：确保向量宽度与tile宽度匹配
    static_assert(V::outer_dim == T::width);

    // 循环遍历tile的每一列
    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        // 将列向量中的两个值打包（针对eager模式）
        dtype packed_left_col  = base_types::packing<dtype>::pack(col_values[j][0].x);  // 第一个值（左列）
        dtype packed_right_col = base_types::packing<dtype>::pack(col_values[j][0].y);  // 第二个值（右列）
        
        // 循环遍历tile的每一行
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            // 循环处理每个tile内部的打包数据，每次处理2个元素
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k+=2) {
                // 对打包数据中的偶数索引元素应用三元操作，使用左列值作为第三个操作数
                dst.tiles[i][j].data[k+0] = op::template op<dtype>(a.tiles[i][j].data[k+0], b.tiles[i][j].data[k+0], packed_left_col);
                // 对打包数据中的奇数索引元素应用三元操作，使用右列值作为第三个操作数
                dst.tiles[i][j].data[k+1] = op::template op<dtype>(a.tiles[i][j].data[k+1], b.tiles[i][j].data[k+1], packed_right_col);
            }
        }
    }
}


/* ----------  美化包装函数  ---------- */

// 所有繁琐的类型限定符*应该*会在编译时自动推断。
// 因此，语法应该只是 kittens::add_row(tile, colvec);

/**
 * @brief 将tile的所有元素设置为零
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * 
 * 这是一个方便的函数包装器，调用一元映射操作将tile的所有元素设为零。
 * 注意：dst也作为源参数传入，但实际上源值未被使用。
 */
template<ducks::rt::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief 将tile的所有元素设置为一
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * 
 * 调用一元映射操作将tile的所有元素设为一。
 */
template<ducks::rt::all T>
__device__ static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief 将tile的所有元素设置为正无穷
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * 
 * 调用一元映射操作将tile的所有元素设为正无穷。
 * 在浮点数运算中，常用于表示上界或未定义值。
 */
template<ducks::rt::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief 将tile的所有元素设置为负无穷
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * 
 * 调用一元映射操作将tile的所有元素设为负无穷。
 * 在浮点数运算中，常用于表示下界或未定义值。
 */
template<ducks::rt::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief 对tile的每个元素应用指数函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用指数函数的源tile
 * 
 * 计算自然指数 e^x，其中 x 是源tile中的元素值。
 */
template<ducks::rt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}

/**
 * @brief exp函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用指数函数的源tile
 * @return T 包含指数运算结果的新tile
 * 
 * 创建新tile存储结果，然后调用exp函数，最后返回结果tile。
 */
template<ducks::rt::all T>
__device__ static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}

/**
 * @brief 对tile的每个元素应用以2为底的指数函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用指数函数的源tile
 * 
 * 计算 2^x，其中 x 是源tile中的元素值。
 * 在计算机科学中常用，因为2的幂次运算通常更高效。
 */
template<ducks::rt::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}

/**
 * @brief exp2函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用指数函数的源tile
 * @return T 包含指数运算结果的新tile
 */
template<ducks::rt::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}

/**
 * @brief 对tile的每个元素应用自然对数函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用对数函数的源tile
 * 
 * 计算 ln(x)，其中 x 是源tile中的元素值。
 * 在科学计算和机器学习中广泛应用。
 */
template<ducks::rt::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}

/**
 * @brief log函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用对数函数的源tile
 * @return T 包含对数运算结果的新tile
 */
template<ducks::rt::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}

/**
 * @brief 对tile的每个元素应用以2为底的对数函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用对数函数的源tile
 * 
 * 计算 log₂(x)，其中 x 是源tile中的元素值。
 * 在信息论和计算机算法中常用。
 */
template<ducks::rt::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}

/**
 * @brief log2函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用对数函数的源tile
 * @return T 包含对数运算结果的新tile
 */
template<ducks::rt::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}

/**
 * @brief 对tile的每个元素应用绝对值函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用绝对值函数的源tile
 * 
 * 计算 |x|，其中 x 是源tile中的元素值。
 * 返回每个元素的非负值。
 */
template<ducks::rt::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}

/**
 * @brief abs函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用绝对值函数的源tile
 * @return T 包含绝对值运算结果的新tile
 */
template<ducks::rt::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}

/**
 * @brief 对tile的每个元素应用整流线性单元(ReLU)函数
 *
 * @tparam T tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用ReLU函数的源tile
 * 
 * 计算 max(0, x)，其中 x 是源tile中的元素值。
 * 这是深度学习中最常用的激活函数之一。
 */
template<ducks::rt::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}

/**
 * @brief relu函数的返回值版本
 *
 * @tparam T tile类型
 * @param src[in] 应用ReLU函数的源tile
 * @return T 包含ReLU运算结果的新tile
 */
template<ducks::rt::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}

/**
 * @brief 将一个tile的元素复制到另一个tile
 *
 * @tparam T 目标tile类型
 * @tparam U 源tile类型
 * @param dst[out] 存储复制结果的目标tile
 * @param src[in] 要复制的源tile
 * 
 * 调用二元映射操作，将源tile的值复制到目标tile。
 * 注意：使用了copy2操作，可能支持类型转换或特定的复制语义。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}

/**
 * @brief 对两个tile逐元素应用最大值操作，或对tile和标量应用最大值操作
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 操作的左侧源tile
 * @param rhs[in] 操作的右侧源tile或标量
 * 
 * 计算每个位置的最大值：dst = max(lhs, rhs)。
 * 如果rhs是标量，则将lhs的每个元素与该标量比较取最大值。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}

/**
 * @brief max函数的返回值版本
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param lhs[in] 操作的左侧源tile
 * @param rhs[in] 操作的右侧源tile或标量
 * @return T 包含最大值运算结果的新tile
 */
template<ducks::rt::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}

/**
 * @brief 对两个tile逐元素应用最小值操作，或对tile和标量应用最小值操作
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 操作的左侧源tile
 * @param rhs[in] 操作的右侧源tile或标量
 * 
 * 计算每个位置的最小值：dst = min(lhs, rhs)。
 * 如果rhs是标量，则将lhs的每个元素与该标量比较取最小值。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}

/**
 * @brief min函数的返回值版本
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param lhs[in] 操作的左侧源tile
 * @param rhs[in] 操作的右侧源tile或标量
 * @return T 包含最小值运算结果的新tile
 */
template<ducks::rt::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}

/**
 * @brief 两个tile逐元素相加，或将标量与tile的每个元素相加
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 加法的左侧源tile
 * @param rhs[in] 加法的右侧源tile或标量
 * 
 * 计算 dst = lhs + rhs。
 * 这是最基本的算术运算之一。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}

/**
 * @brief 两个tile逐元素相减，或将标量与tile的每个元素相减
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 减法的左侧源tile
 * @param rhs[in] 减法的右侧源tile或标量
 * 
 * 计算 dst = lhs - rhs。
 * 注意：运算顺序是 lhs - rhs。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}

/**
 * @brief 两个tile逐元素相乘，或将标量与tile的每个元素相乘
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 乘法的左侧源tile
 * @param rhs[in] 乘法的右侧源tile或标量
 * 
 * 计算 dst = lhs × rhs。
 * 常用于缩放操作或元素级乘积。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}

/**
 * @brief 两个tile逐元素相除，或将标量与tile的每个元素相除
 *
 * @tparam T tile类型
 * @tparam U 第二个操作数类型，可以是tile或标量
 * @param dst[out] 存储结果的目标tile
 * @param lhs[in] 除法的左侧源tile
 * @param rhs[in] 除法的右侧源tile或标量
 * 
 * 计算 dst = lhs ÷ rhs。
 * 注意：运算顺序是 lhs 除以 rhs。
 */
template<ducks::rt::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

/**
 * @brief 将行值加到tile的每一行
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用加法操作的源tile
 * @param row_values[in] 包含要加到每一行的值的列向量
 * 
 * 调用row_map操作，将列向量中的每个值加到tile的对应行的所有元素上。
 * 例如，row_values[0]会被加到第一行的所有元素上。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}

/**
 * @brief 从tile的每一行减去行值
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用减法操作的源tile
 * @param row_values[in] 包含要从每一行减去的值的列向量
 * 
 * 计算 dst = src - row_values（按行广播）。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}

/**
 * @brief 将tile的每一行乘以行值
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用乘法操作的源tile
 * @param row_values[in] 包含要乘以每一行的值的列向量
 * 
 * 计算 dst = src × row_values（按行广播）。
 * 常用于行缩放操作。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}

/**
 * @brief 将tile的每一行除以行值
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用除法操作的源tile
 * @param row_values[in] 包含要除以每一行的值的列向量
 * 
 * 计算 dst = src ÷ row_values（按行广播）。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}

/**
 * @brief 将向量广播到tile的每一行
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param dst[out] 存储结果的目标tile
 * @param row_values[in] 包含要广播到每一行的值的列向量
 * 
 * 将列向量复制到tile的每一行，实现行广播。
 * 例如，创建一个每行都相同的tile。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}

/**
 * @brief broadcast_row函数的返回值版本
 *
 * @tparam T tile类型
 * @tparam V 列向量类型
 * @param row_values[in] 包含要广播到每一行的值的列向量
 * @return T 包含广播结果的新tile
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline T broadcast_row(const V &row_values) {
    T dst;
    broadcast_row(dst, row_values);
    return dst;
}


/* ----------  列映射操作包装函数  ---------- */

/**
 * @brief 将列值加到tile的每一列
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用加法操作的源tile
 * @param col_values[in] 包含要加到每一列的行向量
 * 
 * 调用col_map操作，将行向量中的每个值加到tile的对应列的所有元素上。
 * 例如，col_values[0]会被加到第一列的所有元素上。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}

/**
 * @brief 从tile的每一列减去列值
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用减法操作的源tile
 * @param col_values[in] 包含要从每一列减去的值的行向量
 * 
 * 计算 dst = src - col_values（按列广播）。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}

/**
 * @brief 将tile的每一列乘以列值
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用乘法操作的源tile
 * @param col_values[in] 包含要乘以每一列的行向量
 * 
 * 计算 dst = src × col_values（按列广播）。
 * 常用于列缩放操作。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}

/**
 * @brief 将tile的每一列除以列值
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 应用除法操作的源tile
 * @param col_values[in] 包含要除以每一列的行向量
 * 
 * 计算 dst = src ÷ col_values（按列广播）。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}

/**
 * @brief 将向量广播到tile的每一列
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param dst[out] 存储结果的目标tile
 * @param col_values[in] 包含要广播到每一列的行向量
 * 
 * 将行向量复制到tile的每一列，实现列广播。
 * 例如，创建一个每列都相同的tile。
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

/**
 * @brief broadcast_col函数的返回值版本
 *
 * @tparam T tile类型
 * @tparam V 行向量类型
 * @param col_values[in] 包含要广播到每一列的行向量
 * @return T 包含广播结果的新tile
 */
template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline T broadcast_col(const V &col_values) {
    T dst;
    broadcast_col(dst, col_values);
    return dst;
}

/* ----------  三角掩码操作  ---------- */

/**
 * @brief 创建下三角掩码（包括对角线）
 *
 * @tparam RT tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 源tile
 * @param diagonal 对角线偏移（默认0）
 * @param val 填充值（默认0）
 * 
 * 保留矩阵对角线及其以下元素，其他位置填充指定值。
 * 常用于实现下三角矩阵操作，如Cholesky分解或注意力掩码。
 * 
 * @param diagonal 对角线偏移：
 *   - diagonal = 0：包含主对角线及以下
 *   - diagonal > 0：包含主对角线上方第diagonal条对角线及以下
 *   - diagonal < 0：包含主对角线下方第|diagonal|条对角线及以下
 */
template<ducks::rt::all RT>
__device__ static inline void tril(RT &dst, const RT &src, int diagonal=0, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    apply(dst, src, [val, diagonal]__device__(int row, int col, auto &src_val) {
        // 如果列索引 <= 行索引 + 对角线偏移，保留原值；否则填充指定值
        return col <= row + diagonal ? src_val : val;
    });
}

/**
 * @brief 创建上三角掩码（包括对角线）
 *
 * @tparam RT tile类型
 * @param dst[out] 存储结果的目标tile
 * @param src[in] 源tile
 * @param diagonal 对角线偏移（默认0）
 * @param val 填充值（默认0）
 * 
 * 保留矩阵对角线及其以上元素，其他位置填充指定值。
 * 常用于实现上三角矩阵操作。
 * 
 * @param diagonal 对角线偏移：
 *   - diagonal = 0：包含主对角线及以上
 *   - diagonal > 0：包含主对角线下方第diagonal条对角线及以上
 *   - diagonal < 0：包含主对角线上方第|diagonal|条对角线及以上
 */
template<ducks::rt::all RT>
__device__ static inline void triu(RT &dst, const RT &src, int diagonal=0, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    apply(dst, src, [val, diagonal]__device__(int row, int col, auto &src_val) {
        // 如果列索引 >= 行索引 + 对角线偏移，保留原值；否则填充指定值
        return col >= row + diagonal ? src_val : val;
    });
}