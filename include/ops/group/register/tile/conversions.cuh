/**
 * @file
 * @brief 寄存器瓦片的数据布局和类型转换。
 * 
 * 该文件包含用于在寄存器瓦片（register tiles）之间转换数据布局和类型的函数。
 * 主要功能包括矩阵转置、布局交换等低层次优化操作。
 */

/* ----------  布局交换函数  ---------- */

/**
 * @brief 使用内联汇编对8个bf16_2元素组成的块执行矩阵转置。
 * 
 * 这是一个底层操作，被更高层的布局交换函数用来在寄存器瓦片内转置bf16_2元素的布局。
 * 该函数利用内联PTX汇编高效地交换给定块的布局。
 * 
 * @param[out] dst 目标bf16_2元素的引用，转置结果将存储在这里。
 * @param[in] src 源bf16_2元素的引用，将被转置。
 */
__device__ static inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    KITTENS_CHECK_WARP// 检查当前是否在warp内执行（可能是某种调试或验证宏）
    asm volatile (
        "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"       // PTX汇编指令：同步转置8x8 bf16矩阵
    :   "+r"(*(uint32_t*)(&dst))        // 输出操作数：将dst作为32位整数寄存器操作数
    :   "r"(*(uint32_t*)(&src))         // 输入操作数：将src作为32位整数寄存器操作数
    );
}

/**
 * @brief 交换寄存器基础瓦片（rt_base）的布局。
 * 
 * 该函数通过对组成寄存器基础瓦片的bf16_2元素执行一系列布局交换来改变其数据布局。
 * 用于在寄存器瓦片内部转换数据排列方式。
 * 
 * @tparam T 寄存器瓦片元素的数据类型。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param[out] dst 目标寄存器基础瓦片的引用，结果将存储在这里。
 * @param[in] src 源寄存器基础瓦片的引用，将被交换布局。
 */
template<typename T, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    // 交换第一个数据块
    swap_layout_8(dst.data[0], src.data[0]);
    // 技术说明：如果可以简单地重新解释寄存器的布局，这个交换可以消除
    // 但这很可能导致错误，不值得这样做。
    typename rt_base<T, layout>::T2 data1_cache = src.data[1]; // 重要：缓存第二个元素，因为后面的操作会覆盖它

    // 交换剩余的三个数据块（注意索引的变化）    
    swap_layout_8(dst.data[1], src.data[2]);    // dst[1] <- src[2]
    swap_layout_8(dst.data[2], data1_cache);    // dst[2] <- 缓存的src[1]
    swap_layout_8(dst.data[3], src.data[3]);    // dst[3] <- src[3]
}

/**
 * @brief 交换寄存器瓦片（rt）的布局。
 * 
 * 该函数通过遍历寄存器瓦片的高度和宽度，对其每个基础元素执行布局交换。
 * 
 * @tparam T2 寄存器瓦片元素的数据类型。
 * @tparam _height 寄存器瓦片的高度。
 * @tparam _width 寄存器瓦片的宽度。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param[out] dst 目标寄存器瓦片的引用，结果将存储在这里。
 * @param[in] src 源寄存器瓦片的引用，将被交换布局。
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    // 循环遍历所有瓦片行
    #pragma unroll      // 编译器指令：完全展开循环以提升性能
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 对每个基础瓦片执行布局交换
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief 原地交换寄存器基础瓦片的布局。
 * 
 * 该函数通过将寄存器基础瓦片转换为转置布局类型，然后执行布局交换来实现原地转换。
 * 
 * @tparam T2 寄存器瓦片元素的数据类型。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param[in] src 要原地交换布局的寄存器基础瓦片的引用。
 * @return 交换后的寄存器基础瓦片的引用。
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    // 将源瓦片重新解释为目标布局类型（通过指针转换）
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    // 执行布局交换
    swap_layout(dst, src);
    return dst; // 返回转换后的引用
}


/**
 * @brief 原地交换寄存器瓦片的布局。
 * 
 * 该函数通过遍历寄存器瓦片的高度和宽度，对每个基础元素执行原地布局交换。
 * 
 * @tparam T2 寄存器瓦片元素的数据类型。
 * @tparam _rows 寄存器瓦片的高度（行数）。
 * @tparam _cols 寄存器瓦片的宽度（列数）。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param[in,out] tile 要原地交换布局的寄存器瓦片的引用。
 * @return 交换后的寄存器瓦片的引用。
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            // 对每个基础瓦片执行原地布局交换
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    // 将整个瓦片重新解释为转置布局类型并返回
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief 转置寄存器基础瓦片（8x8）。
 * 将8x8的寄存器瓦片进行转置操作。
 *
 * @tparam T 寄存器瓦片中元素的数据类型。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param dst[out] 存储转置后结果的寄存器瓦片引用。
 * @param src[in] 需要被转置的源寄存器瓦片引用。
 */
template<typename T, ducks::rt_layout::all layout>
__device__ static inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    // 交换第0个8x8瓦片的布局（实现转置）
    swap_layout_8(dst.data[0], src.data[0]);
    // 从技术上讲，如果我们在代码中其他地方简单地重新解释寄存器的布局，
    // 可以消除这个交换操作，但这很容易引发bug，不值得这样做。
    typename rt_base<T, layout>::T2 data1_cache = src.data[1]; // 缓存data[1]，对交换很重要！
    swap_layout_8(dst.data[1], src.data[2]);// 将src.data[2]转置到dst.data[1]
    swap_layout_8(dst.data[2], data1_cache);// 将缓存的src.data[1]转置到dst.data[2]
    swap_layout_8(dst.data[3], src.data[3]);// 转置最后一个8x8瓦片
}


/**
 * @brief 转置寄存器瓦片（分离版本）。
 * 此函数标记为"sep"，意味着dst的底层寄存器必须与src的底层寄存器分离（即不能是同一内存）。
 * 用于转置由多个8x8基础瓦片组成的更大瓦片。
 *
 * @tparam RT 寄存器瓦片类型，包含数据类型、行数、列数和布局信息。
 * @param dst[out] 存储转置后结果的寄存器瓦片引用。
 * @param src[in] 需要被转置的源寄存器瓦片引用。
 */
template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll // 展开循环以提高性能
    for(int i = 0; i < RT::height; i++) {// 遍历目标瓦片的行（源瓦片的列）
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {// 遍历目标瓦片的列（源瓦片的行）
            // 转置每个8x8基础瓦片，并交换行列位置            
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief 原地转置寄存器基础瓦片（8x8）。
 * 直接在原寄存器瓦片上执行转置操作，不分配额外存储。
 *
 * @tparam T2 寄存器瓦片中元素的数据类型。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param src[in] 需要被转置的寄存器瓦片引用。
 * @return 返回转置后的寄存器基础瓦片引用（与原对象相同）。
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    // 调用transpose函数，将结果存回原对象
    transpose(src, src);
    return src;
}

/**
 * @brief 原地转置方形寄存器瓦片。
 * 直接在原寄存器瓦片上执行转置操作，要求瓦片必须是方形的。
 *
 * @tparam T2 寄存器瓦片中元素的数据类型。
 * @tparam _rows 寄存器瓦片的高度（以16为单位）。
 * @tparam _cols 寄存器瓦片的宽度（以16为单位）。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param tile[in] 需要被转置的方形寄存器瓦片引用。
 * @return 返回转置后的寄存器瓦片引用（与原对象相同）。
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    // 静态断言：确保瓦片是方形的，否则不允许原地转置
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {// 遍历行
        #pragma unroll
        for(int j = 0; j < i; j++) {// 遍历上三角部分（不包括对角线）
            // 临时存储当前元素            
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);

            // 交换对称位置的元素并进行转置
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        // 对角线上的瓦片原地转置
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief 复制寄存器基础瓦片，并在必要时进行数据类型转换。
 * 将源寄存器基础瓦片的数据复制到目标寄存器基础瓦片，支持不同类型间的转换。
 *
 * @tparam T 目标寄存器元素的数据类型。
 * @tparam U 源寄存器元素的数据类型。
 * @tparam layout 寄存器基础瓦片的布局。
 * @param[out] dst 目标寄存器基础瓦片的引用。
 * @param[in] src 源寄存器基础瓦片的引用。
 */
template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;// 获取T类型的打包类型（如float4、half4等）
    using U2 = typename base_types::packing<U>::packed_type;// 获取U类型的打包类型
    // 循环展开优化    
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        // 使用类型转换器将源数据类型转换为目标数据类型
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}

// 仅在Hopper或Blackwell架构上编译以下代码
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 复制寄存器瓦片，并在必要时进行数据类型转换。
 * 处理更复杂的类型转换，特别是FP8与其他浮点类型（float、half、bf16）之间的转换。
 * 这些转换涉及线程间的数据重排，因为FP8使用不同的内存布局。
 *
 * @tparam T2 目标寄存器元素的数据类型。
 * @tparam U2 源寄存器元素的数据类型。
 * @tparam _height 寄存器瓦片的高度（以16为单位）。
 * @tparam _width 寄存器瓦片的宽度（以16为单位）。
 * @tparam layout 寄存器瓦片的布局。
 * @param[out] dst 目标寄存器瓦片的引用。
 * @param[in] src 源寄存器瓦片的引用。
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    // 情况1：从float/half/bf16转换为FP8（fp8e4m3或fp8e5m2）
    // 源数据类型是float/half/bf16，目标数据类型是FP8
    if constexpr (
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e5m2>)
    ) {
        // FLOAT (SRC -- 1H x 2W) to FP8 (DST -- 1H x 1W)
        // 从FLOAT（源 - 1高度x2宽度）转换为FP8（目标 - 1高度x1宽度）
        // FP8使用更紧凑的存储，所以宽度减半
        int laneid = threadIdx.x % 32;  // 获取线程在warp内的ID（0-31）

        #pragma unroll
        for(int i = 0; i < dst.height; i++) {// 遍历瓦片高度
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {// 遍历瓦片宽度（目标宽度是源宽度的一半）
                #pragma unroll
                for(int k = 0; k < dst.tiles[0][0].packed_per_thread; k++) {
                    
                    // 根据源数据类型选择对应的打包类型
                    using src_t = std::conditional_t<std::is_same_v<U2, float>, float2, std::conditional_t<std::is_same_v<U2, kittens::bf16>, bf16_2, half2>>;
                    src_t val1, val2;// 用于存储从源读取的两个值

                    // 数据重排阶段1：从源瓦片读取数据
                    // 根据线程ID的奇偶性决定读取模式
                    if (laneid % 2 == 0) { 
                        // 偶数线程：先读取左侧核心矩阵的值作为位置0和2
                        // 注意：源宽度是目标宽度的两倍，所以使用2*j + k/2索引
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+0];// 位置0
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+2];// 位置2
                    } else { 
                        // 奇数线程：先读取右侧核心矩阵的值作为位置1和3
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+2];// 位置2（作为val1）
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+0];// 位置0（作为val2）
                    }

                    // 数据重排阶段2：线程间洗牌交换数据
                    // 计算行掩码和行偏移，用于确定洗牌源
                    int row_mask = 4 * ( laneid / 4 );// 每4个线程一组
                    int row_offset = row_mask + ( (laneid-row_mask) / 2 ) + ( laneid % 2 );
                    // 从偶数线程获取数据
                    int src_offset = (laneid % 2 == 0 ) ? row_offset + 0 : ( row_offset + 1 );
                    src_t val01 = packed_shfl_sync(MASK_ALL, val1, src_offset);  // Get from even thread

                    int src_offset2 = (laneid % 4 < 2 ) ? src_offset + 1 : (src_offset - 1);
                    // 从奇数线程获取数据
                    src_t val23 = packed_shfl_sync(MASK_ALL, val2, src_offset2);  // Get from odd thread
                    
                    // 数据转换阶段：将获取的数据转换为FP8
                    float4 f4;  // 用于存储4个浮点值
                    // 根据目标FP8类型选择对应的打包类型
                    using fp8_4_t = std::conditional_t<std::is_same_v<T2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t f4_fp8;// 存储转换后的FP8值


                    // 根据线程ID的低2位决定如何组合数据
                    if ( laneid % 4 < 2 ) { 
                        // 线程组0和1：组合val01和val23的前两个分量
                        f4.x = val01.x;  // 来自偶数线程的第一个值
                        f4.y = val01.y;  // 来自偶数线程的第二个值
                        f4.z = val23.x;  // 来自奇数线程的第一个值
                        f4.w = val23.y;  // 来自奇数线程的第二个值
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);  // 转换为FP8
                        dst.tiles[i][j].data[k] = f4_fp8;  // 存储到目标
                    } else {
                        // 线程组2和3：交换组合顺序
                        f4.x = val23.x;  // 来自奇数线程的第一个值
                        f4.y = val23.y;  // 来自奇数线程的第二个值
                        f4.z = val01.x;  // 来自偶数线程的第一个值
                        f4.w = val01.y;  // 来自偶数线程的第二个值
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);  // 转换为FP8
                        dst.tiles[i][j].data[k] = f4_fp8;  // 存储到目标
                    }
                }
            }
        }
    }
    // 情况2：从FP8（fp8e4m3或fp8e5m2）转换为float/half/bf16
    // 源数据类型是FP8，目标数据类型是float/half/bf16
    else if constexpr (
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, half>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, half>)
    ) {
        // FP8 (SRC -- 1H x 1W) to FLOAT (DST -- 1H x 2W)
        // 从FP8（源 - 1高度x1宽度）转换为FLOAT（目标 - 1高度x2宽度）
        int laneid = threadIdx.x % 32;  // 获取线程在warp内的ID（0-31）

        #pragma unroll
        for(int i = 0; i < src.height; i++) {// 遍历源瓦片高度
            #pragma unroll
            for(int j = 0; j < src.width; j++) {// 遍历源瓦片宽度
                #pragma unroll
                for(int k = 0; k < src.tiles[0][0].packed_per_thread; k++) {
                    int dst_j = 2*j + k/2;// 计算目标瓦片宽度索引（宽度加倍）

                    // 读取FP8数据并转换为float4
                    using fp8_4_t = std::conditional_t<std::is_same_v<U2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t val = src.tiles[i][j].data[k]; // 读取FP8数据
                    float4 f4 = base_types::convertor<float4, fp8_4_t>::convert(val);// 转换为float4

                    // 将float4拆分为两个float2
                    float2 f2_0, f2_1;
                    if ( laneid % 4 < 2 ) { 
                        // 线程组0和1：使用.x和.y作为第一个float2，.z和.w作为第二个
                        f2_0 = make_float2(f4.x, f4.y);
                        f2_1 = make_float2(f4.z, f4.w);
                    }
                    else {                         // 线程组2和3：交换顺序
                        f2_0 = make_float2(f4.z, f4.w);
                        f2_1 = make_float2(f4.x, f4.y);
                    }
                    // 线程间洗牌交换数据
                    int row_offset = 4 * (laneid/4) + (laneid%2) * 2 + (laneid%4) / 2;
                    float2 f2_0_shfl = packed_shfl_sync(MASK_ALL, f2_0, row_offset);
                    float2 f2_1_shfl = packed_shfl_sync(MASK_ALL, f2_1, row_offset^2);// 异或2以获取配对线程的数据

                    // 如果需要，转换为目标数据类型（half或bf16）
                    using dst_t = std::conditional_t<std::is_same_v<T2, float>, float2, std::conditional_t<std::is_same_v<T2, kittens::bf16>, bf16_2, half2>>;
                    if constexpr (!(std::is_same_v<T2, float>)) {
                        // 目标类型不是float，需要转换
                        dst_t f2_0_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_0_shfl);
                        dst_t f2_1_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_1_shfl);
                        if (laneid % 2 == 0) {  
                            // 偶数线程：存储f2_0_shfl_t到位置0，f2_1_shfl_t到位置2
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl_t;
                        } else {
                            // 奇数线程：交换存储顺序
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl_t;
                        }
                    } else {
                        // 目标类型是float，直接存储
                        if (laneid % 2 == 0) {  
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl;
                        }
                    }
                }
            }
        }
    }
    // 默认情况：类型布局一对一映射，直接调用基础瓦片复制函数
    else {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {// 遍历瓦片高度
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {// 遍历瓦片宽度
                // 对每个基础瓦片调用copy函数                
                copy(dst.tiles[i][j], src.tiles[i][j]);
            }
        }
    }
}
#else// 如果不是Hopper或Blackwell架构，这里可能有其他实现，但当前代码片段未提供

/**
 * @brief 复制寄存器瓦片，必要时转换底层数据类型。
 * 
 * 该函数将源寄存器瓦片复制到目标寄存器瓦片，如果源和目标的元素数据类型不同，会进行类型转换。
 * 保持相同的布局不变。
 * 
 * @tparam T2 目标寄存器元素的数据类型。
 * @tparam U2 源寄存器元素的数据类型。
 * @tparam _height 寄存器瓦片的高度（以16为单位）。
 * @tparam _width 寄存器瓦片的宽度（以16为单位）。
 * @tparam layout 寄存器瓦片的当前布局。
 * @param[out] dst 目标寄存器瓦片的引用。
 * @param[in] src 源寄存器瓦片的引用。
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 调用基础瓦片拷贝函数（可能包含类型转换）
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}
#endif  // 结束条件编译或头文件保护

/* ----------  因果掩码函数  ---------- */

/**
 * @brief 通过将主对角线上方的元素置零，使方形寄存器瓦片具有因果性（下三角矩阵）。
 * 
 * 该函数原地修改方形寄存器瓦片，使其具有因果性。主对角线上方的所有元素被设为零，
 * 而主对角线及其下方的元素保持不变。这对于注意力机制中的因果掩码非常有用。
 * 
 * @tparam RT 寄存器瓦片类型，必须是行布局（row_layout）。
 * @param[out] dst 目标寄存器瓦片的引用，将存储因果掩码结果。
 * @param[in] src 源寄存器瓦片的引用，提供原始数据。
 * @param[in] val 用于填充上方对角线的值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行
    // 将解包的值打包为寄存器瓦片的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    // 在Hopper或Blackwell架构上，检查不支持的数据类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif

    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // 对角线下方：复制源数据
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // 对角线上方：用指定值填充
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // 对角线上的特殊处理：部分元素保留，部分置零
                // 用于对角线处理的魔法掩码（针对8x8瓦片内的32个线程）
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                // 对角线上的4个数据块：索引1在对角线下方（复制），索引2在对角线上方（置零）
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // 下三角部分，复制
                dst.tiles[i][j].data[2] = packed_val;              // 上三角部分，置零
                
                // 根据掩码决定data[0].x和data[3].x是保留还是置零
                if((MASK_X >> laneid()) & 1) {  // laneid()获取当前线程在warp中的ID（0-31）
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                // 根据掩码决定data[0].y和data[3].y是保留还是置零
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
            __syncwarp();// 同步warp内的所有线程
        }
    }
}

/**
 * @brief 通过将主对角线下方的元素置零，使方形寄存器瓦片具有反因果性（上三角矩阵）。
 * 
 * 该函数是make_causal的转置版本，用于创建上三角掩码。主对角线下方的所有元素被设为零，
 * 而主对角线及其上方的元素保持不变。这在某些注意力机制变体中可能有用。
 * 
 * @tparam RT 寄存器瓦片类型，必须是行布局（row_layout）。
 * @param[out] dst 目标寄存器瓦片的引用，将存储反因果掩码结果。
 * @param[in] src 源寄存器瓦片的引用，提供原始数据。
 * @param[in] val 用于填充下方对角线的值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP// 检查是否在warp内执行
    // 将解包的值打包为寄存器瓦片的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    // 在Hopper或Blackwell架构上，检查不支持的数据类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<typename RT::dtype, fp8e4m3_4> && !std::is_same_v<typename RT::dtype, fp8e5m2_4>, "Unsupported type for make_causal");
    #endif
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列        
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // 对角线上方：复制源数据
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // 对角线下方：用指定值填充
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // 对角线上的特殊处理：部分元素保留，部分置零
                // 用于对角线处理的魔法掩码（与make_causal不同，针对转置布局）
                constexpr uint32_t MASK_X = 0x88CCEEF; 
                constexpr uint32_t MASK_Y = 0x88CCEEFF;

                // 对角线上的4个数据块：索引1在对角线下方（置零），索引2在对角线上方（复制）
                dst.tiles[i][j].data[1] = packed_val;              // 下三角部分，置零
                dst.tiles[i][j].data[2] = src.tiles[i][j].data[2]; // 上三角部分，复制

                // 根据掩码决定data[0].x和data[3].x是保留还是置零
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }

                // 根据掩码决定data[0].y和data[3].y是保留还是置零
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                // below the diagonal
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
                
            }
            __syncwarp(); // 同步warp内的所有线程
        }
    }
}

/* ----------  三角填充函数  ---------- */

/**
 * @brief 通过将指定对角线以上的元素置零，使寄存器瓦片变为下三角矩阵。
 * 
 * 对于行布局（row_layout）的寄存器瓦片，该函数根据全局行索引和列索引的关系，
 * 将对角线（考虑偏移diagonal）以上的元素置为指定值，其余元素从源瓦片复制。
 * 这用于创建带偏移的下三角掩码（下三角矩阵，对角线偏移由diagonal参数控制）。
 * 
 * @tparam RT 寄存器瓦片类型，必须是行布局（row_layout）。
 * @param[in,out] dst 目标寄存器瓦片的引用，将存储下三角掩码结果。
 * @param[in] src 源寄存器瓦片的引用，提供原始数据。
 * @param[in] diagonal 对角线偏移值。实际对角线为 row_idx = col_idx - diagonal。
 *                    当diagonal=0时为标准下三角矩阵（对角线以上置零）。
 * @param[in] val 用于填充对角线以上元素的默认值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行
    // 将解包的值打包为寄存器瓦片的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环遍历每个基础瓦片中的打包数据块（通常为4个，对应4个8x8子块）
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 计算当前元素在全局矩阵中的行索引
                // 注意：行布局中，k%2决定在8x8子块中的行组，laneid()/4决定在组内的具体行
                const int global_row_idx   = (i * dst.tile_size_row) + ((k % 2) * 8) + (laneid() / 4);

                // 计算当前元素在全局矩阵中的列索引（x和y分量分别计算，因为bf16_2包含两个元素）
                // 注意：k/2决定在8x8子块中的列组，laneid()%4决定在组内的列对，乘以2是因为每个元素包含两个分量
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                
                // x分量：如果列索引 <= 行索引 + diagonal（在对角线或下方），则复制源数据；否则填充指定值
                if (global_col_idx_x <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                // y分量：处理逻辑同上
                if (global_col_idx_y <= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();// 同步warp内的所有线程
    }
}

/**
 * @brief 列布局版本的下三角填充函数。
 * 
 * 这是针对列布局（col_layout）寄存器瓦片的tril函数重载版本。
 * 由于列布局与行布局的内存排列不同，计算全局索引的方式也有所不同。
 * 
 * @tparam RT 寄存器瓦片类型，必须是列布局（col_layout）。
 * @param[in,out] dst 目标寄存器瓦片的引用。
 * @param[in] src 源寄存器瓦片的引用。
 * @param[in] diagonal 对角线偏移值。
 * @param[in] val 用于填充对角线以上元素的默认值，默认为0。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环遍历每个基础瓦片中的打包数据块
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 列布局中，计算全局索引的方式与行布局不同
                // 行索引计算：k/2决定在8x8子块中的行组，laneid()%4决定在组内的行对
                const int global_row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                // 列索引计算：k%2决定在8x8子块中的列组，laneid()/4决定在组内的具体列
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4);
                
                // x分量：如果列索引 <= 行索引 + diagonal（在对角线或下方），则复制源数据；否则填充指定值
                if (global_col_idx <= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                // y分量：处理逻辑同上
                if (global_col_idx <= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();   // 同步warp内的所有线程
    }
}

/**
 * @brief 通过将指定对角线以下的元素置零，使寄存器瓦片变为上三角矩阵。
 * 
 * 对于行布局（row_layout）的寄存器瓦片，该函数根据全局行索引和列索引的关系，
 * 将对角线（考虑偏移diagonal）以下的元素置为指定值，其余元素从源瓦片复制。
 * 这用于创建带偏移的上三角掩码（上三角矩阵，对角线偏移由diagonal参数控制）。
 * 
 * @tparam RT 寄存器瓦片类型，必须是行布局（row_layout）。
 * @param[in,out] dst 目标寄存器瓦片的引用，将存储上三角掩码结果。
 * @param[in] src 源寄存器瓦片的引用，提供原始数据。
 * @param[in] diagonal 对角线偏移值。实际对角线为 row_idx = col_idx - diagonal。
 *                    当diagonal=0时为标准上三角矩阵（对角线以下置零）。
 * @param[in] val 用于填充对角线以下元素的默认值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环遍历每个基础瓦片中的打包数据块            
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 计算当前元素在全局矩阵中的行索引和列索引（与tril函数相同）
                const int global_row_idx   = (i * dst.tile_size_row) + ((k % 2) * 8) + (laneid() / 4);
                const int global_col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                
                // x分量：如果列索引 >= 行索引 + diagonal（在对角线或上方），则复制源数据；否则填充指定值
                if (global_col_idx_x >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }

                // y分量：处理逻辑同上                
                if (global_col_idx_y >= global_row_idx + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();// 同步warp内的所有线程
    }
}

/**
 * @brief 列布局版本的上三角填充函数。
 * 
 * 这是针对列布局（col_layout）寄存器瓦片的triu函数重载版本。
 * 
 * @tparam RT 寄存器瓦片类型，必须是列布局（col_layout）。
 * @param[in,out] dst 目标寄存器瓦片的引用。
 * @param[in] src 源寄存器瓦片的引用。
 * @param[in] diagonal 对角线偏移值。
 * @param[in] val 用于填充对角线以下元素的默认值，默认为0。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int diagonal, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    // 循环遍历所有瓦片行
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 循环遍历所有瓦片列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 循环遍历每个基础瓦片中的打包数据块
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 列布局中的全局索引计算（与tril的列布局版本相同）
                const int global_row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int global_row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4);
                // x分量：如果列索引 >= 行索引 + diagonal（在对角线或上方），则复制源数据；否则填充指定值
                if (global_col_idx >= global_row_idx_x + diagonal) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                                               { dst.tiles[i][j].data[k].x = val; }
                // y分量：处理逻辑同上
                if (global_col_idx >= global_row_idx_y + diagonal) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                                               { dst.tiles[i][j].data[k].y = val; }
            }
        }
        __syncwarp();// 同步warp内的所有线程
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief 向右填充寄存器瓦片。
 * 将寄存器瓦片中从指定列索引开始到最右侧的区域填充为给定值。
 *
 * @tparam RT 寄存器瓦片类型（必须是行布局）。
 * @param dst[in,out] 要填充的目标寄存器瓦片。
 * @param src[in] 源寄存器瓦片，用于复制未填充区域的数据。
 * @param col_idx[in] 列索引，从该列开始向右填充（包含该列）。
 * @param val[in] 填充值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP// 检查是否在warp内执行，确保线程同步
    if(col_idx >= dst.cols) return;// 如果列索引超出瓦片列数，直接返回
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历瓦片的高度维度（以基础瓦片为单位）
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历瓦片的宽度维度（以基础瓦片为单位）
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历每个基础瓦片中的打包数据元素
                // 计算当前线程处理的元素在全局瓦片中的列索引（x和y分量）
                const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                // 对x分量：如果列索引大于等于填充起始列，则填充给定值，否则复制源数据
                if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                // 对y分量：同理
                if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief 向右填充寄存器瓦片（列布局版本）。
 * 列布局版本使用不同的内存访问模式。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    // 将解包的值打包为列布局所需的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的列索引
                const int t_col_idx = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4); 

                // 如果列索引大于等于填充起始列，则填充打包值，否则复制源数据
                if (t_col_idx >= col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        __syncwarp();// 同步warp内的所有线程，确保内存访问一致性
    }
}

/**
 * @brief 向左填充寄存器瓦片。
 * 将寄存器瓦片中从最左侧到指定列索引（不包含）的区域填充为给定值。
 *
 * @tparam RT 寄存器瓦片类型（必须是行布局）。
 * @param dst[in,out] 要填充的目标寄存器瓦片。
 * @param src[in] 源寄存器瓦片，用于复制未填充区域的数据。
 * @param col_idx[in] 列索引，填充该列左侧的区域（不包含该列）。
 * @param val[in] 填充值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(col_idx <= 0) return;// 如果列索引小于等于0，直接返回
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的列索引
                const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                // 对x分量：如果列索引小于填充边界列，则填充给定值，否则复制源数据
                if (col_idx_x < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                // 对y分量：同理
                if (col_idx_y < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief 向左填充寄存器瓦片（列布局版本）。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    // 将解包的值打包为列布局所需的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的列索引
                const int thread_col = (j * dst.tile_size_col) + ((k % 2) * 8) + ((laneid() / 4));

                // 如果列索引小于填充边界列，则填充打包值，否则复制源数据
                if (thread_col < col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        __syncwarp();// 同步warp内的所有线程
    }
}

/**
 * @brief 向上填充寄存器瓦片。
 * 将寄存器瓦片中从顶部到指定行索引（不包含）的区域填充为给定值。
 *
 * @tparam RT 寄存器瓦片类型（必须是行布局）。
 * @param dst[in,out] 要填充的目标寄存器瓦片。
 * @param src[in] 源寄存器瓦片，用于复制未填充区域的数据。
 * @param row_idx[in] 行索引，填充该行上方的区域（不包含该行）。
 * @param val[in] 填充值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(row_idx <= 0) return;    // 如果行索引小于等于0，直接返回

    // 将解包的值打包为行布局所需的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的行索引
                const int thread_row = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                // 如果行索引小于填充边界行，则填充打包值，否则复制源数据
                if (thread_row < row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}

/**
 * @brief 向上填充寄存器瓦片（列布局版本）。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的行索引（x和y分量）
                const int row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;

                // 对x分量：如果行索引小于填充边界行，则填充给定值，否则复制源数据
                if (row_idx_x < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                // 对y分量：同理
                if (row_idx_y < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief 向下填充寄存器瓦片。
 * 将寄存器瓦片中从指定行索引开始到底部的区域填充为给定值。
 *
 * @tparam RT 寄存器瓦片类型（必须是行布局）。
 * @param dst[in,out] 要填充的目标寄存器瓦片。
 * @param src[in] 源寄存器瓦片，用于复制未填充区域的数据。
 * @param row_idx[in] 行索引，从该行开始向下填充（包含该行）。
 * @param val[in] 填充值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    if(row_idx >= dst.rows) return;// 如果行索引超出瓦片行数，直接返回
    
    // 将解包的值打包为行布局所需的数据类型
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的行索引
                const int thread_row = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));

                // 如果行索引大于等于填充起始行，则填充打包值，否则复制源数据
                if (thread_row >= row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                        { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}

/**
 * @brief 向下填充寄存器瓦片（列布局版本）。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {// 遍历高度维度
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {// 遍历宽度维度
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {// 遍历打包数据
                // 计算当前线程处理的元素在全局瓦片中的行索引（x和y分量）
                const int row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                const int row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                // 对x分量：如果行索引大于等于填充起始行，则填充给定值，否则复制源数据
                if (row_idx_x >= row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                // 对y分量：同理
                if (row_idx_y >= row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief 用给定值填充寄存器瓦片（register tile）的右上角区域。
 *        对于行主序（row_layout）的瓦片，从指定行索引向上（到顶部）和从指定列索引向右（到右侧）填充。
 *
 * @tparam RT 寄存器瓦片的类型，必须是行主序布局（ducks::rt::row_layout）。
 * @param dst[in,out] 目标寄存器瓦片，将被填充的瓦片。
 * @param src[in] 源寄存器瓦片，从中复制未填充区域的值。
 * @param row_idx[in] 行索引，填充从此索引上方（不包含该行）到瓦片顶部的行。如果为负值，表示没有行需要填充。
 * @param col_idx[in] 列索引，填充从此索引（包含）向右到瓦片右侧的列。如果大于等于瓦片列数，表示没有列需要填充。
 * @param val[in] 用于填充的值，默认为0。
 */
template<ducks::rt::row_layout RT>
__device__ static inline void upper_right_fill(RT &dst, const RT &src, const int row_idx, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行（调试/验证宏）
    
    // 边界检查：如果列索引超出瓦片列数，或行索引小于0，则无需填充，直接返回
    if(col_idx >= dst.cols || row_idx < 0) return;
    
    // 遍历瓦片中的所有子瓦片（tiles），按行优先顺序
    #pragma unroll  // 提示编译器展开循环以优化性能
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 遍历每个子瓦片中的打包数据（packed data）
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 计算当前打包元素对应的全局行索引（行主序布局）：
                // i * dst.tile_size_row: 子瓦片所在的行偏移
                // (k % 2) * 8: 在打包数据中，每8个元素为一组（x和y交替打包）
                // laneid() / 4: 根据线程ID（laneid）计算行索引
                const int row_idx_xy = (i * dst.tile_size_row) + ((k % 2) * 8) + ((laneid() / 4));
                // 计算当前打包元素的x分量的全局列索引
                const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                // 计算当前打包元素的y分量的全局列索引（比x分量多1）
                const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                // 填充x分量：如果当前元素位于填充区域（列索引≥col_idx且行索引<row_idx）
                if (col_idx_x >= col_idx && row_idx_xy < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                // 填充y分量：逻辑同上
                if (col_idx_y >= col_idx && row_idx_xy < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
        __syncwarp();// 同步warp中的所有线程，确保内存操作的一致性
    }
}


/**
 * @brief 用给定值填充寄存器瓦片（register tile）的右上角区域。
 *        对于列主序（col_layout）的瓦片，从指定行索引向上（到顶部）和从指定列索引向右（到右侧）填充。
 *
 * @tparam RT 寄存器瓦片的类型，必须是列主序布局（ducks::rt::col_layout）。
 * @param dst[in,out] 目标寄存器瓦片，将被填充的瓦片。
 * @param src[in] 源寄存器瓦片，从中复制未填充区域的值。
 * @param row_idx[in] 行索引，填充从此索引上方（不包含该行）到瓦片顶部的行。
 * @param col_idx[in] 列索引，填充从此索引（包含）向右到瓦片右侧的列。
 * @param val[in] 用于填充的值，默认为0。
 */
template<ducks::rt::col_layout RT>
__device__ static inline void upper_right_fill(RT &dst, const RT &src, const int row_idx, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    KITTENS_CHECK_WARP

    // 将未打包的值打包成RT::dtype类型（列主序布局可能需要不同的打包方式）
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    // 边界检查
    if(col_idx >= dst.cols || row_idx < 0) return;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                // 计算当前打包元素的x分量的全局行索引（列主序布局）：
                // i * dst.tile_size_row: 子瓦片所在的行偏移
                // (k / 2) * 8: 打包数据中的行偏移
                // (laneid() % 4) * 2: 根据线程ID计算行偏移
                const int row_idx_x = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2);
                // y分量的全局行索引（比x分量多1）
                const int row_idx_y = (i * dst.tile_size_row) + ((k / 2) * 8) + ((laneid() % 4) * 2) + 1;
                // 计算当前打包元素对应的全局列索引（列主序布局）：
                const int col_idx_xy = (j * dst.tile_size_col) + ((k % 2) * 8) + (laneid() / 4); 

                // 填充x分量：如果当前元素位于填充区域（行索引<row_idx且列索引<col_idx）
                // 注意：这里与行主序版本的条件略有不同，填充的是左下角？需要确认逻辑
                // 注释可能有问题，实际代码逻辑需要确认                
                if (row_idx_x < row_idx && col_idx_xy < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                // 填充y分量
                if (row_idx_y < row_idx && col_idx_xy < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
        __syncwarp();
    }
}
/* ----------  SUBTILE  ---------- */

/**
 * @brief 返回给定瓦片中一个子瓦片（subtile）的引用。
 *        用于从大瓦片中提取一个较小的子瓦片进行原地操作。
 *
 * @tparam subtile_rows 子瓦片的高度（行数）。
 * @tparam RT 输入瓦片的类型，必须满足ducks::rt::all概念（可以是行主序或列主序）。
 * @param src 输入瓦片，从中提取子瓦片。
 * @param idx 子瓦片的坐标索引。对于行主序布局，表示第几个子瓦片行；对于列主序布局，表示第几个子瓦片列。
 * @return 对子瓦片的引用，可以直接操作。
 *
 * @note 子瓦片高度必须能整除瓦片高度（考虑瓦片行维度的平铺因子TILE_ROW_DIM）。
 */
template<int subtile_rows, ducks::rt::all RT>
__device__ static inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    KITTENS_CHECK_WARP
    using T = typename RT::T;  // 获取瓦片的数据类型
    
    // 静态断言：确保子瓦片高度能整除瓦片高度
    // 注意：这里除以TILE_ROW_DIM<T>是因为瓦片高度可能以平铺行维度为单位
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");

    // 使用reinterpret_cast将源瓦片的一部分重新解释为子瓦片类型
    // 注意：这里假设内存布局是兼容的，且通过索引计算子瓦片的起始位置
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)] // 计算子瓦片在源瓦片中的起始位置
    );
}
