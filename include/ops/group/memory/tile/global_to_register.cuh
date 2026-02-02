/**
 * @file
 * @brief 全局内存与寄存器之间的协作数据传输函数
 * 
 * 该文件提供工作组（协作warp）在全局内存和寄存器之间直接传输数据的函数，
 * 支持行主序(row-major)和列主序(col-major)两种内存布局。
 */

/**
 * @brief 从全局内存协作加载数据到行主序布局的寄存器tile
 *
 * @tparam axis 加载操作的轴方向（控制数据访问模式）
 * @tparam RT 行主序寄存器tile类型（必须满足ducks::rt::row_layout概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标寄存器tile，数据将加载到这里
 * @param[in] src 源全局内存数组，从中加载数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将全局内存中的数据协作加载到寄存器tile中，支持行主序布局。
 * 所有warp中的线程协作完成数据传输，确保内存访问的合并。
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    // 类型定义：目标数据类型和源数据类型
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    
    // Hopper和Blackwell架构不支持fp8e4m3_4和fp8e5m2_4类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif
    // 计算源数据指针和行步长
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();

    // 使用打包数据类型（用于向量化加载）
    using U2 = base_types::packing<U>::packed_type;

    // 计算warp内的线程ID和warp ID
    int warp_laneid = threadIdx.x % WARP_THREADS;// warp内的线程ID（0-31）
    int local_warpid;// 本地warp ID（根据工作组大小重新映射）

    // 根据GROUP_WARPS是否能被4整除，计算本地warp ID的映射
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();

    // 计算行偏移（每个warp处理不同的行块）
    const int row_offset = dst.rows*local_warpid;

    // 主循环：遍历tile中的每个子块
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 计算当前子块的行索引
        int row = row_offset + i*dst.tile_size_row + (warp_laneid / 4);

        // 第一组数据加载：处理每个子块的前半部分列
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 计算当前子块的列索引
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
            // 加载前8列的数据（打包加载）
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+0)]));
            // 加载后8列的数据（打包加载）
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+8)]));
        }

        // 第二组数据加载：处理每个子块的后半部分列（偏移8行）
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + 2*(warp_laneid % 4);

            // 加载前8列的数据（打包加载）
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+0)]));

            // 加载后8列的数据（打包加载）
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+8)]));
        }
    }
}

/**
 * @brief 从全局内存协作加载数据到列主序布局的寄存器tile
 *
 * @tparam axis 加载操作的轴方向（控制数据访问模式）
 * @tparam RT 列主序寄存器tile类型（必须满足ducks::rt::col_layout概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标寄存器tile，数据将加载到这里
 * @param[in] src 源全局内存数组，从中加载数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将全局内存中的数据协作加载到寄存器tile中，支持列主序布局。
 * 列主序布局适用于某些矩阵运算，如矩阵乘法中的B矩阵。
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    // 类型定义：目标数据类型和源数据类型
    using T = typename RT::T;
    using U = typename GL::dtype;
    
    // Hopper和Blackwell架构不支持fp8e4m3和fp8e5m2类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<T, fp8e5m2>, "Unsupported type for load/store");
    #endif
    
    // 计算源数据指针和行步长
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();

    // 计算warp内的线程ID和warp ID
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;

    // 根据GROUP_WARPS是否能被4整除，计算本地warp ID的映射
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();

    // 计算行偏移（每个warp处理不同的行块）
    const int row_offset = dst.rows*local_warpid;

    // 主循环：遍历tile中的每个子块（列主序布局需要更复杂的访问模式）
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 计算当前子块的行索引（列主序：线程处理相邻的2行）
        int row = row_offset + i*dst.tile_size_row + 2*(warp_laneid % 4);
        // 第一组数据加载：处理每个子块的第1行
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 计算当前子块的列索引
            int col = j*dst.tile_size_col + (warp_laneid / 4);
            // 加载第1行的前8列数据
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+0)]);
            // 加载第1行的后8列数据
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+8)]);
        }
        // 第二组数据加载：处理每个子块的第2行
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (warp_laneid / 4);
            // 加载第2行的前8列数据
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+0)]);
            // 加载第2行的后8列数据
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+8)]);
        }
        // 第三组数据加载：处理每个子块的第9行（偏移8行）
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (warp_laneid / 4);
            // 加载第9行的前8列数据
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+0)]);
            // 加载第9行的后8列数据
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+8)]);
        }

        // 第四组数据加载：处理每个子块的第10行（偏移8+1行）
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + (warp_laneid / 4);
            // 加载第10行的前8列数据
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+0)]);
            // 加载第10行的后8列数据
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+8)]);
        }
    }
}

/**
 * @brief 从全局内存协作加载数据到寄存器tile（默认轴版本）
 *
 * @tparam RT 寄存器tile类型（必须满足ducks::rt::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标寄存器tile，数据将加载到这里
 * @param[in] src 源全局内存数组，从中加载数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是load函数的简化版本，使用默认轴方向(axis=2)。
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    // 调用默认轴方向(axis=2)的load函数
    load<2>(dst, src, idx);
}


/**
 * @brief 从行主序布局的寄存器tile协作存储数据到全局内存
 *
 * @tparam axis 存储操作的轴方向（控制数据访问模式）
 * @tparam RT 行主序寄存器tile类型（必须满足ducks::rt::row_layout概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标全局内存数组，数据将存储到这里
 * @param[in] src 源寄存器tile，从中存储数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将寄存器tile中的数据协作存储到全局内存中，支持行主序布局。
 * 存储操作是加载操作的逆过程。
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    // 类型定义：源数据类型和目标数据类型
    using T2 = RT::dtype;
    using U = typename GL::dtype;

    // Hopper和Blackwell架构不支持fp8e4m3_4和fp8e5m2_4类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for load/store");
    #endif
    // 计算目标数据指针和行步长
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();

    // 使用打包数据类型（用于向量化存储）
    using U2 = base_types::packing<U>::packed_type;

    // 计算warp内的线程ID和warp ID
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;

    // 根据GROUP_WARPS是否能被4整除，计算本地warp ID的映射
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();

    // 计算行偏移（每个warp处理不同的行块）
    const int row_offset = src.rows*local_warpid;

    // 主循环：遍历tile中的每个子块，将数据存储到全局内存
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        // 计算当前子块的行索引
        int row = row_offset + i*src.tile_size_row + (warp_laneid / 4);
        // 第一组数据存储：处理每个子块的前半部分列
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            // 计算当前子块的列索引
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            // 存储前8列的数据（打包存储）
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            // 存储后8列的数据（打包存储）
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
        }
        // 第二组数据存储：处理每个子块的后半部分列（偏移8行）
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + 2*(warp_laneid % 4);
            // 存储前8列的数据（打包存储）
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            // 存储后8列的数据（打包存储）
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}

/**
 * @brief 从列主序布局的寄存器tile协作存储数据到全局内存
 *
 * @tparam axis 存储操作的轴方向（控制数据访问模式）
 * @tparam RT 列主序寄存器tile类型（必须满足ducks::rt::col_layout概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标全局内存数组，数据将存储到这里
 * @param[in] src 源寄存器tile，从中存储数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这个函数将寄存器tile中的数据协作存储到全局内存中，支持列主序布局。
 * 存储操作是加载操作的逆过程。
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    // 类型定义：源数据类型（解包后）和目标数据类型    
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    // Hopper和Blackwell架构不支持fp8e4m3_4和fp8e5m2_4类型
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T, fp8e4m3_4> && !std::is_same_v<T, fp8e5m2_4>, "Unsupported type for load/store");
    #endif
    // 计算目标数据指针和行步长
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    // 计算warp内的线程ID和warp ID    
    int warp_laneid = threadIdx.x % WARP_THREADS;
    int local_warpid;
    // 根据GROUP_WARPS是否能被4整除，计算本地warp ID的映射
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    // 计算行偏移（每个warp处理不同的行块）
    const int row_offset = src.rows*local_warpid;
    // 主循环：遍历tile中的每个子块，将数据存储到全局内存
    #pragma unroll
    for(int i = 0; i < src.height; i++) {

        // 计算当前子块的行索引（列主序：线程处理相邻的2行）
        int row = row_offset + i*src.tile_size_row + 2*(warp_laneid % 4);
        // 第一组数据存储：处理每个子块的第1行
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            // 计算当前子块的列索引
            int col = j*src.tile_size_col + (warp_laneid / 4);
            // 存储第1行的前8列数据
            dst_ptr[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            // 存储第1行的后8列数据
            dst_ptr[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        // 第二组数据存储：处理每个子块的第2行
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + (warp_laneid / 4);
            dst_ptr[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}

/**
 * @brief 从寄存器tile协作存储数据到全局内存（默认轴版本）
 *
 * @tparam RT 寄存器tile类型（必须满足ducks::rt::all概念）
 * @tparam GL 全局内存数组类型（必须满足ducks::gl::all概念）
 * @tparam COORD 坐标类型（必须满足ducks::coord::tile概念），默认为coord<rt<...>>
 * @param[out] dst 目标全局内存数组，数据将存储到这里
 * @param[in] src 源寄存器tile，从中存储数据
 * @param[in] idx 在全局内存数组中的tile坐标
 * 
 * 这是store函数的简化版本，使用默认轴方向(axis=2)。
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<rt<typename RT::T, GROUP_WARPS*RT::rows, RT::cols, typename RT::layout>>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2>(dst, src, idx);
}
