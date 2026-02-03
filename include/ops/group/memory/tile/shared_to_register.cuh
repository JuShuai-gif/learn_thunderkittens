/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 * @details 这是一个 warpgroup（通常为4个warp）协作函数，用于将共享内存中的数据加载到寄存器中。
 *          每个warp负责处理数据的不同部分，通过线程间协作高效完成数据传输。
 * 
 * @tparam RT The register tile type 寄存器块类型，包含数据类型、布局等信息
 * @tparam ST The shared tile type  共享内存块类型，包含数据类型、布局、尺寸等信息
 * @param dst[out] The destination register tile. 目标寄存器块
 * @param src[in]  The source shared tile. 源共享内存块
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    // 获取当前warp在block中的高度（每个warp处理的行数）
    constexpr int warp_height = RT::height;
    // 静态断言：确保共享内存块高度是寄存器块高度的整数倍，且等于GROUP_WARPS（warpgroup中的warp数量）
    static_assert(ST::rows/RT::rows == GROUP_WARPS, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::rows%RT::rows == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    // 静态断言：确保共享内存块和寄存器块的宽度相等
    static_assert(ST::cols==RT::cols, "Group load / store requires tile widths to match.");
    // 计算warpgroup内部的warp ID（用于确定当前warp处理数据块中的哪一部分）
    int local_warpid;
    // 如果GROUP_WARPS是4的倍数，则使用特殊的映射方式（可能是为了更好的bank冲突避免）
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();

    // 类型别名定义    
    using T2 = RT::dtype;       // 寄存器块的数据类型（可能是打包类型，如half2）
    using U  = ST::dtype;       // 共享内存块的数据类型
    using T  = base_types::packing<T2>::unpacked_type;      // 寄存器数据类型的解包类型
    using U2 = base_types::packing<U>::packed_type;         // 共享内存数据类型的打包类型

    // 获取当前线程在warp内的lane ID（0-31）
    int warp_laneid = ::kittens::laneid();

    // 将共享内存指针转换为共享内存空间的32位地址（用于后续地址计算）
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    
    // 外层循环：遍历寄存器块的行（每个warp处理的垂直部分）
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        // 内层循环：遍历寄存器块的列（水平方向）
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            // 情况1：处理16位数据类型（如half/half2）
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // 处理行主序布局的16位数据类型
                // 使用4个临时变量存储从共享内存加载的数据
                U2 tmp[4];

                // 计算当前线程需要加载的数据在共享内存中的位置
                // 行位置 = (当前warp在group中的ID * 每个warp处理的行数 + 当前行索引) * 行块大小 + (lane ID的低4位)
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (warp_laneid % 16);

                // 列位置 = 当前列索引 * 列块大小 + (lane ID的高位部分) * 8
                int col = j*dst.tile_size_col + (warp_laneid / 16) * 8;

                // 根据寄存器块布局选择加载方式：行主序使用ldsm4，否则使用转置加载ldsm4t
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }

                // 将加载的数据转换为寄存器块需要的类型，并存入寄存器块
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            // 情况2：处理8位数据类型（如int8）且寄存器块布局为行主序
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1) {
                // 处理行主序布局的8位数据类型
                
                // 将warp（32线程）分为两组，每组16个线程
                int warp_group_16 = (warp_laneid / 16);  // 0或1，表示当前线程属于哪一组
                int lane_in_16 = warp_laneid % 16;       // 在16线程组内的位置（0-15）

                // 计算数据位置                
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (lane_in_16 % 16); // find base row for warp in warpgroup and then distribute the 16 threads in the warp across the rows
                int col = j*dst.tile_size_col + warp_group_16 * 16; // find base column and then *16 for second half of the warp

                U2 tmp[4];
                // 根据布局选择加载方式
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }
                // 类型转换并存储
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            // 情况3：处理32位数据类型（如float）且寄存器块布局为行主序
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // 处理行主序布局的32位数据类型
                
                // 计算数据位置：每个线程处理2个连续的32位元素
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (warp_laneid / 4);
                int col = j*dst.tile_size_col + 2*(warp_laneid % 4);

                // 如果共享内存块是子块（sub-tile），需要调整偏移量
                if constexpr (ST::rows != ST::underlying_rows || ST::cols != ST::underlying_cols) { // subtile case
                    row += src.row_offset;
                    col += src.col_offset;
                }

                // 计算bank冲突避免的偏移量（swizzle）
                int blit = sizeof(typename ST::dtype) * ((warp_laneid%4) / 2);
                U2 tmp[4];

                // swizzle相关常量：swizzle重复间隔和子块列数
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);

                // 计算外层索引（用于处理swizzle模式）
                const int outer_idx = col/subtile_cols;
                // 计算两个8行间隔的地址（用于向量化加载）
                const uint32_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uint32_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);

                // 计算swizzle偏移（避免bank冲突）
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;

                // 执行8次标量加载操作，每次加载一个32位数据
                move<U>::lds(tmp[0].x, (addr_1+ 0)^swizzle_1);
                move<U>::lds(tmp[0].y, (addr_1+ 4)^swizzle_1);
                move<U>::lds(tmp[2].x, (addr_1+32)^swizzle_1);
                move<U>::lds(tmp[2].y, (addr_1+36)^swizzle_1);
                move<U>::lds(tmp[1].x, (addr_2+ 0)^swizzle_2);
                move<U>::lds(tmp[1].y, (addr_2+ 4)^swizzle_2);
                move<U>::lds(tmp[3].x, (addr_2+32)^swizzle_2);
                move<U>::lds(tmp[3].y, (addr_2+36)^swizzle_2);

                // 类型转换并存储
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);

                // 如果需要，交换向量中的元素顺序
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        dst.tiles[i][j].data[k] = T2{dst.tiles[i][j].data[k].y, dst.tiles[i][j].data[k].x};
                    }
                }
            }
            // 情况4：默认情况，处理列主序布局（适用于各种数据类型）
            else {
                // 处理列主序布局（通常用于矩阵运算如GEMM）
                
                // 计算数据位置：每个线程处理2行2列的数据
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + 2*(warp_laneid % 4);
                int col = j*dst.tile_size_col + (warp_laneid / 4);

                U2 tmp[4];
                // 执行8次标量加载，形成2x2的数据块                
                move<U>::lds(tmp[0].x, src.idx(shared_addr, {row+0, col+0}));
                move<U>::lds(tmp[0].y, src.idx(shared_addr, {row+1, col+0}));
                move<U>::lds(tmp[1].x, src.idx(shared_addr, {row+0, col+8}));
                move<U>::lds(tmp[1].y, src.idx(shared_addr, {row+1, col+8}));
                move<U>::lds(tmp[2].x, src.idx(shared_addr, {row+8, col+0}));
                move<U>::lds(tmp[2].y, src.idx(shared_addr, {row+9, col+0}));
                move<U>::lds(tmp[3].x, src.idx(shared_addr, {row+8, col+8}));
                move<U>::lds(tmp[3].y, src.idx(shared_addr, {row+9, col+8}));

                // 类型转换并存储
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
        }
    }
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 * @details 这是load函数的逆操作，将寄存器中的数据存储回共享内存。
 *          同样通过warpgroup协作完成，每个warp负责存储数据的不同部分。
 * 
 * @tparam RT The register tile type 寄存器块类型
 * @tparam ST The shared tile type 共享内存块类型
 * @param dst[out] The destination shared tile. 目标共享内存块
 * @param src[in]  The source register tile. 源寄存器块
 */
template<ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    // 获取每个warp处理的行数
    constexpr int warp_height = RT::height;

    // 静态断言：确保尺寸匹配（与load函数相同）
    static_assert(ST::rows/RT::rows == GROUP_WARPS, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::rows%RT::rows == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::cols==RT::cols, "Group load / store requires tile widths to match.");

    // 计算warpgroup内部的warp ID（映射方式与load函数相同）
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();

    // 类型别名定义
    using T2 = RT::dtype;           // 寄存器块数据类型
    using U  = ST::dtype;           // 共享内存块数据类型
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;

    // 获取当前线程在warp内的lane ID
    int warp_laneid = ::kittens::laneid();

    // 将共享内存指针转换为共享内存空间的32位地址
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    
    // 外层循环：遍历寄存器块的行（每个warp处理的垂直部分）
    #pragma unroll
    for(int i = 0; i < warp_height; i++) {
        // 内层循环：遍历寄存器块的列（水平方向）
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            // 情况1：处理16位数据类型（如half/half2）
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // 处理行主序布局的16位数据类型
                U2 tmp[4];

                // 将寄存器数据转换为共享内存数据类型                
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                // 针对Hopper或Blackwell架构的特殊处理
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
                // 计算数据在共享内存中的位置
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid % 16);
                int col = j*src.tile_size_col + (warp_laneid / 16) * 8;
                // 根据布局选择存储方式
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::stsm4(dst.idx(shared_addr, {row, col}), tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                else {
                    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
                }
#else
                // 非Hopper/Blackwell架构的处理方式
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    // 行主序布局：计算位置并执行4次向量存储
                    int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid / 4);
                    int col = j*src.tile_size_col + 2*(warp_laneid % 4);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[1]);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[2]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3]);
                }
                else {
                    int row = (local_warpid*warp_height + i)*src.tile_size_row + 2*(warp_laneid % 4);
                    int col = j*src.tile_size_col + (warp_laneid / 4);
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x);
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y);
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x);
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y);
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x);
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y);
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x);
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y);
                }
#endif
            }
            // 情况2：处理8位数据类型且寄存器块布局为行主序
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1) { 
                // 处理行主序布局的8位数据类型
                
                // 将warp分为两组，每组16个线程
                int warp_group_16 = (warp_laneid / 16);  // divide each warp into two groups of 16 threads
                int lane_in_16 = warp_laneid % 16;       // position in group of 16 threads

                // 计算数据位置
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (lane_in_16 % 16); // find base row for warp in warpgroup and then distribute the 16 threads in the warp across the rows
                int col = j*src.tile_size_col + warp_group_16 * 16; // find base column and then *16 for second half of the warp

                U2 tmp[4];
                // 类型转换
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                // 根据布局选择存储方式
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::stsm4(dst.idx(shared_addr, {row, col}), tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                else {
                    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
                }
            }
            // 情况3：处理32位数据类型且寄存器块布局为行主序
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // 处理行主序布局的32位数据类型
                
                // 计算数据位置
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid / 4);
                int col = j*src.tile_size_col + 2*(warp_laneid % 4);

                // 如果是子块，调整偏移量
                if constexpr (ST::rows != ST::underlying_rows || ST::cols != ST::underlying_cols) { // subtile case
                    row += dst.row_offset;
                    col += dst.col_offset;
                }
                // 计算bank冲突避免的偏移量
                int blit = sizeof(typename ST::dtype) * ((warp_laneid%4) / 2);

                // 临时寄存器变量
                T2 reg_tmp[4];

                // 如果需要，交换向量中元素的顺序
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = T2{src.tiles[i][j].data[k].y, src.tiles[i][j].data[k].x};
                    }
                }
                else {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = src.tiles[i][j].data[k];
                    }
                }

                // 类型转换到共享内存数据类型
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(reg_tmp[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(reg_tmp[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(reg_tmp[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(reg_tmp[3]);

                // swizzle相关常量
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);

                // 计算外层索引
                const int outer_idx = col/subtile_cols;

                // 计算两个8行间隔的地址
                const uint32_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uint32_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);

                // 计算swizzle偏移
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;

                // 执行8次标量存储操作
                move<U>::sts((addr_1+ 0)^swizzle_1, tmp[0].x);
                move<U>::sts((addr_1+ 4)^swizzle_1, tmp[0].y);
                move<U>::sts((addr_1+32)^swizzle_1, tmp[2].x);
                move<U>::sts((addr_1+36)^swizzle_1, tmp[2].y);
                move<U>::sts((addr_2+ 0)^swizzle_2, tmp[1].x);
                move<U>::sts((addr_2+ 4)^swizzle_2, tmp[1].y);
                move<U>::sts((addr_2+32)^swizzle_2, tmp[3].x);
                move<U>::sts((addr_2+36)^swizzle_2, tmp[3].y);
            }
            // 情况4：默认情况，处理列主序布局
            else {
                // 处理列主序布局（通常用于矩阵运算如GEMM）
                
                // 计算数据位置
                int row = (local_warpid*warp_height + i)*src.tile_size_row + 2*(warp_laneid % 4);
                int col = j*src.tile_size_col + (warp_laneid / 4);

                // 类型转换
                U2 tmp[4];

                // 执行8次标量存储操作，形成2x2的数据块
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x);
                move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y);
                move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x);
                move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y);
                move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x);
                move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y);
                move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x);
                move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y);
            }
        }
    }
}

/**
 * @brief Load a vector from a shared tile (naive layout).
 * @details 从共享内存中加载一个向量到寄存器向量中，使用简单的布局。
 *          这个函数由单个warp执行，不需要warpgroup协作。
 * 
 * @tparam RV 寄存器向量类型（简单布局）
 * @tparam ST 共享内存块类型
 * @param dst 目标寄存器向量
 * @param src 源共享内存块
 * @param row_col 起始位置（行，列）
 */
template<ducks::rv::naive_layout RV, ducks::st::all ST>
__device__ inline static auto load(RV &dst, const ST &src, int2 row_col) {

    // 检查是否在warp级别调用（通常用于调试）    
    KITTENS_CHECK_WARP;

    // 静态断言：确保共享内存块宽度足够容纳向量长度
    static_assert(ST::cols>=RV::length, "Shared tile must be at least as wide as the vector.");

    // 类型别名
    using T = RV::T;
    using U = ST::T;

    // 获取当前线程在warp内的lane ID
    int warp_laneid = ::kittens::laneid();

    // 将共享内存指针转换为共享内存空间的32位地址
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    // 循环：每个线程处理向量的一部分（跨warp的列循环）
    #pragma unroll
    for(int col = warp_laneid; col < dst.length; col+=WARP_THREADS) {
        U tmp;
        // 从共享内存加载单个元素
        move<U>::lds(tmp, src.idx(shared_addr, {row_col.x, row_col.y + col}));

        // 类型转换并存储到寄存器向量
        dst.data[col/WARP_THREADS][0] = base_types::convertor<T, U>::convert(tmp);
    }
}

/**
 * @brief Store a vector into a shared tile (naive layout).
 * @details 将寄存器向量中的数据存储到共享内存中，使用简单的布局。
 *          这是load向量函数的逆操作。
 * 
 * @tparam RV 寄存器向量类型（简单布局）
 * @tparam ST 共享内存块类型
 * @param dst 目标共享内存块
 * @param src 源寄存器向量
 * @param row_col 起始位置（行，列）
 */
template<ducks::rv::naive_layout RV, ducks::st::all ST>
__device__ inline static auto store(ST &dst, const RV &src, int2 row_col) {
    KITTENS_CHECK_WARP;
    static_assert(ST::cols>=RV::length, "Shared tile must be at least as wide as the vector.");
    using T = RV::T;
    using U = ST::T;
    int warp_laneid = ::kittens::laneid();

    // 将共享内存指针转换为共享内存空间的32位地址
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    // 循环：每个线程存储向量的一部分
    #pragma unroll
    for(int col = warp_laneid; col < src.length; col+=WARP_THREADS) {
        // 类型转换
        U tmp = base_types::convertor<U, T>::convert(src.data[col/WARP_THREADS][0]);
        // 存储到共享内存
        move<U>::sts(dst.idx(shared_addr, {row_col.x, row_col.y + col}), tmp);
    }
}