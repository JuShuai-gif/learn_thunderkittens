/**
 * @file
 * @brief Group (collaborative warp) ops for loading tensor tiles into register tiles.
 * @brief 协作warp组操作，用于将张量块加载到寄存器块中。
 * @details 这部分代码主要使用Tensor Core指令进行异步加载，针对不同的数据布局和warp组大小进行优化。
 */

/**
 * @brief Load data from a tensor tile into a register tile asynchronously.
 * @brief 异步从张量内存块加载数据到寄存器块。
 * @details 这是张量内存和寄存器之间的高性能数据传输函数，使用Tensor Core指令实现异步加载。
 *          支持单warp和多warp协作模式，根据GROUP_WARPS的值选择不同路径。
 * 
 * @tparam RT The register tile type 寄存器块类型（要求为行主序布局）
 * @tparam TM The tensor memory tile type 张量内存块类型
 * @param dst[out] The destination register tile. 目标寄存器块
 * @param src[in]  The source tensor tile. 源张量内存块
 */
template<ducks::rt::row_layout RT, ducks::tt::all TM>
__device__ inline static void load_async(RT &dst, const TM &src) {
    // 情况1：单warp模式（GROUP_WARPS == 1）
    if constexpr (GROUP_WARPS == 1) {
        // 静态断言：确保寄存器块和张量内存块的尺寸匹配
        static_assert(RT::rows == TM::rows, "register tile and tensor tile must match rows");
        static_assert(RT::cols == TM::cols, "register tile and tensor tile must match cols");
        
        // 类型别名定义
        using T2 = RT::dtype;               // 寄存器块数据类型
        using U  = typename TM::dtype;      // 张量内存数据类型
        using U2 = base_types::packing<typename TM::dtype>::packed_type;        // 打包类型
        
        // 子情况1.1：处理8位数据类型（如FP8/INT8）
        if constexpr (sizeof(typename TM::dtype) == 1) {
            // 外层循环：遍历寄存器块的行
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                // 内层循环：遍历寄存器块的列
                #pragma unroll
                for(int j = 0; j < dst.width; j++) {
                    // 内联汇编：使用Tensor Core指令异步加载数据
                    // tcgen05.ld.sync.aligned.16x128b.x2.b32: 从张量内存加载16x128位数据，每个线程加载2个128位向量
                    // 地址计算：src.addr + 行偏移<<16 + 列偏移/(4/sizeof(U))
                    // <<16是因为张量内存地址格式中，行偏移在地址的高16位
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];\n"    // pack::16b doesn't make sense for fp8
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),      // 输出操作数：4个32位寄存器，对应4个128位向量
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))     // 输入操作数：地址
                    );
                }
            }
        } else if constexpr (sizeof(typename TM::dtype) == 2) {         // 子情况1.2：处理16位数据类型（如FP16/BF16）
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                #pragma unroll
                for(int j = 0; j < dst.width; j++) {
                    // 内联汇编：使用Tensor Core指令异步加载数据
                    // pack::16b: 表示加载的数据是16位打包格式
                    // 每个线程加载4个128位向量，每个向量包含8个16位值
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)) // 地址计算，不需要除法调整
                    );
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {        // 子情况1.3：处理32位数据类型（如FP32）
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                // 优化处理：根据寄存器块宽度进行不同粒度的加载
                
                // 情况1.3.1：宽度是4的倍数，使用最宽的内存访问模式
                if constexpr (dst.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j+=4) {
                        // 临时存储从内存加载的数据
                        U2 data[16];        // 16个64位向量（每个U2是两个32位浮点数）
                        // 内联汇编：使用Tensor Core指令异步加载数据
                        // .16x256b.x8: 每个线程加载8个256位向量，总共32个32位浮点数
                        // 输出操作数：32个32位浮点寄存器，对应16个64位向量                        
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),     // 第一个64位向量（两个32位浮点数）
                            "=f"(data[1].x), "=f"(data[1].y),       // 第二个64位向量
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y),
                            "=f"(data[4].x), "=f"(data[4].y),
                            "=f"(data[5].x), "=f"(data[5].y),
                            "=f"(data[6].x), "=f"(data[6].y),
                            "=f"(data[7].x), "=f"(data[7].y),
                            "=f"(data[8].x), "=f"(data[8].y),
                            "=f"(data[9].x), "=f"(data[9].y),
                            "=f"(data[10].x), "=f"(data[10].y),
                            "=f"(data[11].x), "=f"(data[11].y),
                            "=f"(data[12].x), "=f"(data[12].y),
                            "=f"(data[13].x), "=f"(data[13].y),
                            "=f"(data[14].x), "=f"(data[14].y),
                            "=f"(data[15].x), "=f"(data[15].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))// 地址计算
                        );
                        // 将加载的数据重新组织到寄存器块中
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            // 将16个数据块重新映射到4个连续的列中
                            dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                            dst.tiles[i][j+2].data[k] = base_types::convertor<T2, U2>::convert(data[k+8]);
                            dst.tiles[i][j+3].data[k] = base_types::convertor<T2, U2>::convert(data[k+12]);
                        }
                    }
                }
                // 情况1.3.2：宽度是2的倍数（但不是4的倍数）
                else if constexpr (dst.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j+=2) {
                        U2 data[8];// 8个64位向量
                        
                        // 内联汇编：每个线程加载4个256位向量
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),
                            "=f"(data[1].x), "=f"(data[1].y),
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y),
                            "=f"(data[4].x), "=f"(data[4].y),
                            "=f"(data[5].x), "=f"(data[5].y),
                            "=f"(data[6].x), "=f"(data[6].y),
                            "=f"(data[7].x), "=f"(data[7].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                        );
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            // 将8个数据块重新映射到2个连续的列中
                            dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                        }
                    }
                }
                // 情况1.3.3：任意宽度（逐列处理）
                else {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j++) {
                        U2 data[4]; // 4个64位向量
                        
                        // 内联汇编：每个线程加载2个256位向量
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),
                            "=f"(data[1].x), "=f"(data[1].y),
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                        );
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            // 将4个数据块直接存入当前列
                            dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                        }
                    }
                }
            }
        }
    }
    // 情况2：多warp协作模式（GROUP_WARPS == 4 或 8）
    else {
        // 静态断言：确保warp组大小是4或8
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        // 计算每个warp处理的行数
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        // 静态断言：确保张量内存块和寄存器块的列数匹配
        static_assert(TM::cols==RT::cols);
        // 静态断言：确保每个warp处理的行数与寄存器块行数匹配
        static_assert(warp_rows==RT::rows);
        // 子情况2.1：4个warp协作
        if constexpr (GROUP_WARPS == 4) {
            // 创建子张量块：每个warp处理总行数的1/4
            // warpid(): 当前warp在block中的ID（0-7或0-15等）
            // 32*warpid(): 计算当前warp在张量内存中的起始行（假设每个warp处理32行）
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            // 递归调用单warp版本的load_async函数
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
        // 子情况2.2：8个warp协作
        else {
            // 更复杂的warp映射：可能是4x2或2x4的网格布局
            // warpid()%4: warp在x方向的索引（0-3）
            // warpid()/4: warp在y方向的索引（0-1）
            // 32*(warpid()%4)+16*(warpid()/4): 计算当前warp在张量内存中的起始行
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            // 递归调用单warp版本的load_async函数
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
    }
}


/**
 * @brief Store data into a tensor tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination tensor tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::tt::all TM>
__device__ inline static void store_async(TM &dst, const RT &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::rows == TM::rows, "register tile and tensor tile must match rows");
        static_assert(RT::cols == TM::cols, "register tile and tensor tile must match cols");

        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = TM::dtype;
        using U2 = base_types::packing<U>::packed_type;

        if constexpr (sizeof(typename TM::dtype) == 2) {
            #pragma unroll
            for(int i = 0; i < src.height; i++) {
                if constexpr (src.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=4) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[3])
                        );
                    }
                }
                else if constexpr (src.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=2) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3])
                        );
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < src.width; j++) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[3])
                        );
                    }
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {
            #pragma unroll
            for(int i = 0; i < src.height; i++) {
                if constexpr(src.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=4) {
                        U2 data[16];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                            data[k+8] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+2].data[k]);
                            data[k+12] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+3].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y),
                            "f"(data[4].x), "f"(data[4].y),
                            "f"(data[5].x), "f"(data[5].y),
                            "f"(data[6].x), "f"(data[6].y),
                            "f"(data[7].x), "f"(data[7].y),
                            "f"(data[8].x), "f"(data[8].y),
                            "f"(data[9].x), "f"(data[9].y),
                            "f"(data[10].x), "f"(data[10].y),
                            "f"(data[11].x), "f"(data[11].y),
                            "f"(data[12].x), "f"(data[12].y),
                            "f"(data[13].x), "f"(data[13].y),
                            "f"(data[14].x), "f"(data[14].y),
                            "f"(data[15].x), "f"(data[15].y)
                        );
                    }
                }
                else if constexpr(src.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=2) {
                        U2 data[8];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y),
                            "f"(data[4].x), "f"(data[4].y),
                            "f"(data[5].x), "f"(data[5].y),
                            "f"(data[6].x), "f"(data[6].y),
                            "f"(data[7].x), "f"(data[7].y)
                        );
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < src.width; j++) {
                        U2 data[4];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y)
                        );
                    }
                }
            }
        }
    }
    else {
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        static_assert(TM::cols==RT::cols);
        static_assert(warp_rows==RT::rows);
        if constexpr (GROUP_WARPS == 4) {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
        else {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
    }
}