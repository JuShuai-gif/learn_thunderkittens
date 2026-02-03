/**
 * @file
 * @brief 用于warpgroup协作地在全局内存和寄存器之间直接传输数据的函数
 */

/**
 * @brief 协作地从全局内存加载数据到寄存器向量
 * 
 * @tparam RV 寄存器向量类型
 * @tparam U 源数组的数据类型
 * @param[out] dst 目标寄存器向量，数据将加载到此处
 * @param[in] src 全局内存中的源数组
 * @param idx 坐标索引，指定在源数组中的位置
 */
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void load(RV &dst, const GL &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    // 如果是单个warp的情况
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;
        // 获取源指针和当前lane ID
        U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();
                
        // 处理对齐布局（align_l）
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            // 循环处理外层维度，每次处理4个元素（w循环）
            for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
                // 计算全局内存索引
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                // 计算目标寄存器向量的外层维度和内层维度索引
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // 执行合并加载操作
                if(idx < dst.outer_dim*16)
                    dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
            }
            // 通过shuffle操作在warp内共享数据
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = 8*(w%4) + (laneid%4); // 每64列重复一次
                dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
                dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {// 处理正交布局（ortho_l）
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            // 循环处理外层维度，每次处理2个元素
            #pragma unroll
            for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
                // 计算全局内存索引（希望反向访问也能触发合并）
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // 执行合并加载
                if(idx < dst.outer_dim*16) {
                    T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                    // 根据laneid的奇偶性存储到不同的向量分量
                    if(laneid%2==0) dst[o_dim][0].x = tmp;
                    else dst[o_dim][0].y = tmp;
                }
            }
            // 通过shuffle操作在warp内共享数据
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = (laneid/4)*4 + 2*(w%2); // 每64列重复一次
                dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
                dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
            }
        }
        // 处理朴素布局（naive_l）
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            // 直接按顺序加载数据
            for(auto w = 0; w < dst.outer_dim; w++) {
                // 处理边界条件
                if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
                    dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[w*32 + laneid]);
                }
            }
        }
    }
    else {    // 多个warp的情况：调用warp级别的load函数
        // Call warp level load
        ::kittens::group<1>::load(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}

/**
 * @brief 协作地将数据从寄存器向量存储到全局内存
 * 
 * @tparam RV 寄存器向量类型
 * @tparam U 目标数组的数据类型
 * @param[out] dst 全局内存中的目标数组
 * @param[in] src 源寄存器向量
 * @param idx 坐标索引，指定在目标数组中的位置
 */
template<ducks::rv::all RV, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RV &src, const coord<rv<typename RV::T, GROUP_WARPS*RV::length, typename RV::layout>> &idx) {
    // 如果是单个warp的情况    
    if constexpr (GROUP_WARPS == 1) {
        using T2 = RV::dtype;
        using U = typename GL::dtype;
        using U2 = base_types::packing<U>::packed_type;
        using T = base_types::packing<T2>::unpacked_type;
        // 获取目标指针和当前lane ID        
        U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
        int laneid = ::kittens::laneid();
        // 处理对齐布局（align_l）        
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            // 循环处理外层维度，每次处理4个元素
            for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // 执行合并存储操作
                if(idx < src.outer_dim*16)
                    *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
            }
        }        // 处理正交布局（ortho_l）
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            // 循环处理外层维度，每次处理2个元素
            #pragma unroll
            for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // 执行合并存储
                if(idx < src.outer_dim*16) {
                    U tmp;
                    // 根据laneid的奇偶性从不同的向量分量读取
                    if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                    else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                    dst_ptr[idx] = tmp;
                }
            }
        }        // 处理朴素布局（naive_l）
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
            #pragma unroll
            // 直接按顺序存储数据
            for(auto w = 0; w < src.outer_dim; w++) {
                // 处理边界条件
                if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                    dst_ptr[w*32 + laneid] = base_types::convertor<U, T>::convert(src[w][0]);
                }
            }
        }
    }
    else {
        // 多个warp的情况：调用warp级别的store函数
        ::kittens::group<1>::store(dst, src, coord<RV>(idx.b, idx.d, idx.r, idx.c*GROUP_WARPS+warpid()));
    }
}