/**
 * @file
 * @brief 用于组协作在共享内存和寄存器之间直接传输数据的函数
 */

/**
 * @brief 协作地从共享向量加载数据到跨warpgroup分割的寄存器向量
 *
 * @tparam RV 寄存器向量类型
 * @tparam SV 共享向量类型
 * @param dst[out] 目标寄存器向量
 * @param src[in]  源共享向量
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    // 如果是单个warp的情况    
    if constexpr (GROUP_WARPS == 1) {
        // 静态断言：确保共享向量和寄存器向量长度匹配        
        static_assert(SV::length == RV::length);
        
        int laneid = ::kittens::laneid();
        // 获取共享内存源指针的32位表示
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
        // 同步warp内所有线程，确保内存访问的一致性
        __syncwarp();
        // 处理对齐布局（align_l）
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            // 循环处理外层维度，每次处理4个元素
            for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
                // 计算共享内存索引
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                // 计算目标寄存器向量的外层维度和内层维度索引
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // 执行合并加载
                if(idx < dst.outer_dim*16) {
                    U2 tmp;
                    // 从共享内存加载数据
                    move<U2>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*idx);
                    // 数据类型转换并存储到寄存器
                    dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(tmp);
                }
            }
            __syncwarp();
            // 通过shuffle操作在warp内共享数据
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = 8*(w%4) + (laneid%4); // 每64列重复一次
                dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
                dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {        // 处理正交布局（ortho_l）
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            // 循环处理外层维度，每次处理2个元素
            #pragma unroll
            for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // 执行合并加载（希望反向访问也能触发合并）
                if(idx < dst.outer_dim*16) {
                    U tmp;
                    move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*idx);
                    // 根据laneid的奇偶性存储到不同的向量分量
                    if(laneid%2==0) dst[o_dim][0].x =  base_types::convertor<T, U>::convert(tmp);
                    else dst[o_dim][0].y = base_types::convertor<T, U>::convert(tmp);
                }
            }
            __syncwarp();
            // 通过shuffle操作在warp内共享数据
            #pragma unroll
            for(auto w = 0; w < dst.outer_dim; w++) {
                int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
                dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
                dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {        // 处理朴素布局（naive_l）
            #pragma unroll
            // 直接按顺序加载数据
            for(auto w = 0; w < dst.outer_dim; w++) {
                // 处理边界条件
                if(w < dst.outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    U tmp;
                    move<U>::lds(tmp, src_ptr + sizeof(typename SV::dtype)*(w*32 + laneid));
                    dst[w][0] = base_types::convertor<T, U>::convert(tmp);
                }
            }
        }
    }
    else {    // 多个warp的情况
        // 静态断言：确保共享向量长度是寄存器向量长度的整数倍
        static_assert(SV::length == RV::length*GROUP_WARPS);// confirm size correct
        // 提取当前warp对应的共享向量子部分
        auto &_src = src.template subvec<RV::length>(warpid()); // pretend it's smaller and do warp-level load
        // 调用warp级别的load函数
        ::kittens::group<1>::load(dst, _src); // warp-level
    }
}

/**
 * @brief 协作地将数据从跨warpgroup分割的寄存器向量存储到共享向量
 *
 * @tparam RV 寄存器向量类型
 * @tparam SV 共享向量类型
 * @param dst[out] 目标共享向量
 * @param src[in]  源寄存器向量
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    // 如果是单个warp的情况
    if constexpr (GROUP_WARPS == 1) {
        // 静态断言：确保共享向量和寄存器向量长度匹配        
        static_assert(SV::length == RV::length);
        
        int laneid = ::kittens::laneid();
        // 获取共享内存目标指针的32位表示
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        // 同步warp内所有线程，确保内存访问的一致性
        __syncwarp();
        // 处理对齐布局（align_l）
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            // 循环处理外层维度，每次处理4个元素
            for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
                int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
                int o_dim = w*4 + (laneid/4) / 2;
                int i_dim = (laneid/4) % 2;
                // this should be a maximally coalesced store. I hope!
                // 执行合并存储
                if(idx < src.outer_dim*16) {
                    // 数据类型转换
                    U2 tmp = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
                    // 存储到共享内存
                    move<U2>::sts(dst_ptr + sizeof(typename SV::dtype)*idx, tmp);
                }
            }
        }
        // 处理正交布局（ortho_l）
        else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
            // otherwise there will be some pain :/
            #pragma unroll
            // 循环处理外层维度，每次处理2个元素
            for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
                int idx = w*32 + (laneid%4)*8 + (laneid/4);
                int o_dim = w*2 + (laneid%4) / 2;
                // 执行合并存储（希望反向访问也能触发合并）
                if(idx < src.outer_dim*16) {
                    U tmp;
                    // 根据laneid的奇偶性从不同的向量分量读取
                    if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                    else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                    // 存储到共享内存
                    move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*idx, tmp);
                }
            }
        }
        else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {        // 处理朴素布局（naive_l）
            #pragma unroll
            // 直接按顺序存储数据
            for(auto w = 0; w < src.outer_dim; w++) {
                // 处理边界条件
                if(w < src.outer_dim-1 || RV::length%32 == 0 || laneid<16) {
                    // 数据类型转换
                    U tmp = base_types::convertor<U, T>::convert(src[w][0]);
                    // 存储到共享内存
                    move<U>::sts(dst_ptr + sizeof(typename SV::dtype)*(w*32 + laneid), tmp);
                }
            }
        }
    }
    else {    // 多个warp的情况
        // 静态断言：确保共享向量长度是寄存器向量长度的整数倍
        static_assert(SV::length == RV::length*GROUP_WARPS);// confirm size correct
        // 提取当前warp对应的共享向量子部分
        auto &_dst = dst.template subvec<RV::length>(warpid()); // pretend it's smaller and do warp-level load
        // 调用warp级别的store函数
        ::kittens::group<1>::store(_dst, src); // warp-level
    }
}