/**
 * @file
 * @brief 寄存器向量转换操作
 * 
 * 提供在不同寄存器向量布局之间进行转换的功能。
 * 这些布局包括: ortho_l(正交布局), align_l(对齐布局), naive_l(朴素布局)
 */

struct vec_conversion_detail {

    // 这些辅助函数用于计算不同布局间的索引映射关系
    // 由于NVIDIA的寄存器布局设计复杂，需要这些工具来推导正确的索引
    
    // 从二维布局(inner_dim维度)中根据laneid和x/y坐标计算行索引
__device__ static inline int row_from_indices_dim2(int laneid, int inner_dim, int x_or_y) {
    return 8*inner_dim + (laneid%4)*2 + x_or_y;
}
    // 从一维布局中根据laneid和x/y坐标计算行索引
__device__ static inline int row_from_indices_dim1(int laneid, int x_or_y) {
    return 8*x_or_y + (laneid/4);
}
    // 计算二维布局中的规范源lane(用于shuffle操作)
    // 将偶数行(0,2,4,6)映射到lane 0-3，奇数行(1,3,5,7)映射到lane 4-7
__device__ static inline int canonical_src_lane_dim2(int row) {
    return (row/2)%4 + 4*(row%2); // draw even rows from 0...3 and odds from 4...7
}
    // 计算一维布局中的规范源lane(用于shuffle操作)
    // 每行包含4个连续元素，按列优先排列
__device__ static inline int canonical_src_lane_dim1(int row) {
    return (row*4)%32;
}

};

/**
 * @brief 将一个寄存器向量的数据复制到另一个寄存器向量
 * 
 * @tparam RV1 目标寄存器向量类型
 * @tparam RV2 源寄存器向量类型
 * @param dst[out] 目标寄存器向量，接收复制数据
 * @param src[in]  源寄存器向量，提供数据
 * 
 * @note 支持不同类型和布局之间的转换
 *       当布局相同时执行简单的类型转换
 *       当布局不同时执行复杂的shuffle操作重新排列数据
 */
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    KITTENS_CHECK_WARP// 检查是否在完整的warp中执行
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;  // 目标数据类型
    using D2 = RV2::dtype;  // 源数据类型
    
    // 情况1: 布局相同，执行简单的类型转换复制
    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy / typecast
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {// 外层维度循环
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {// 内层维度循环
                // 使用类型转换器进行类型安全的转换
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    }
    // 情况2: 布局不同，需要重新排列数据
    else {
        int laneid = ::kittens::laneid();  // 获取当前线程的lane ID(0-31)
        
        // 情况2.1: align_l -> ortho_l 布局转换
        // 从对齐布局(4x8)转换为正交布局(2x16)
        if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && 
                      std::is_same_v<typename RV2::layout, align_l>) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                // 处理x分量: 从源的前4个lanes(x.0-x.3)获取低16位数据
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][0].x : src[i][0].y,  // lane0-3取x, lane4-7取y
                    vec_conversion_detail::canonical_src_lane_dim2(
                        vec_conversion_detail::row_from_indices_dim1(laneid, 0)
                    )
                );
                
                // 处理y分量: 从源的后4个lanes(x.4-x.7)获取高16位数据
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    laneid < 4 ? src[i][1].x : src[i][1].y,  // lane0-3取第二组的x, lane4-7取第二组的y
                    vec_conversion_detail::canonical_src_lane_dim2(
                        vec_conversion_detail::row_from_indices_dim1(laneid, 1)
                    )
                );
            }
        }
        // 情况2.2: ortho_l -> align_l 布局转换
        // 从正交布局(2x16)转换为对齐布局(4x8)
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && 
                          std::is_same_v<typename RV2::layout, ortho_l>) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                // 处理x分量的低8位(来自正交布局的前8行)
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x,  // 正交布局的第一个16位数据
                    vec_conversion_detail::canonical_src_lane_dim1(
                        vec_conversion_detail::row_from_indices_dim2(laneid, 0, 0)
                    )
                );
                
                // 处理x分量的高8位(来自正交布局的前8行)
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].x,  // 正交布局的第一个16位数据
                    vec_conversion_detail::canonical_src_lane_dim1(
                        vec_conversion_detail::row_from_indices_dim2(laneid, 0, 1)
                    )
                );
                
                // 处理y分量的低8位(来自正交布局的后8行)
                dst[i][1].x = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y,  // 正交布局的第二个16位数据
                    vec_conversion_detail::canonical_src_lane_dim1(
                        vec_conversion_detail::row_from_indices_dim2(laneid, 1, 0)
                    )
                );
                
                // 处理y分量的高8位(来自正交布局的后8行)
                dst[i][1].y = packed_shfl_sync(
                    kittens::MASK_ALL,
                    src[i][0].y,  // 正交布局的第二个16位数据
                    vec_conversion_detail::canonical_src_lane_dim1(
                        vec_conversion_detail::row_from_indices_dim2(laneid, 1, 1)
                    )
                );
            }
        }
        // 情况2.3: naive_l -> ortho_l 布局转换
        // 从朴素布局(连续的寄存器)转换为正交布局
        else if constexpr (std::is_same_v<typename RV1::layout, ortho_l> && 
                          std::is_same_v<typename RV2::layout, naive_l>) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                // 处理低16位: 从源寄存器的前16个元素获取
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],  // 每2个正交向量共享一个朴素向量
                    16*(i%2) + 0 + (laneid/4)  // 计算偏移量
                );
                
                // 处理高16位: 从源寄存器的后16个元素获取
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],
                    16*(i%2) + 8 + (laneid/4)  // 偏移16字节+行偏移
                );
            }
        }
        // 情况2.4: ortho_l -> naive_l 布局转换
        // 从正交布局转换为朴素布局
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && 
                          std::is_same_v<typename RV2::layout, ortho_l>) {
            int lane_replication = laneid%4;  // 每个lane重复4次(0-3)
            
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D1 tmp = 0;
                // 处理边界情况: 当长度不是32的倍数时，需要特殊处理最后一个元素
                if(RV1::length%32==0 || i < RV1::outer_dim-1 || lane_replication<2) {
                    // 根据lane_replication选择正交向量的x或y分量
                    tmp = lane_replication%2 ? 
                          src[2*i + (lane_replication>=2)][0].y :  // 取y分量
                          src[2*i + (lane_replication>=2)][0].x;   // 取x分量
                }
                
                // 通过shuffle重新排列到朴素布局
                dst[i][0] = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    tmp,
                    (laneid%8)*4 + (laneid/8)  // 重新映射lane索引
                );
            }
        }
        // 情况2.5: naive_l -> align_l 布局转换
        // 从朴素布局转换为对齐布局
        else if constexpr (std::is_same_v<typename RV1::layout, align_l> && 
                          std::is_same_v<typename RV2::layout, naive_l>) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                // 处理第一个x分量(低8位)
                dst[i][0].x = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],
                    16*(i%2) + 0 + 2*(laneid%4) + 0  // 计算源索引
                );
                
                // 处理第一个y分量(次低8位)
                dst[i][0].y = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],
                    16*(i%2) + 0 + 2*(laneid%4) + 1
                );
                
                // 处理第二个x分量(高8位)
                dst[i][1].x = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],
                    16*(i%2) + 8 + 2*(laneid%4) + 0
                );
                
                // 处理第二个y分量(次高8位)
                dst[i][1].y = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    src[i/2][0],
                    16*(i%2) + 8 + 2*(laneid%4) + 1
                );
            }
        }
        // 情况2.6: align_l -> naive_l 布局转换
        // 从对齐布局转换为朴素布局
        else if constexpr (std::is_same_v<typename RV1::layout, naive_l> && 
                          std::is_same_v<typename RV2::layout, align_l>) {
            int lane_replication = laneid/8;  // 每8个lanes为一组(0-3)
            
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                D1 tmp = 0;
                // 处理边界情况
                if(RV1::length%32==0 || i < RV1::outer_dim-1 || laneid<16) {
                    // 根据laneid的低3位选择x或y分量
                    tmp = (laneid%8)<4 ? 
                          src[2*i + (lane_replication>=2)][lane_replication%2].x :  // 取x分量
                          src[2*i + (lane_replication>=2)][lane_replication%2].y;   // 取y分量
                }
                
                // 通过shuffle重新排列到朴素布局
                dst[i][0] = packed_shfl_sync(
                    kittens::MASK_ALL, 
                    tmp,
                    4*(laneid%2) + (laneid%8)/2 + (laneid&0b11000)  // 重新映射lane索引
                );
            }
        }
    }
}