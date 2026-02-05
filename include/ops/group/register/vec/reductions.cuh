/**
 * @file
 * @brief 存储在寄存器中的向量上的规约操作。
 */

/* ----------  向量规约  ---------- */

/**
 * @brief 在一个warp内对寄存器向量的元素执行规约操作。
 *
 * 此函数应用指定的操作，将寄存器向量 `src` 的元素规约为单个值。
 * 结果存储在 `accum` 中。如果 `reset` 参数为真，规约将包含初始值 `src_accum`。
 * 规约操作在warp范围内执行，确保warp内线程之间的同步。
 *
 * @tparam op 要在元素上执行的操作。必须提供静态 `op` 方法。
 * @tparam RV 寄存器向量的类型。必须满足 `ducks::rv::all` 概念。
 * @tparam reset 布尔标志，指示是否在规约中包含初始值。
 * @param[out] dst_accum 规约操作的结果。
 * @param[in] src 要规约的寄存器向量。
 * @param[in] src_accum 如果 `reset` 为false，则规约中包含的初始值。
 */
template<typename op, ducks::rv::all RV, bool reset>
__device__ static inline void reduce(
        typename base_types::packing<typename RV::dtype>::unpacked_type &dst_accum,
        const RV &src,
        const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行
    using T = base_types::packing<typename RV::dtype>::unpacked_type;  // 定义解包后的类型别名
    int laneid = kittens::laneid();  // 获取当前线程在warp内的lane ID
    
    // 正交布局（ortho_l）处理
    if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);  // 初始化accum为第一个元素的x和y分量的规约结果
        
        #pragma unroll  // 循环展开优化
        for(int i = 1; i < src.outer_dim; i++) {
            accum = op::template op<T>(accum, src[i][0].x);  // 累加x分量
            accum = op::template op<T>(accum, src[i][0].y);  // 累加y分量
        }
        
        // 现在已将所有元素规约为8个不同的值，每个值在x, x+1, x+2, x+3（其中x≡0(mod4)）的lanes之间复制
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));  // 向下洗牌16位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));   // 向下洗牌8位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));   // 向下洗牌4位
        
        // 现在已将所有元素规约为1个不同的值，在lanes 0, 1, 2, 3之间复制
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);  // 如果reset为false，合并初始值
        
        // 最终结果已经获得（如果需要，已合并src_accum），最后广播回所有线程
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);  // 从lane 0广播结果
    }
    // 对齐布局（align_l）处理
    else if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);  // 初始化accum为第一个元素的两个分量的规约结果
        accum = op::template op<T>(accum,       src[0][1].x);    // 累加第一个元素的第三个分量
        accum = op::template op<T>(accum,       src[0][1].y);    // 累加第一个元素的第四个分量
        
        #pragma unroll  // 循环展开优化
        for(int i = 1; i < src.outer_dim; i++) {
            // 使用shfl_sync可能会更快，但复制可能更好。当然更简单。
            accum = op::template op<T>(accum, src[i][0].x);  // 累加x分量
            accum = op::template op<T>(accum, src[i][0].y);  // 累加y分量
            accum = op::template op<T>(accum, src[i][1].x);  // 累加第二个x分量
            accum = op::template op<T>(accum, src[i][1].y);  // 累加第二个y分量
        }
        
        // 现在已将所有元素规约为4个不同的值，每个值在x, x+4, x+8, ..., x+28（其中x<4）的lanes之间复制
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));  // 向下洗牌2位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));  // 向下洗牌1位
        
        // 现在已将所有元素规约为1个不同的值，在lanes 0, 4, 8, 12, ..., 28之间复制
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);  // 如果reset为false，合并初始值
        
        // 最终结果已经获得（如果需要，已合并src_accum），最后从lane 0广播回所有线程
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);  // 从lane 0广播结果
    }
    // 朴素布局（naive_l）处理
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        T accum = src[0][0];  // 初始化accum为第一个元素
        
        #pragma unroll  // 循环展开优化
        for(int i = 1; i < src.outer_dim; i++) {
            // 边界检查：如果不在最后一个块，或者索引在有效范围内
            if (i < src.outer_dim-1 || i*kittens::TILE_ROW_DIM<T>*2 + laneid < src.length) {
                accum = op::template op<T>(accum, src[i][0]);  // 累加元素
            }
        }
        
        // 使用树状规约模式，通过洗牌操作逐步合并warp内的值
        if(src.length > 16) accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));  // 向下洗牌16位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));  // 向下洗牌8位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));  // 向下洗牌4位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));  // 向下洗牌2位
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));  // 向下洗牌1位
        
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);  // 如果reset为false，合并初始值
        
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);  // 从lane 0广播最终结果
    }
}


/**
 * @brief Finds the maximum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The register vector to find the maximum in.
 */
template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src) {
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
    return max_val;
}

/**
 * @brief Finds the minimum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The register vector to find the minimum in.
 */
template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src) {
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The register vector to sum.
 */
template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src) {
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The register vector to multiply.
 */
template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src) {
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
    return prod_val;
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
    return max_val;
}

/**
 * @brief Finds the minimum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
    return prod_val;
}