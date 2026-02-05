/**
 * @file
 * @brief 存储在寄存器中的向量上的映射操作。
 */

/* ----------  向量映射  ---------- */

/**
 * @brief 对向量执行一元操作。
 *
 * @tparam op 要执行的一元操作。
 * @tparam T 向量的类型。
 * @param dst[out] 存储结果的目标向量。
 * @param src[in] 要执行操作的源向量。
 */
template<typename op, ducks::rv::all T>
__device__ static inline void unary_op(T &dst, const T &src) {
    #pragma unroll// 循环展开优化
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll// 内循环展开优化
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j]);// 应用一元操作
        }
    }
}
/**
 * @brief 对两个向量执行二元操作。
 *
 * @tparam op 要执行的二元操作。
 * @tparam T 向量的类型。
 * @param dst[out] 存储结果的目标向量。
 * @param lhs[in] 操作的左侧向量。
 * @param rhs[in] 操作的右侧向量。
 */
template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll// 循环展开优化
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll// 内循环展开优化
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(lhs[i][j], rhs[i][j]);// 应用二元操作
        }
    }
}
/**
 * @brief 对向量和标量执行二元操作。
 *
 * @tparam op 要执行的二元操作。
 * @tparam T 向量的类型。
 * @param dst[out] 存储结果的目标向量。
 * @param src[in] 操作的源向量。
 * @param param[in] 操作的标量参数。
 */
template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll // 循环展开优化
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll// 内循环展开优化
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j], param);// 应用二元操作（向量和标量）
        }
    }
}
/**
 * @brief 对向量和解包标量执行二元操作。
 *
 * @tparam op 要执行的二元操作。
 * @tparam T 向量的类型（瓦片布局）。
 * @param dst[out] 存储结果的目标向量。
 * @param src[in] 操作的源向量。
 * @param param[in] 操作的解包标量参数。
 */
template<typename op, ducks::rv::tile_layout T>
__device__ static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/**
 * @brief 对向量的每个元素应用lambda函数。
 *
 * @tparam RV 向量类型。
 * @tparam Lambda lambda函数类型。
 * @param dst[out] 存储结果的目标向量。
 * @param src[in] 源向量。
 * @param lambda 要应用的lambda函数。
 */
template<ducks::rv::all RV, typename Lambda>
__device__ static inline void apply(RV &dst, const RV &src, Lambda &&lambda) {
    int group_offset = 0;  // 组偏移量，用于多warp分组
    if constexpr(GROUP_WARPS > 1) {
        group_offset = warpid()*RV::length;  // 计算当前warp的偏移量
    }
    static_assert(sizeof(typename RV::dtype) != 1, "Cannot apply lambda to 8-bit types");  // 静态断言：不支持8位类型
    

    static_assert(sizeof(RV::T) != 1, "Cannot apply lambda to 8-bit types");

    // 正交布局（ortho_layout）处理
    if constexpr (ducks::rv::ortho_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*16 + ::kittens::laneid()/4;  // 计算基础索引
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);  // 处理x分量
            dst[i][0].y = lambda(base_idx+8, src[i][0].y);  // 处理y分量
        }
    }
    // 对齐布局（align_layout）处理
    else if constexpr (ducks::rv::align_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*16 + 2*(::kittens::laneid()%4);  // 计算基础索引
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);  // 处理第一个x分量
            dst[i][0].y = lambda(base_idx+1, src[i][0].y);  // 处理第一个y分量
            dst[i][1].x = lambda(base_idx+8, src[i][1].x);  // 处理第二个x分量
            dst[i][1].y = lambda(base_idx+9, src[i][1].y);  // 处理第二个y分量
        }
    }
    // 默认布局处理
    else {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*32 + ::kittens::laneid();  // 计算基础索引
            // 边界检查：如果不在最后一个块或长度是32的倍数，或线程ID小于16
            if (i < dst.outer_dim-1 || dst.length%32 == 0 || ::kittens::laneid()<16) {
                dst[i][0] = lambda(base_idx, src[i][0]);  // 应用lambda函数
            }
        }
    }
}

/**
 * @brief 对向量的每个元素应用lambda函数，返回新向量（函数式风格）。
 *
 * @tparam RV 向量类型。
 * @tparam Lambda lambda函数类型。
 * @param src[in] 源向量。
 * @param lambda 要应用的lambda函数。
 * @return RV 返回应用lambda后的新向量。
 */
template<ducks::rv::all RV, typename Lambda>
__device__ static inline RV apply(const RV &src, Lambda &&lambda) {
    RV dst; // 创建目标向量
    apply<RV, Lambda>(dst, src, std::forward<Lambda>(lambda));// 调用上述apply函数
    return dst;// 返回结果向量
}

/* ----------  美化操作的包装器  ---------- */

// ---- 常量操作 ----

/**
 * @brief 将寄存器向量的所有元素设置为零。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 要设置为零的目标向量。
 */
template<ducks::rv::all T>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);// 使用zero一元操作
}
/**
 * @brief 将寄存器向量的所有元素设置为一。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 要设置为一的目标向量。
 */
template<ducks::rv::all T>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);// 使用one一元操作
}
/**
 * @brief 将寄存器向量的所有元素设置为正无穷大。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 要设置为正无穷大的目标向量。
 */
template<ducks::rv::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);// 使用pos_infty一元操作
}
/**
 * @brief 将寄存器向量的所有元素设置为负无穷大。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 要设置为负无穷大的目标向量。
 */
template<ducks::rv::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);// 使用neg_infty一元操作
}

// ---- 一元操作 ----

/**
 * @brief 将元素从一个寄存器向量复制到另一个。
 *
 * @tparam T 目标寄存器向量类型。
 * @tparam U 源向量类型。
 * @param dst[out] 要复制到的目标向量。
 * @param src[in] 要复制的源向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // 使用copy2二元操作（忽略第二个参数）
}
/**
 * @brief 对寄存器向量逐元素应用指数函数（自然底数e）。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储指数值的目标向量。
 * @param src[in] 要应用指数函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);// 使用exp一元操作
}


template<ducks::rv::all T>
__device__ static inline T exp(const T &src) {
    T dst;// 创建临时向量
    exp(dst, src);// 调用exp操作
    return dst;// 返回结果
}

/**
 * @brief 对寄存器向量逐元素应用指数函数（底数2）。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储指数值的目标向量。
 * @param src[in] 要应用指数函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);// 使用exp2一元操作
}
template<ducks::rv::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}

/**
 * @brief 对寄存器向量逐元素应用自然对数函数。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储对数值的目标向量。
 * @param src[in] 要应用对数函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}
/**
 * @brief 对寄存器向量逐元素应用以2为底的对数函数。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储对数值的目标向量。
 * @param src[in] 要应用对数函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}
/**
 * @brief 对寄存器向量逐元素应用绝对值函数。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储绝对值的目标向量。
 * @param src[in] 要应用绝对值函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}
/**
 * @brief 对寄存器向量逐元素应用修正线性单元（ReLU）函数。
 *
 * @tparam T 寄存器向量类型。
 * @param dst[out] 存储ReLU值的目标向量。
 * @param src[in] 要应用ReLU函数的源向量。
 */
template<ducks::rv::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}

// ---- 二元操作 ----

/**
 * @brief 计算两个寄存器向量逐元素的最大值。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储最大值的目标向量。
 * @param lhs[in] 最大值操作的第一个向量。
 * @param rhs[in] 最大值操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}
/**
 * @brief 计算两个寄存器向量逐元素的最小值。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储最小值的目标向量。
 * @param lhs[in] 最小值操作的第一个向量。
 * @param rhs[in] 最小值操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}
/**
 * @brief 计算两个寄存器向量逐元素的和。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储和的目标向量。
 * @param lhs[in] 加法操作的第一个向量。
 * @param rhs[in] 加法操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief 计算两个寄存器向量逐元素的差。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储差的目标向量。
 * @param lhs[in] 减法操作的第一个向量。
 * @param rhs[in] 减法操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief 计算两个寄存器向量逐元素的积。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储积的目标向量。
 * @param lhs[in] 乘法操作的第一个向量。
 * @param rhs[in] 乘法操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief 计算两个寄存器向量逐元素的商。
 *
 * @tparam T 寄存器向量类型。
 * @tparam U 第二个向量的类型。
 * @param dst[out] 存储商的目标向量。
 * @param lhs[in] 除法操作的第一个向量。
 * @param rhs[in] 除法操作的第二个向量。
 */
template<ducks::rv::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}