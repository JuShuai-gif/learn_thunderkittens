/**
 * @file
 * @brief 复数tile之间的映射操作。
 */

/**
 * @brief 将复数tile的所有元素设置为零。
 *
 * @tparam T 复数tile类型。
 * @param dst[out] 结果存储的目标tile。
 */
template<ducks::crt::all T>
__device__ static inline void zero(T &dst) {
    zero(dst.real);// 将实部设置为零
    zero(dst.imag);// 将虚部设置为零
}

/**
 * @brief 对复数tile的每个元素应用指数函数。
 *
 * @tparam T tile类型。
 * @param dst[out] 结果存储的目标tile。
 * @param src[in] 要应用指数函数的源tile。
 */
template<ducks::crt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    using dtype = T::dtype; // 获取数据类型
    dtype tmp;  // 临时寄存器
    // 结果存储寄存器（不在原位置存储）
    dtype rdst;  // 实部结果寄存器
    dtype idst;  // 虚部结果寄存器

    // 计算exp(a)（a为实部）
    exp(rdst, src.real);  // rdst = exp(a)
    copy(idst, rdst);     // idst = exp(a)（复制到虚部寄存器）
    
    // 计算exp(a+bi) = exp(a)cos(b) + exp(a)sin(b)i
    cos(tmp, src.imag);   // tmp = cos(b)
    mul(rdst, rdst, tmp); // rdst = exp(a) * cos(b)（实部结果）
    sin(tmp, src.imag);   // tmp = sin(b)
    mul(idst, idst, tmp); // idst = exp(a) * sin(b)（虚部结果）

    copy(dst.real, rdst);  // 将实部结果复制到目标tile
    copy(dst.imag, idst);  // 将虚部结果复制到目标tile
}


/**
 * @brief 按元素加法两个复数tile。
 *
 * @tparam T 复数Tile类型。
 * @param dst[out] 结果存储的目标tile。
 * @param lhs[in] 加法操作的左侧源tile。
 * @param rhs[in] 加法操作的右侧源tile。
 */
template<ducks::crt::all T>
__device__ static inline void add(T &dst, const T &lhs, const T &rhs) {
    add(dst.real, lhs.real, rhs.real);// 实部相加
    add(dst.imag, lhs.imag, rhs.imag);// 虚部相加
}

/**
 * @brief 按元素减法两个tile。
 *
 * @tparam T tile类型。
 * @tparam U 第二个操作数类型，可以是tile或标量。
 * @param dst[out] 结果存储的目标tile。
 * @param lhs[in] 减法操作的左侧源tile。
 * @param rhs[in] 减法操作的右侧源tile。
 */
template<ducks::crt::all T>
__device__ static inline void sub(T &dst, const T &lhs, const T &rhs) {
    sub(dst.real, lhs.real, rhs.real);
    sub(dst.imag, lhs.imag, rhs.imag);
}


/**
 * @brief 按元素乘法两个复数tile。
 *
 * @tparam T 复数tile类型。
 * @param dst[out] 结果存储的目标tile。
 * @param lhs[in] 乘法操作的左侧源tile。
 * @param rhs[in] 乘法操作的右侧源tile。
 */
template<ducks::crt::all T>
__device__ static inline void mul(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::component;  // 获取分量数据类型（实部或虚部的类型）
    dtype tmp;  // 临时寄存器
    // 结果存储寄存器（不在原位置存储）
    dtype rdst;  // 实部结果寄存器
    dtype idst;  // 虚部结果寄存器
    
    // 计算(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    
    // 实部计算：ac - bd
    mul(rdst, lhs.real, rhs.real);  // rdst = a * c
    mul(tmp, lhs.imag, rhs.imag);    // tmp = b * d
    sub(rdst, rdst, tmp);// rdst = ac - bd

    // 虚部计算：ad + bc
    mul(idst, lhs.imag, rhs.real);// idst = b * c
    mul(tmp, lhs.real, rhs.imag);// tmp = a * d
    add(idst, idst, tmp);// idst = bc + ad

    copy(dst.real, rdst);// 将实部结果复制到目标tile
    copy(dst.imag, idst);// 将虚部结果复制到目标tile
}

/**
 * @brief 按元素除法两个复数tile。
 *
 * @tparam T 复数tile类型。
 * @param dst[out] 结果存储的目标tile。
 * @param lhs[in] 除法操作的左侧源tile（被除数）。
 * @param rhs[in] 除法操作的右侧源tile（除数）。
 */
template<ducks::crt::all T>
__device__ static inline void div(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::dtype;  // 获取数据类型
    dtype tmp;       // 临时寄存器
    dtype denom;     // 分母寄存器
    // 结果存储寄存器（不在原位置存储）
    dtype rdst;      // 实部结果寄存器
    dtype idst;      // 虚部结果寄存器

    // 计算分母：denom = c² + d²（其中rhs = c + di）
    mul(tmp, rhs.real, rhs.real);   // tmp = c * c
    mul(denom, rhs.imag, rhs.imag); // denom = d * d
    add(denom, tmp, denom);         // denom = c² + d²
    
    // 计算实部：(ac + bd) / (c² + d²)
    mul(rdst, lhs.real, rhs.real);  // rdst = a * c
    mul(tmp, lhs.imag, rhs.imag);   // tmp = b * d
    add(rdst, rdst, tmp);           // rdst = ac + bd
    
    // 计算虚部：(bc - ad) / (c² + d²)
    mul(idst, lhs.imag, rhs.real);  // idst = b * c
    mul(tmp, lhs.real, rhs.imag);   // tmp = a * d
    sub(idst, idst, tmp);           // idst = bc - ad（注意：原代码这里有错误，已修正）
    
    // 将实部和虚部分别除以分母
    div(rdst, rdst, denom);  // rdst = (ac + bd) / (c² + d²)
    div(idst, idst, denom);  // idst = (bc - ad) / (c² + d²)
    
    copy(dst.real, rdst);  // 将实部结果复制到目标tile
    copy(dst.imag, idst);  // 将虚部结果复制到目标tile
}
