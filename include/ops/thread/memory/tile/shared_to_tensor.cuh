/**
 * @file
 * @brief 将共享内存加载到张量瓦片的单线程操作
 * @note 使用Tensor Core G5 (tcgen05) 指令进行高效的内存传输
 */

#include "../../mma/tcgen05.cuh"


/**
 * @brief 异步加载MxNV缩放瓦片（无信号量版本）
 * 
 * 从共享内存瓦片异步加载数据到张量内存瓦片，用于缩放操作。
 * 该函数使用tcgen05.cp指令进行跨CTA的内存复制，支持1个或2个CTA。
 * 
 * @tparam ncta CTA数量，必须为1或2
 * @tparam TT 目标张量瓦片类型，必须是完整的张量瓦片
 * @tparam ST 源共享内存瓦片类型，必须是共享内存瓦片
 * @param dst 目标张量瓦片引用
 * @param src 源共享内存瓦片引用
 * 
 * @note 
 * 1. 目标张量瓦片类型必须是fp8e8m0或fp8e4m3（用于缩放操作）
 * 2. 源和目标必须具有相同的数据类型
 * 3. 张量内存缩放瓦片固定为128x16，共享内存缩放瓦片固定为32x16
 * 4. 共享内存瓦片不能使用TMA-swizzle排列
 */
template<int ncta=1, kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async(TT &dst, const ST &src) {
    static_assert(ncta == 1 || ncta == 2, "ncta must be 1 or 2"); ///< 仅支持1个或2个CTA
    static_assert(std::is_same_v<typename TT::T, kittens::fp8e8m0> || 
                  std::is_same_v<typename TT::T, kittens::fp8e4m3>, 
                  "Scale TT must be fp8e8m0 or fp8e4m3"); ///< 目标类型必须是8位浮点缩放类型
    static_assert(std::is_same_v<typename TT::T, typename ST::T>, 
                  "Scale TT and ST must have the same type"); ///< 源和目标类型必须相同
    static_assert(TT::rows == 128 && TT::cols == 16, 
                  "Tensor memory scale tile must always be 128x16"); ///< 张量内存瓦片尺寸固定
    static_assert(ST::rows == 32 && ST::cols == 16, 
                  "Shared memory scale tile must always be 32x16"); ///< 共享内存瓦片尺寸固定
    static_assert(!ST::swizzle, 
                  "Shared memory scale tile must not be TMA-swizzled"); ///< 禁止TMA-swizzle排列

    // 创建共享内存描述符，描述源数据的布局（128字节步长，0偏移）
    uint64_t st_desc = kittens::detail::matrix_descriptor_raw(&src.data[0], 128, 128, 0);

    uint64_t st_desc = kittens::detail::matrix_descriptor_raw(&src.data[0], 128, 128, 0);

    if constexpr (ncta == 1) {
        // 单CTA版本：使用1个CTA组，32x128位数据，4个warp
        asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" 
                     :: "r"(dst.addr), "l"(st_desc));
    } else {
        // 双CTA版本：使用2个CTA组，32x128位数据，4个warp
        asm volatile("{tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;}" 
                     :: "r"(dst.addr), "l"(st_desc));
    }
}

/**
 * @brief 异步加载MxNV缩放瓦片（带信号量版本）
 * 
 * 从共享内存瓦片异步加载数据到张量内存瓦片，并在加载后提交操作到信号量。
 * 在信号量等待前插入张量操作的栅栏。
 * 
 * @tparam ncta CTA数量，必须为1或2
 * @tparam TT 目标张量瓦片类型，必须是完整的张量瓦片
 * @tparam ST 源共享内存瓦片类型，必须是共享内存瓦片
 * @param dst 目标张量瓦片引用
 * @param src 源共享内存瓦片引用
 * @param sem 信号量引用，用于同步操作完成
 * 
 * @note 
 * 1. 执行异步加载操作
 * 2. 提交操作到信号量
 * 3. 在信号量等待前插入张量操作栅栏（确保后续同步操作正确）
 */
template<int ncta=1, kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async(TT &dst, const ST &src, kittens::semaphore &sem) {
    // 调用无信号量版本的异步加载
    load_mxnv_scale_async<ncta>(dst, src);
    // 提交操作到信号量，使用Tensor Core G5的提交机制
    kittens::detail::tcgen05::commit<ncta>(sem);
    // 在信号量等待前插入张量操作的栅栏
    kittens::tensor_before_thread_sync();
}

/**
 * @brief 双CTA版本的MxNV缩放瓦片异步加载（无信号量版本）
 * 
 * 这是load_mxnv_scale_async<2>的便捷包装函数，固定使用2个CTA。
 * 
 * @tparam TT 目标张量瓦片类型，必须是完整的张量瓦片
 * @tparam ST 源共享内存瓦片类型，必须是共享内存瓦片
 * @param dst 目标张量瓦片引用
 * @param src 源共享内存瓦片引用
 */
template<kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async2(TT &dst, const ST &src) {
    load_mxnv_scale_async<2>(dst, src);
}

/**
 * @brief 双CTA版本的MxNV缩放瓦片异步加载（带信号量版本）
 * 
 * 这是load_mxnv_scale_async<2>的便捷包装函数，固定使用2个CTA并接受信号量。
 * 
 * @tparam TT 目标张量瓦片类型，必须是完整的张量瓦片
 * @tparam ST 源共享内存瓦片类型，必须是共享内存瓦片
 * @param dst 目标张量瓦片引用
 * @param src 源共享内存瓦片引用
 * @param sem 信号量引用，用于同步操作完成
 */
template<kittens::ducks::tt::full TT, kittens::ducks::st::all ST>
__device__ inline static void load_mxnv_scale_async2(TT &dst, const ST &src, kittens::semaphore &sem) {
    load_mxnv_scale_async<2>(dst, src, sem);
}
