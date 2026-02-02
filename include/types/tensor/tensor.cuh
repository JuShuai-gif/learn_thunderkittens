/**
 * @file
 * @brief An aggregate header file for all the tensor types defined by ThunderKittens.
 */

#pragma once

#include "tt.cuh"

// 一个轻量级包装器，允许在分配张量内存时进行某些编译时检查
namespace kittens {
namespace ducks {
/**
 * @namespace tensor_allocator
 * @brief 张量内存分配器的概念和抽象类型所在的命名空间
 */
namespace tensor_allocator {


/**
 * @brief 用于标识张量内存的虚拟类型
 */
struct identifier {};


/**
 * @brief 所有tensor_allocator类型的概念约束
 * @tparam T 要检查是否符合概念要求的类型
 * 
 * 要求:
 * - T必须有一个嵌套类型identifier，且与tensor_allocator::identifier相同
 */
template<typename T> concept all = requires {
    typename T::identifier; // 检查T::identifier是否存在
} && std::is_same_v<typename T::identifier, identifier>; // 检查T::identifier是否与ducks::tensor_allocator::identifier相同
} // namespace tensor_allocator
} // namespace ducks

/**
 * @class tensor_allocator
 * @brief 张量内存分配器，使用Tensor Core指令管理共享内存
 * @tparam _nblocks_per_sm 每个流多处理器(SM)的块数，必须是1或2
 * @tparam _ncta 协作线程数组(CTA)的数量，必须是1或2
 * 
 * 这个分配器使用Tensor Core的tcgen05指令来分配共享内存，支持在编译时检查内存边界。
 */
template<int _nblocks_per_sm, int _ncta> struct tensor_allocator {
    // 静态断言确保模板参数的有效性
    static_assert(_nblocks_per_sm == 1 || _nblocks_per_sm == 2, "nblocks_per_sm must be 1 or 2");
    static_assert(_ncta == 1 || _ncta == 2, "ncta must be 1 or 2");
    // 标识符，用于概念检查
    using identifier = ducks::tensor_allocator::identifier;
    // 静态常量：每个SM的块数
    static constexpr int nblocks_per_sm = _nblocks_per_sm;
    // 静态常量：分配器的列数（对齐到32字节）
    static constexpr int cols =((MAX_TENSOR_COLS/nblocks_per_sm) / 32) * 32;
    // 静态常量：CTA数量
    static constexpr int ncta = _ncta;
    // 分配的内存基地址
    uint32_t addr;

    /**
     * @brief 在编译时检查张量内存分配的边界
     * @tparam TT 张量类型（必须满足ducks::tt::all概念）
     * @tparam col_offset 列偏移量
     * 
     * 在编译时静态检查分配的tile是否超出分配器的边界。
     */
    template<ducks::tt::all TT, int col_offset> __device__ inline void check_bounds() {
        static_assert(col_offset >= 0 && col_offset + TT::cols <= cols, "Tile allocation extends out of bounds of the tensor allocator!");
    }


    /**
     * @brief 构造函数，分配共享内存
     * 
     * 使用Tensor Core指令(tcgen05.alloc)分配共享内存。根据ncta的值，
     * 使用不同的CTA组分配模式。分配后，通过同步屏障确保所有线程都能看到分配的地址。
     */
    __device__ inline tensor_allocator() {
        __shared__ uint32_t shared_addr;// 共享内存中的地址变量
        static_assert(cols>0 && cols%32==0, "cols must be a multiple of 32");
        if constexpr (ncta == 1) {
            // 单CTA分配模式
            if(warpid() == 0) { // 只在每个warp的第一个线程执行
                asm volatile(
                    "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;\n"
                ::  "l"((uint64_t)&shared_addr), "n"(cols)      // 分配cols列的共享内存
                );
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");   // 释放分配许可
            }
        }
        else {
            // 双CTA分配模式
            if(warpid() == 0) {
                asm volatile(
                    "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
                ::  "l"((uint64_t)&shared_addr), "n"(cols)  // 分配cols列的共享内存
                );
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");   // 释放分配许可
            }
        }
        // 内存屏障和同步
        asm volatile("tcgen05.fence::before_thread_sync;\n");   // 线程同步前的内存屏障
        asm volatile("bar.sync 0;\n");                          // 线程块同步
        asm volatile("tcgen05.fence::after_thread_sync;\n");    // 线程同步后的内存屏障

        // 从共享内存读取分配的地址到本地变量
        addr = shared_addr;
    }


    /**
     * @brief 获取指定超级通道(superlane)和列偏移量的内存地址
     * @param superlane 超级通道索引（0或1）
     * @param col_offset 列偏移量
     * @return 计算得到的内存地址
     * 
     * 对于半精度张量，每个超级通道有16行，因此地址偏移为(superlane*16)<<16
     */
    __device__ inline uint32_t get_addr(int superlane, int col_offset) const { 
        return addr + ((superlane*16) << 16) + col_offset; 
    }


    /**
     * @brief 获取指定列偏移量的内存地址（全精度版本）
     * @param col_offset 列偏移量
     * @return 计算得到的内存地址
     * 
     * 全精度张量不使用超级通道，所以没有superlane偏移
     */
    __device__ inline uint32_t get_addr(int col_offset) const { 
        return addr + col_offset; 
    }

    /**
     * @brief 分配半精度张量内存
     * @tparam TT 半精度张量类型（必须满足ducks::tt::half概念）
     * @param superlane 超级通道索引（0或1）
     * @param col_offset 列偏移量
     * @return 构造的张量对象
     * 
     * 在调试模式下，会进行运行时边界检查。对于fp8e8m0和fp8e4m3数据类型，
     * 需要将列数除以4，因为每个8位元素占用1/4个32位字的存储空间。
     */
    template<ducks::tt::half TT> __device__ inline auto allocate(int superlane, int col_offset) {
#ifndef NDEBUG
        // 计算实际分配的列数（考虑数据类型大小）
        int allocate_cols = std::is_same_v<typename TT::dtype, fp8e8m0> ? TT::cols/4 : TT::cols; // for fp8e8m0 and fp8e4m3, we need to divide by 4 to get the correct number of columns
        // 运行时边界检查        
        if(col_offset + allocate_cols > cols) {
            if(laneid() == 0) printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");  // 触发陷阱，用于调试
        }
        if(superlane < 0 || superlane > 1) {
            printf("Superlane must be 0 or 1! superlane: %d\n", superlane);
            asm volatile("trap;");
        }
#endif
        // 返回构造的张量对象
        return TT(get_addr(superlane, col_offset));
    }

    /**
     * @brief 分配全精度张量内存
     * @tparam TT 全精度张量类型（必须满足ducks::tt::full概念）
     * @param col_offset 列偏移量
     * @return 构造的张量对象
     * 
     * 在调试模式下，会进行运行时边界检查。
     */
    template<ducks::tt::full TT> __device__ inline auto allocate(int col_offset) {
#ifndef NDEBUG
        // 计算实际分配的列数（考虑数据类型大小）
        int allocate_cols = std::is_same_v<typename TT::dtype, fp8e8m0> ? TT::cols/4 : TT::cols;
        // 运行时边界检查
        if(col_offset + allocate_cols > cols) {
            if(laneid() == 0) printf("Tile allocation extends out of bounds of the tensor allocator! col_offset: %d, TT::cols: %d, allocator cols: %d\n", col_offset, TT::cols, cols);
            asm volatile("trap;");// 触发陷阱，用于调试
        }
#endif
        // 返回构造的张量对象
        return TT(get_addr(0, col_offset));
    }
    
    /**
     * @brief 析构函数，释放分配的内存
     * 
     * 使用Tensor Core指令(tcgen05.dealloc)释放共享内存。
     * 对于双CTA模式，需要额外的集群屏障同步。
     */
    __device__ inline ~tensor_allocator() {
        if constexpr (ncta == 1) {
            // 单CTA释放模式
            if(warpid() == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1;\n"
                ::  "r"(addr), "n"(cols)        // 释放之前分配的内存
                );
            }
        } else {
            // 双CTA释放模式
            if(warpid() == 0) {
                // 集群屏障同步：确保所有CTA都完成了内存访问
                asm volatile ("barrier.cluster.arrive.release.aligned;\n");
                asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
                // 释放内存
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1;\n"
                ::  "r"(addr), "n"(cols)        // 释放之前分配的内存
                );
            }
        }
    }
};

} // namespace kittens