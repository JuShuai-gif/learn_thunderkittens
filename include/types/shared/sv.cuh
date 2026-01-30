#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
// 用于类型推断的标识结构体
namespace kittens{

namespace ducks{


namespace sv{
// 向量类型标识符基础结构
struct identifier 
{
    
};
// 概念检查：检查类型T是否具有正确的向量标识符
template <typename T>
concept all = requires{
    typename T::identifier;// 检查T::identifier是否存在
} && std::is_same_v<typename T::identifier,identifier>;// 检查T::identifier是否为ducks::sv::identifier
}
}
/**
 * @brief 共享内存向量结构，支持不同数据类型的向量操作
 *
 * @tparam _T 元素的数据类型（未打包的原始类型）
 * @tparam _length 向量的长度（元素数量）
 */
template<typename _T,size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {// 默认对齐的共享内存向量结构
    using identifier = ducks::sv::identifier;// 向量类型标识符
    using T = base_types::packing<_T>::unpacked_type;// 未打包的数据类型
    using T2 = base_types::packing<_T>::packed_type;// 打包的数据类型
    using dtype = T; ///< 向量中元素的数据类型

    static constexpr int length = _length; ///< 向量的长度（以元素为单位）
    static_assert(length % TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = length / TILE_ROW_DIM<T>;  ///< 以子tile为单位的长度

// 架构特定的类型检查
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    // Hopper和Blackwell架构不支持某些FP8的打包格式
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
#endif
#if defined(DF_BLACKWELL)
    // Blackwell架构的额外类型检查
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4> || !std::is_same_v<T2, fp8e8m0_4>, "Unsupported type for fp8");
#endif

// 架构特定的内存分配策略
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    // Hopper和Blackwell架构：向上舍入到最近的128字节边界
    // 这是为了内存对齐优化，确保向量数据在128字节边界对齐
    static constexpr int num_alloc_elements = ((length * sizeof(dtype) + 127) / 128) * (128 / sizeof(dtype)); // round up to the nearest 128-byte boundary
#else
    // 其他架构：直接使用向量长度
    static constexpr int num_alloc_elements = length;
#endif
    dtype data[num_alloc_elements]; ///< 实际的共享向量数据存储

    /**
     * @brief 根据索引计算内存地址
     * @param ptr 指向数据的指针
     * @param idx 元素索引
     * @return 指向计算地址的指针
     * 注意：虽然看起来简单，但在共享地址空间中进行计算时很有用
     */
    __device__ static inline T* idx(T* ptr,int idx){// useful for computations in shared address space, as silly as it sounds.
        return ptr[idx];// 简单的数组索引
    }

    // 下标运算符重载（支持1D索引）
    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    /**
     * @brief 创建子向量引用
     * @tparam sub_length 子向量的长度
     * @param idx 子向量在原始向量中的索引（以子向量为单位）
     * @return 对子向量的引用
     */
    template<int sub_length> __device__ inline sv<_T, sub_length> &subvec(int idx) {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }

    /**
     * @brief 创建子向量引用（const版本）
     * @tparam sub_length 子向量的长度
     * @param idx 子向量在原始向量中的索引（以子向量为单位）
     * @return 对子向量的常量引用
     */
    template<int sub_length> __device__ inline const sv<_T, sub_length> &subvec(int idx) const {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }

    /**
     * @brief 赋值运算符，用于将整个向量设置为指定值（默认在warp范围内执行）
     * @param value 要设置的值
     * 使用warp级并行化：每个线程处理向量的一部分元素
     */
    __device__ inline void operator=(const dtype &value) { // runs at warp scope by default
        #pragma unroll// 循环展开优化
        for(int i = kittens::laneid(); i < length; i += WARP_THREADS) {
            data[i] = value;
        }
    }


};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// 常用向量类型的别名，提供更简洁的语法
template<size_t _length> using sv_bf = sv<bf16,  _length>;// bfloat16类型的向量
template<size_t _length> using sv_hf = sv<half,  _length>;// half类型的向量
template<size_t _length> using sv_fl = sv<float, _length>;// float类型的向量
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<int _length> using sv_fp8e4m3 = sv<fp8e4m3, _length>; // FP8 E4M3类型的向量
template<int _length> using sv_fp8e5m2 = sv<fp8e5m2, _length>;// FP8 E5M2类型的向量
#endif
#if defined(DF_BLACKWELL)
template<int _length> using sv_fp8e8m0 = sv<fp8e8m0, _length>;// FP8 E8M0类型的向量
template<int _length> using sv_fp4e2m1_2 = sv<fp4e2m1_2, _length>;// FP4 E2M1（打包为2个）类型的向量
#endif

/* ----------  PRINTOUTS  ---------- */

template<ducks::sv::all SV>
__device__ inline void print(const SV& sv) {
    printf("Shared Vector %d:\n", SV::length);
    for(int i = 0; i < SV::length; i++) {
        if constexpr (std::is_same_v<typename SV::dtype, bf16>) {
            printf("%f ", __bfloat162float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, half>) {
            printf("%f ", __half2float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, float>) {
            printf("%f ", sv[i]);
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e4m3>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
#ifdef DF_BLACKWELL
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e8m0>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
        } else {
            printf("%d ", (int)(sv[i]));
        }
    }
    printf("\n");
}
}






















