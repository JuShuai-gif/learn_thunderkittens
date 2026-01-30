/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rv_layout.cuh"


namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register vectors live.
 * 
 * 该命名空间包含与寄存器向量相关的概念（concepts）和抽象类型（abstract types）。
 */
namespace rv {
/**
 * @brief A dummy type used to identify register vectors.
 * 
 * 用于标识寄存器向量的虚拟类型。
 * 
 * 对于一个类型，如果它定义了 `ducks::rv::identifier` 作为其标识符，那么它就可以被当作寄存器向量类型来处理。
 */
struct identifier {};
/**
 * @brief Concept for all register vectors.
 * 
 * 用于定义所有寄存器向量的概念。
 * 
 * @tparam T The type to check against the concept requirements.
 *
 * 检查类型 `T` 是否符合寄存器向量的要求：
 * - 类型 `T` 必须具有一个嵌套类型 `identifier`，并且该类型必须与 `ducks::rv::identifier` 相同。
 */
template<typename T>
concept all = requires {
    typename T::identifier; // 检查类型 T 是否具有成员类型 identifier
} && std::is_same_v<typename T::identifier, identifier>; // 检查 T::identifier 是否与 ducks::rv::identifier 相同
// 各种布局类型概念定义
template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
// tile_layout 适用于与瓦片进行交互的向量布局
template<typename T> concept tile_layout  = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.
}
}

/**
 * @brief Register vector structure.
 * 
 * 寄存器向量结构体，用于在寄存器中表示一个向量。
 *
 * @tparam _T The packed data type used for the vector elements.
 *        向量元素使用的数据类型（打包后的数据类型）
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 *        瓦片的外部维度，单位为 TILE_DIM（通常为 16）
 * @tparam _inner_dim Controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *        控制瓦片布局的内部维度，决定了哪些轴在寄存器瓦片布局中进行映射。
 * 
 * 寄存器向量用于在瓦片中累积和映射值。如果需要，也可以直接对其进行计算，但它们并不是最优化的向量，因为它们有较大的重复和特殊的布局，
 * 以便与张量核心的寄存器布局高效配合。ThunderKittens 鼓励您尽量在瓦片级别进行计算！
 */
template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    using identifier = ducks::rv::identifier;  ///< rv 结构体的类型标识符，便于区分寄存器向量类型。
    static_assert(kittens::ducks::base_types::T1<_T>); // 确保传入的类型是支持的基本类型
    using layout = _layout;///< 使用的寄存器布局类型。
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;///< 判断布局是否为 naive 布局。
    // 提取打包类型的未打包和打包类型
    using T = kittens::base_types::packing<_T>::unpacked_type;///< 向量元素的未打包类型。
    using T2 = kittens::base_types::packing<_T>::packed_type; ///< 向量元素的打包类型。
    // 通过条件判断选择合适的数据类型
    using dtype = std::conditional_t<is_naive, T, T2>; ///< 根据布局类型选择数据类型，如果是 naive 布局则使用 T，否则使用 T2。

    static constexpr int length = _length;  ///< 向量的元素数量。
    static_assert(length % kittens::TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");

    // 确保元素数量是瓦片维度的整数倍。
    static constexpr int tiles  = _length / kittens::TILE_ROW_DIM<T>; ///< 计算瓦片数目，与 sv 类型保持一致。
    static constexpr int inner_dim = layout::inner_dim; ///< 子瓦片内部布局维度，通常为 1 或 2。
    static constexpr int outer_dim = is_naive ? (tiles+1)/2 : tiles; ///< 外部维度（也即瓦片数目），如果是 naive 布局，外维度是瓦片数的一半。
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
    #endif
    // 确保不使用不支持的 fp8 类型。

    dtype data[outer_dim][inner_dim]; ///< 实际存储寄存器向量数据的二维数组。
    // 对寄存器向量数据的访问操作符重载
    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    // 赋值操作符重载
    __device__ inline void operator=(const T &value) {
        dtype value2;
        if constexpr(is_naive) {
            value2 = value;
        } else {
            value2 = base_types::packing<T>::pack(value);// 使用打包类型转换
        }
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = value2;// 将值赋给寄存器向量的每个元素
            }
        }
    }
    // 将另一个寄存器向量赋值给当前寄存器向量
    template<typename U>
    __device__ inline void operator=(const rv<U, length, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < inner_dim; j++) {
                data[i][j] = base_types::convertor<T2, U2>::convert(other.data[i][j]);
            }
        }
    }
};
// 针对不同数据类型创建寄存器向量类型别名
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, layout>;



/* ----------  PRINT FUNCTION  ---------- */

/**
 * @brief Print the contents of a register vector as a formatted output.
 * 
 * This function prints register vectors with information about their dimensions
 * and data contents, handling both packed and unpacked data types.
 * 
 * @param vec The register vector to print
 */
template<ducks::rv::all RV>
__device__ void print(const RV &vec) {
    if (laneid() == 0) { // Only first thread in warp prints
        printf("Block %d, Warp %d: Register Vector %d (Type: %s, Layout: %s) - Distributed View:\n", 
               blockIdx.x, threadIdx.x / WARP_THREADS, RV::length,
               std::is_same_v<typename RV::T, float> ? "float" :
               std::is_same_v<typename RV::T, bf16> ? "bf16" :
               std::is_same_v<typename RV::T, half> ? "half" :
#if defined(KITTENS_BLACKWELL)
               std::is_same_v<typename RV::T, fp8e8m0> ? "fp8e8m0" :
#endif
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
               std::is_same_v<typename RV::T, fp8e4m3> ? "fp8e4m3" :
               std::is_same_v<typename RV::T, fp8e5m2> ? "fp8e5m2" : "unknown",
#endif
               RV::is_naive ? "naive" : "tile");
        printf("Each thread holds %dx%d elements\n", RV::outer_dim, RV::inner_dim);
        printf("\n");
    }
    __syncwarp();
    
    // Each thread prints its own data
    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);
            
            // Print the vector data this thread holds
            for (int i = 0; i < RV::outer_dim; i++) {
                printf("Outer[%d]: ", i);
                for (int j = 0; j < RV::inner_dim; j++) {
                    auto value = vec.data[i][j];
                    
                    if constexpr (std::is_same_v<typename RV::dtype, typename RV::T>) {
                        // Unpacked type, print directly
                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("%.3f ", value);
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {
                            printf("%.3f ", __half2float(value));
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {
                            printf("%.3f ", __bfloat162float(value));
#if defined(KITTENS_BLACKWELL)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {
                            printf("%.3f ", (float)value);
#endif
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {
                            printf("%.3f ", (float)value);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {
                            printf("%.3f ", (float)value);
#endif  
                        } else {
                            printf("%.3f ", (float)value);
                        }
                    } else {
                        // Packed type - check what type we're dealing with
                        if constexpr (std::is_same_v<typename RV::T, float>) {
                            printf("[%.3f, %.3f] ", value.x, value.y);
                        } else if constexpr (std::is_same_v<typename RV::T, bf16>) {
                            // Handle packed bf16_2 type
                            printf("[%.3f, %.3f] ", __bfloat162float(value.x), __bfloat162float(value.y));
                        } else if constexpr (std::is_same_v<typename RV::T, half>) {
                            // Handle packed half2 type
                            printf("[%.3f, %.3f] ", __half2float(value.x), __half2float(value.y));
#if defined(KITTENS_BLACKWELL)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e8m0>) {
                            // Handle packed fp8e8m0_4 types
                            __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
#endif  
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e4m3>) {
                            // Handle packed fp8e4m3_4 types  
                            __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<fp8e4m3_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                        } else if constexpr (std::is_same_v<typename RV::T, fp8e5m2>) {
                            // Handle packed fp8e5m2_4 types
                            __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<fp8e5m2_4*>(&value));
                            printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                   (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
#endif
                        } else {
                            // Other packed types - print the raw packed value
                            printf("0x%x ", *(uint32_t*)&value);
                        }
                    }
                }
                printf(" ");
            }
            printf("\n");
        }
        __syncwarp(); // Ensure threads print in order
    }
    
    if (laneid() == 0) {
        printf("\n");
    }
}

} // namespace kittens















