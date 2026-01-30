/**
 * @file
 * @brief The main ThunderKittens register tile struct, where most computation happens.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rv.cuh"

namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief 用于寄存器瓷砖的概念和抽象类型的命名空间。
 */
namespace rt {
/**
 * @brief 一个虚拟类型，用于标识寄存器瓷砖（rt）。
 * 
 * 对于一个类型要能像 rt 一样工作，它应该定义其标识符为 ducks::rt::identifier。
 * 如果一个类型包含 ducks::rt::identifier 标识符，它就会在编译器检查中被视为一个寄存器瓷砖类型。
 */
struct identifier {};

/**
 * @brief 一个概念，用于检查类型是否是寄存器瓷砖类型。
 * 
 * @tparam T 需要检查的类型
 *
 * 要求：
 * - T 必须有一个嵌套类型 identifier。
 * - T::identifier 必须等于 ducks::rt::identifier。
 */
template<typename T> concept all = requires {
    typename T::identifier; // 检查 T 是否有 identifier 类型
} && std::is_same_v<typename T::identifier, identifier>; // 检查 T::identifier 是否是 ducks::rt::identifier
/**
 * @brief 用于检查寄存器瓷砖是否具有行布局的概念。
 * 
 * @tparam T 需要检查的类型
 *
 * 要求：
 * - T 是一个寄存器瓷砖类型。
 * - T 必须具有内部布局类型 ducks::rt_layout::row。
 */
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
 * @brief 用于检查寄存器瓷砖是否具有列布局的概念。
 * 
 * @tparam T 需要检查的类型
 *
 * 要求：
 * - T 是一个寄存器瓷砖类型。
 * - T 必须具有内部布局类型 ducks::rt_layout::col。
 */
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt
} // namespace ducks

/**
 * @brief 用于在寄存器中操作数据的主瓷砖结构体。
 *
 * @tparam T2 矩阵元素的打包数据类型。
 * @tparam _height 瓷砖的高度，按子瓷砖数量来划分。
 * @tparam _width 瓷砖的宽度，按子瓷砖数量来划分。
 * @tparam _layout 内部基础瓷砖的布局，行优先（row-major）或列优先（column-major）。
 *
 * 这个结构体旨在灵活地处理矩阵瓷砖，允许对由较小子瓷砖组成的瓷砖进行操作。它支持行优先和列优先布局，并包含类型推断的辅助结构。
 * 
 * 通常，你可能希望使用行优先布局，除非你特别想调用矩阵乘法（mma）。
 */
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< 标识符类型，用于标识 rt 结构体类型。
    using layout = _layout; ///< 矩阵瓷砖的布局类型。
    // 确保 _T 是支持的类型
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    // 类型推断，T 表示未打包的元素类型，T2 表示打包后的数据类型
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< 矩阵元素的数据类型

    // 计算瓷砖的行数和列数，并确保它们可以被基础瓷砖的大小整除
    static constexpr int rows                = _rows; ///< Total number of rows.
    static_assert(rows % rt_base<T, layout>::tile_size_row == 0, "Rows must be divisible by the tile size");
    
    static constexpr int cols                = _cols; ///< Total number of columns.
    static_assert(cols % rt_base<T, layout>::tile_size_col == 0, "Columns must be divisible by the tile size");
    
    // 计算瓷砖的高度和宽度
    static constexpr int height              = rows / rt_base<T, layout>::tile_size_row; ///< Height in subtiles.
    static constexpr int width               = cols / rt_base<T, layout>::tile_size_col; ///< Width in subtiles.
    
    // 基础瓷砖的大小
    static constexpr int tile_size_row        = rt_base<T, layout>::tile_size_row;        ///< Size of the base tile.
    static constexpr int tile_size_col        = rt_base<T, layout>::tile_size_col;        ///< Size of the base tile.
    
    // 计算总元素数、每个线程处理的元素数和每个线程处理的打包元素数
    static constexpr int num_elements        = rt_base<T, layout>::num_elements        * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<T, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread   = rt_base<T, layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile     = rt_base<T, layout>::packed_per_thread; ///< Packed elements per tile.
    
    // 定义用于存储矩阵数据的基础瓷砖数组，按子瓷砖划分
    rt_base<T, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.
    
    // 定义行向量和列向量类型，用于表示瓷砖中的向量
    using row_vec = rv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = rv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
    
    // 将一个常数值赋值给矩阵
    __device__ inline void operator=(const T &value) {
        T2 value2 = base_types::packing<T>::pack(value);// 将值打包
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = value2;// 将打包的值赋给每个子瓷砖
                }
            }
        }
    }

    // 将另一个矩阵的值赋给当前矩阵
    template<typename U>
    __device__ inline void operator=(const rt<U, rows, cols, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(other.tiles[i][j].data[k]);
                }
            }
        }
    }
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers
// 为 float 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
// 为 bf16 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16,  _r, _c, layout>;
// 为 half 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_hf = rt<half,  _r, _c, layout>;
// 如果启用了 KITTENS_HOPPER 或 KITTENS_BLACKWELL，定义额外的类型别名
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
// 为 fp8e4m3 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e4m3 = rt<fp8e4m3,  _r, _c, layout>;
// 为 fp8e5m2 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e5m2 = rt<fp8e5m2,  _r, _c, layout>;
#endif

// 如果启用了 KITTENS_BLACKWELL，定义额外的类型别名
#if defined(DF_BLACKWELL)
// 为 fp8e8m0 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e8m0 = rt<fp8e8m0,  _r, _c, layout>;
// 为 fp4e2m1_2 类型的矩阵定义寄存器瓷砖类型别名
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp4e2m1_2 = rt<fp4e2m1_2,  _r, _c, layout>;
#endif

/* ----------  PRINTOUTS  ---------- */

/**
 * @brief 获取寄存器瓷砖类型的可读名称。
 * 
 * @tparam T 寄存器瓷砖元素类型（如 float, bf16, half 等）。
 * @tparam rows 瓷砖的行数。
 * @tparam cols 瓷砖的列数。
 * 
 * @return 返回可读的类型名称字符串。
 */
template<typename T, int rows, int cols>
__device__ constexpr const char* get_rt_type_name() {
    // 根据类型 T 的不同，返回相应的寄存器瓷砖名称
    if constexpr (std::is_same_v<T, float>) {
        return "rt_fl";// 返回 float 类型的寄存器瓷砖名称
    } else if constexpr (std::is_same_v<T, half>) {
        return "rt_hf";// 返回 half 类型的寄存器瓷砖名称
    } else if constexpr (std::is_same_v<T, bf16>) {
        return "rt_bf";// 返回 bf16 类型的寄存器瓷砖名称
#if defined(DF_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp4e2m1_2>) {
        return "rt_fp4_e2m1_2";// 返回 fp4e2m1_2 类型的寄存器瓷砖名称
    } else if constexpr (std::is_same_v<T, fp8e8m0>) {
        return "rt_fp8_e8m0";// 返回 fp8e8m0 类型的寄存器瓷砖名称
#endif
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp8e4m3>) {
        return "rt_fp8_e4m3"; // 返回 fp8e4m3 类型的寄存器瓷砖名称
    } else if constexpr (std::is_same_v<T, fp8e5m2>) {
        return "rt_fp8_e5m2";// 返回 fp8e5m2 类型的寄存器瓷砖名称
#endif
    } else {
        return "rt_unknown";
    }
}

/**
 * @brief Print the contents of a register tile as a formatted table.
 * 
 * This function should be called by all threads in the warp, but only
 * the first thread (laneid() == 0) will coordinate the printing.
 * It shows what each thread holds in its portion of the distributed tile.
 * 
 * @param tile The register tile to print
 */
template<ducks::rt::all RT>
__device__ inline void print(const RT& tile) {
    if (laneid() == 0) { // Only first thread in warp prints
        printf("Block %d, Warp %d: Register Tile %dx%d (Type: %s<%d,%d>) - Distributed View:\n", 
               blockIdx.x, threadIdx.x / WARP_THREADS, RT::rows, RT::cols, 
               get_rt_type_name<typename RT::T, RT::rows, RT::cols>(), RT::rows, RT::cols);
        printf("Each thread holds %d elements (%d packed)\n", 
               RT::elements_per_thread, RT::packed_per_thread);
        printf("\n");
    }
    __syncwarp();
    
    // Each thread prints its own data
    for (int tid = 0; tid < WARP_THREADS; tid++) {
        if (laneid() == tid) {
            printf("Thread %2d: ", tid);
            
            // Print the packed data this thread holds
            for (int i = 0; i < RT::height; i++) {
                for (int j = 0; j < RT::width; j++) {
                    printf("Subtile[%d][%d]: ", i, j);
                    for (int k = 0; k < RT::packed_per_tile && k < 4; k++) { // Limit to first 4 elements to avoid too much output
                        auto packed_val = tile.tiles[i][j].data[k];
                        
                        if constexpr (std::is_same_v<typename RT::dtype, typename RT::T>) {
                            // Unpacked type, print directly
                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("%.3f ", packed_val);
                            } else if constexpr (std::is_same_v<typename RT::T, half>) {
                                printf("%.3f ", __half2float(packed_val));
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                                printf("%.3f ", __bfloat162float(packed_val));
#if defined(KITTENS_BLACKWELL)
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {
                                printf("%.3f ", (float)packed_val);
#endif
                            } else {
                                printf("%.3f ", (float)packed_val);
                            }
                        } else {
                            // Packed type - check what type we're dealing with
                            if constexpr (std::is_same_v<typename RT::T, float>) {
                                printf("[%.3f, %.3f] ", packed_val.x, packed_val.y);
                            } else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                                // Handle packed bf16_2 type
                                printf("[%.3f, %.3f] ", __bfloat162float(packed_val.x), __bfloat162float(packed_val.y));
#if defined(KITTENS_BLACKWELL)
                            } else if constexpr (std::is_same_v<typename RT::T, fp8e8m0>) {
                                // Extract the 4 individual fp8e8m0 values from the packed fp8e8m0_4
                                __nv_fp8_e8m0 *vals = reinterpret_cast<__nv_fp8_e8m0*>(const_cast<fp8e8m0_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ", 
                                       (float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                            } else if constexpr (std::is_same_v<typename RT::T, fp4e2m1>) {
                                // Handle packed fp4e2m1_4 types (4 fp4 values packed together)
                                uint8_t *vals = reinterpret_cast<uint8_t*>(const_cast<fp4e2m1_4*>(&packed_val));
                                printf("[%.3f,%.3f,%.3f,%.3f] ", (float)fp4e2m1(vals[0] & 0xF), (float)fp4e2m1((vals[0] >> 4) & 0xF), (float)fp4e2m1(vals[1] & 0xF), (float)fp4e2m1((vals[1] >> 4) & 0xF));
#endif
                            } else {
                                // Other packed types - print the raw packed value
                                printf("0x%x ", *(uint32_t*)&packed_val);
                            }
                        }
                    }
                    if (RT::packed_per_tile > 4) printf("... ");
                }
            }
            printf("\n");
        }
        __syncwarp(); // Ensure threads print in order
    }
    
    if (laneid() == 0) {
        printf("\n");
    }
    __syncwarp();
}

} // namespace kittens


















