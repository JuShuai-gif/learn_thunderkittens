/**
 * @file
 * @brief 用于寄存器tile的warp操作的聚合头文件
 * 
 * 这个文件包含了所有用于warp级寄存器tile操作的头文件聚合。
 * 它整合了类型转换、映射操作和归约操作，并提供了NaN检测等实用功能。
 */

#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

/**
 * @brief 检查tile中是否存在NaN（非数字）值
 *
 * @tparam RT tile类型，必须满足ducks::rt::all概念
 * @param src[in] 要检查的源tile
 * @return bool 如果任何线程检测到NaN则返回true，否则返回false
 * 
 * 这个函数检查tile中的所有元素，检测是否存在NaN值。
 * 它支持多种浮点类型：float、bfloat16和half。
 * 使用warp级别的ballot操作在所有线程间传播检测结果。
 */
template<ducks::rt::all RT>
__device__ static inline bool hasnan(const RT &src) {
    KITTENS_CHECK_WARP  // 检查warp配置的宏
    
    bool nan_detected = false;  // 本地线程的NaN检测标志
    
    // 遍历tile中的所有元素
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {          // 遍历行
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {       // 遍历列
            #pragma unroll
            for(int k = 0; k < RT::packed_per_tile; k++) {  // 遍历每个打包数据
                // 根据数据类型进行不同的NaN检查
                if constexpr (std::is_same_v<typename RT::T, float>) {
                    // 检查float类型的NaN
                    if(isnan(src.tiles[i][j].data[k].x) || isnan(src.tiles[i][j].data[k].y)) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                    // 检查bfloat16类型的NaN：先转换为float再检查
                    if(isnan(__bfloat162float(src.tiles[i][j].data[k].x)) || isnan(__bfloat162float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RT::T, half>) {
                    // 检查half类型的NaN：先转换为float再检查
                    if(isnan(__half2float(src.tiles[i][j].data[k].x)) || isnan(__half2float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                }
                else {
                    // 静态断言：不支持的数据类型
                    static_assert(sizeof(typename RT::T) == 999, "Unsupported dtype");
                }
            }
        }
    }
    
    // 使用warp ballot操作在所有线程间传播检测结果
    // 如果任何线程检测到NaN，则返回true
    // __ballot_sync返回一个掩码，其中每个线程的位表示该线程的nan_detected值
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}


#include "complex_conversions.cuh"
#include "complex_maps.cuh"