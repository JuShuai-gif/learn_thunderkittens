#pragma once

#include <cuda.h>
#include <assert.h>
#include <functional> // for std::hash
#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"

namespace kittens {
namespace detail {
namespace tma {

/* ----------   Create tile tensor map descriptor (HOST)  ---------- */

/**
 * @brief 创建一个张量映射（Tensor Map），用于描述源张量的内存布局。
 *
 * 该函数根据提供的源张量指针以及 ST 模板参数中指定的布局，创建一个张量映射（CUtensorMap）。
 * 张量映射用于描述张量在内存中的形状和布局。
 * 
 * @tparam ST 需要支持 TMA 的源张量类型。
 * @tparam axis 张量的第一个轴（0、1 或 2，默认为 2）。
 * @param tma_map 需要初始化的 CUtensorMap 对象指针。
 * @param src 源张量数据的指针（在全局内存中）。
 * @param batch 批次维度。
 * @param depth 深度维度。
 * @param rows 行维度。
 * @param cols 列维度。
 */
template<ducks::st::all ST, int axis>
__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, const typename ST::dtype *src, int batch, int depth, int rows, int cols
) {
    using dtype = typename ST::dtype;   // 获取张量的数据类型
    static_assert(axis==0 || axis==1 || axis==2, "axis must be 0, 1, or 2");    // 确保轴的值是 0、1 或 2
#ifdef DF_BLACKWELL
    // 在 Blackwell 架构下，检查 FP4 类型是否只能在轴 2 上使用
    static_assert(!(std::is_same_v<dtype, fp4e2m1_2> && axis != 2), "Axes 0 and 1 are not yet supported for FP4 type");
#endif

    constexpr uint32_t  tma_dim = ST::swizzle ? 5 : 4;  // 如果启用了 swizzle，则 TMA 维度为 5，否则为 4
    void *global_addr = (void*)(src);   // 获取全局内存中的数据指针
    
    // 根据 dtype 确定 TMA 数据类型
    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        std::is_same_v<dtype, fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
#ifdef DF_BLACKWELL
        std::is_same_v<dtype, fp8e8m0> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp4e2m1_2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
#endif
        CUtensorMapDataType(-1)
    );

    // 其他 TMA 设置（包括数据交错、L2 提升等）
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = ST::swizzle ? (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
        CU_TENSOR_MAP_SWIZZLE_NONE
    ) : CU_TENSOR_MAP_SWIZZLE_NONE;

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};  // 用于存储全局内存形状
    uint64_t gmem_stride[4] = {0, 0, 0, 0};     // 用于存储全局内存步长
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};  // 用于存储共享内存形状
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};  // 用于存储共享内存步长

    // 共享内存块的高度和宽度（在 swizzle 时使用）
    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    // TMA 期望全局和共享形状使用元素为单位
    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
    
    // 如果启用了 swizzle，则设置全局内存和共享内存形状
    if constexpr (ST::swizzle) {
        if constexpr (axis == 2) {  // 如果是轴 2
            gmem_shape[0] = swizzle_elements;   // 上取整，处理跨越的情况
            gmem_shape[1] = (uint64_t)rows; 
            gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
            gmem_shape[3] = (uint64_t)depth;
            gmem_shape[4] = (uint64_t)batch;
    
            gmem_stride[0] = (uint64_t)cols * sizeof(dtype); // 2 FP4 elements per col, but sizeof(fp4) = 0.5, so these cancel out
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)rows * cols * sizeof(dtype); // see above
            gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype); // see above
        }
        else if constexpr (axis == 1) {
            gmem_shape[0] = swizzle_elements;
            gmem_shape[1] = (uint64_t)depth;
            gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
            gmem_shape[3] = (uint64_t)rows;
            gmem_shape[4] = (uint64_t)batch;
    
            gmem_stride[0] = (uint64_t)rows * cols * sizeof(dtype);
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
            gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);
    
        }
        else {
            gmem_shape[0] = swizzle_elements;
            gmem_shape[1] = (uint64_t)batch;
            gmem_shape[2] = (uint64_t)(cols+swizzle_elements-1) / swizzle_elements; // round up, note this can potentially screw up out of bounds access handling :/
            gmem_shape[3] = (uint64_t)rows;
            gmem_shape[4] = (uint64_t)depth;
    
            gmem_stride[0] = (uint64_t)depth * rows * cols * sizeof(dtype);
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
            gmem_stride[3] = (uint64_t)rows * cols * sizeof(dtype);
        }

        // 设置共享内存形状
        smem_shape[0] = swizzle_elements;
        smem_shape[1] = shared_tile_height;
        smem_shape[2] = shared_tile_width / swizzle_elements;
        smem_shape[3] = 1;
        smem_shape[4] = 1;
    } else {    // 如果没有启用 swizzle
        static_assert(axis == 2, "For non-swizzled tiles, only axis 2 is supported.");

        gmem_shape[0] = (uint64_t)cols;
        gmem_shape[1] = (uint64_t)rows;
        gmem_shape[2] = (uint64_t)depth;
        gmem_shape[3] = (uint64_t)batch;

        gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[1] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[2] = (uint64_t)depth * rows * cols * sizeof(dtype);

        smem_shape[0] = shared_tile_width;
        smem_shape[1] = shared_tile_height;
        smem_shape[2] = 1;
        smem_shape[3] = 1;
    }

    // 确保全局地址是 16 字节对齐的
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    // 检查全局内存步长是否为 16 的倍数
    assert(gmem_stride[0] % 16 == 0); // gmem_stride[0] elements must be a multiple of 16B
    assert(gmem_stride[1] % 16 == 0); // gmem_stride[1] elements must be a multiple of 16B
    assert(gmem_stride[2] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B
    assert(gmem_stride[3] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B
    // 检查共享内存的维度是否在合法范围内
    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256
    assert(smem_shape[1] <= 256); // smem_shape[1] elements must be <= 256
    assert(smem_shape[2] <= 256); // smem_shape[2] elements must be <= 256
    
    // 检查共享内存步长是否满足要求
    assert((smem_shape[0]*sizeof(dtype)) % 16 == 0); // if wgmma_interleave is none, then smem_shape[0] * sizeof(dtype) must be a multiple of 16B
    
    // 如果没有使用交错，并且使用了 swizzle，检查共享内存的大小是否满足条件
    assert(smem_stride[0] <= 8); // smem_stride[0] must be less <= 8
    assert(smem_stride[1] <= 8); // smem_stride[1] must be less <= 8
    assert(smem_stride[2] <= 8); // smem_stride[2] must be less <= 8
    assert(smem_stride[3] <= 8); // smem_stride[3] must be less <= 8
    assert(smem_stride[4] <= 8); // smem_stride[3] must be less <= 8

    assert(smem_stride[0] == 1); // smem_stride[0] is ignored when wgmma_interleave is none

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }
    // 创建张量映射并处理错误
    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        // 如果创建失败，抛出异常并打印详细错误信息
        std::string error_msg = format_tma_error(
            "tile", error_string,
            batch, depth, rows, cols,
            tma_map, tma_format, tma_dim, global_addr,
            gmem_shape_ptr, gmem_stride_ptr,
            smem_shape_ptr, smem_stride_ptr,
            5, 4, 5, 5,
            tma_interleave, tma_swizzle, tma_l2Promotion, tma_oobFill,
            "ST::rows: " + std::to_string(ST::rows) + "\n  ST::cols: " + std::to_string(ST::cols)
        );
        throw std::runtime_error(error_msg);
    }
}


/**
* @brief 在 GPU 上分配并初始化给定源张量的张量映射。
*
* 该函数创建一个张量映射（CUtensorMap）用于指定的源共享向量类型。张量映射用于描述张量在内存中的形状和布局。
* 函数根据提供的源张量指针和 SV 模板参数指定的布局，设置张量映射。
*
* @tparam SV 源张量类型，必须支持 TMA。
* @param src 指向源张量数据的指针（位于全局内存中）。
* @returns 指向初始化后的 CUtensorMap 对象的指针。
*/
template<ducks::st::all ST>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename ST::dtype *src, int batch, int depth, int rows, int cols) {
    CUtensorMap *tma_map_d;
    // 在 GPU 上分配一个 CUtensorMap 对象的内存空间
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // 在栈上创建一个临时的张量映射
    // 使用 create_tensor_map 函数创建一个张量映射
    create_tensor_map<ST>(&tma_map_host, src, batch, depth, rows, cols);
    // 将主机端的张量映射数据复制到 GPU 上
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    // 返回指向 GPU 上张量映射的指针
    return tma_map_d;
}

/* ----------   Create vector tensor map descriptor (HOST)  ---------- */

// First, we need a template system to determine how to divide up a long shared vector into multiple subvectors.
// We have to do this because the first dimension for TMA is limited to 256 elements.
// Our goal is to find the largest multiple of 16 that is <= 256 and divides the vector length evenly.

template<typename SV, int D=16> struct find_vector_divider {
    static constexpr int value = (SV::length % (16*D) == 0 && (SV::length < 256 || ((16*D)*sizeof(typename SV::dtype)) % 128 == 0)) ?
        16*D : find_vector_divider<SV, D-1>::value;
};
template<typename SV> struct find_vector_divider<SV, 1> { static constexpr int value = 16; }; // base case
template<typename SV> constexpr int sv_tma_dim1 = find_vector_divider<SV>::value; // inner dim
template<typename SV> constexpr int sv_tma_dim2 = (SV::length / sv_tma_dim1<SV>);

/* ----------   创建向量张量映射描述符（HOST）  ---------- */

/**
* @brief 为给定的源向量创建张量映射。
*
* 该函数为指定的源共享向量类型创建一个张量映射（CUtensorMap）。张量映射用于描述张量在内存中的形状和布局。
* 函数根据提供的源张量指针和 SV 模板参数指定的布局，设置张量映射。
*
* @tparam SV 源张量类型，必须支持 TMA。
* @tparam axis 第一个轴（0、1 或 2；默认为 2）
* @param tma_map 指向 CUtensorMap 对象的指针，待初始化。
* @param src 指向源张量数据的指针（位于全局内存中）。
*/
template<ducks::sv::all SV, int axis>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    using dtype = typename SV::dtype;
    static_assert(axis == -1, "for vector TMA, row axis must be -1 as it's unused");
    static_assert(SV::length <= 256 || (SV::length*sizeof(dtype)) % 128 == 0);// 确保长度小于等于256或符合 128 的倍数
    // There is technically a way around ^ that involves instantiating two separate TMA descriptors, one of size 256
    // and the other of size %256, but this is a fairly mild restriction and the other approach is a real PITA and incurs other costs.
    
    constexpr uint32_t  tma_dim     = 4; // 张量映射维度，设置为 4
    void               *global_addr = (void*)(src);// 获取源张量数据的全局内存地址
    // 根据 dtype 类型选择张量映射的数据类型
    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        std::is_same_v<dtype, fp8e4m3> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp8e5m2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
#ifdef DF_BLACKWELL
        std::is_same_v<dtype, fp8e8m0> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
        std::is_same_v<dtype, fp4e2m1_2> ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
#endif// 如果类型不匹配，设置为无效类型
        CUtensorMapDataType(-1) // 如果类型不匹配，设置为无效类型
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      swizzle         = CU_TENSOR_MAP_SWIZZLE_NONE;

    constexpr uint64_t dim1 = sv_tma_dim1<SV>; // 内部维度
    // constexpr uint64_t dim2 = sv_tma_dim2<SV>; outer dim, not used here.

    uint64_t gmem_shape [4] = {(uint64_t)cols, (uint64_t)rows, (uint64_t)depth, (uint64_t)batch};
    uint64_t gmem_stride[3] = {(uint64_t)cols*sizeof(dtype), (uint64_t)cols*rows*sizeof(dtype), (uint64_t)cols*rows*depth*sizeof(dtype)};
    uint32_t smem_shape [4] = {(uint32_t)dim1, 1, 1, 1};// 共享内存形状
    uint32_t smem_stride[4] = {1, 1, 1, 1};// 共享内存步长

    // 确保全局地址始终是 16 字节对齐
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(smem_shape[0] <= 256); // 确保共享内存的第一个维度小于等于 256
    // 调用 CUDA API 创建张量映射
    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        swizzle,
        tma_l2Promotion,
        tma_oobFill
    );
    // 错误处理：如果创建张量映射失败，则抛出异常并显示错误信息
    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::string error_msg = format_tma_error(
            "vector", error_string,
            batch, depth, rows, cols,
            tma_map, tma_format, tma_dim, global_addr,
            gmem_shape_ptr, gmem_stride_ptr,
            smem_shape_ptr, smem_stride_ptr,
            4, 3, 4, 4,
            tma_interleave, swizzle, tma_l2Promotion, tma_oobFill,
            "SV::length: " + std::to_string(SV::length)
        );
        throw std::runtime_error(error_msg);
    }
};

/**
* @brief 在 GPU 上分配内存并初始化给定源张量的张量映射。
*
* 该函数创建一个张量映射（CUtensorMap）用于指定的源共享向量类型。张量映射用于描述张量在内存中的形状和布局。
* 函数根据提供的源张量指针和 SV 模板参数指定的布局，设置张量映射。
*
* @tparam SV 源张量类型，必须支持 TMA。
* @param src 指向源张量数据的指针（位于全局内存中）。
* @param batch 张量的批量维度大小。
* @param depth 张量的深度维度大小。
* @param rows 张量的行维度大小。
* @param cols 张量的列维度大小。
* @returns 指向初始化后的 CUtensorMap 对象的指针。
*/
template<ducks::sv::all SV>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename SV::dtype *src, int batch, int depth, int rows, int cols) {
    // 声明一个指针，用于在 GPU 上分配张量映射内存
    CUtensorMap *tma_map_d;
    // 在 GPU 上分配内存来存储一个 CUtensorMap 对象
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    // 在主机上声明一个 CUtensorMap 对象（用于设置并初始化张量映射）
    CUtensorMap tma_map_host; // 将张量映射对象放在栈上，避免不必要的堆分配
    // 调用 create_tensor_map 函数在主机上初始化张量映射
    // 传入参数：tma_map_host（张量映射对象的主机端副本）、源张量数据指针、批量维度、深度维度、行维度和列维度
    create_tensor_map<SV>(&tma_map_host, src, batch, depth, rows, cols);
    // 将主机端的张量映射数据复制到 GPU 内存中
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    // 返回指向 GPU 上张量映射的指针
    return tma_map_d;
}

} // namespace tma
} // namespace detail
} // namespace kittens