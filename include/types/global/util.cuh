#pragma once

#include <type_traits>
#include <cstddef>
#include <iostream>
#include <sstream>

namespace kittens {
namespace ducks {
namespace gl {

// 编译时维度（compile-time dimension）概念，用于表示静态维度
template<int d> concept cdim = (d > 0); // 只要维度 d 大于 0，就认为是有效的编译时维度

// 运行时维度（runtime dimension）概念，用于表示动态维度
template<int d> concept rdim = (d == -1); // 如果维度 d 是 -1，就认为是有效的运行时维度

// 编译时维度的结构体
template<int _v> struct compiled_dim {
    // 确保维度值是合法的（大于 0）
    static_assert(cdim<_v>, "Invalid compile-time dimension value");
    // 静态常量 v，表示维度值
    static constexpr size_t v = _v;

    // CUDA 核函数中使用的构造函数，接受一个 nullptr_t 参数
    __host__ __device__ inline compiled_dim(const std::nullptr_t &_) {}

    // 重载类型转换运算符，将 compiled_dim 转换为 size_t 类型
    __host__ __device__ inline constexpr operator size_t() const { return v; }
};

// 运行时维度的结构体
struct runtime_dim {
    size_t v;   // 存储维度值
    // 构造函数，初始化运行时维度值
    __host__ __device__ inline runtime_dim(const size_t &_v) : v(_v) {}
    // 重载类型转换运算符，将 runtime_dim 转换为 size_t 类型
    __host__ __device__ inline operator size_t() const { return v; }
};

// 根据模板参数 d，选择合适的维度类型：如果是运行时维度，则使用 runtime_dim；如果是编译时维度，则使用 compiled_dim
template<int d> using make_dim_t = std::conditional_t<rdim<d>, runtime_dim, compiled_dim<d>>;

// 根据模板参数 d，选择合适的参数类型：如果是运行时维度，则使用 size_t；如果是编译时维度，则使用 nullptr_t
template<int d> using make_arg_t = std::conditional_t<rdim<d>, size_t, std::nullptr_t>; // we pass runtime dims as size_t, comptime dims as nullptr_t
}
}

namespace detail {
// 用于匹配不同类型的概念检查
template<typename T> concept tile = ducks::st::all<T> || ducks::rt::all<T> || ducks::cst::all<T> || ducks::crt::all<T>;
// 用于匹配向量类型的概念检查
template<typename T> concept vec  = ducks::sv::all<T> || ducks::rv::all<T> || ducks::csv::all<T> || ducks::crv::all<T>;

// 仅在宏 KITTENS_HOPPER 或 KITTENS_BLACKWELL 被定义时才会编译
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
namespace tma {

// 定义一个静态内联函数，用于格式化TMA错误信息    
__host__ static inline std::string format_tma_error(
    const char* error_type,     // 错误类型
    const char* error_string,   // 错误信息
    int batch, int depth, int rows, int cols,   // 与TMA相关的维度信息
    CUtensorMap* tma_map,       // 指向TMA映射的指针
    CUtensorMapDataType tma_format, // TMA格式
    uint32_t tma_dim,               // TMA维度
    void* global_addr,              // 全局地址指针
    const uint64_t* gmem_shape,     // 全局内存形状
    const uint64_t* gmem_stride,    // 全局内存步长
    const uint32_t* smem_shape,     // 共享内存形状
    const uint32_t* smem_stride,    // 共享内存步长
    size_t gmem_shape_size,         // 全局内存形状大小
    size_t gmem_stride_size,        // 全局内存步长大小
    size_t smem_shape_size,         // 共享内存形状大小
    size_t smem_stride_size,        // 共享内存步长大小
    CUtensorMapInterleave tma_interleave,   // TMA交错方式
    CUtensorMapSwizzle tma_swizzle,         // TMA置换方式
    CUtensorMapL2promotion tma_l2Promotion, // TMA L2提升标志
    CUtensorMapFloatOOBfill tma_oobFill,    // TMA超出范围填充方式
    const std::string& extra_info = ""      // 可选的额外信息
) {
    std::ostringstream oss; // 用于构建错误信息的输出流
    oss << "Error in " << error_type << " TMA descriptor creation: ";   // 错误信息的开始部分
    oss << (error_string ? error_string : "Unknown CUDA error");        // 如果错误信息存在，则输出，否则输出“未知CUDA错误”
    oss << "\nParameters:"; // 显示参数信息
    oss << "\n  batch: " << batch;
    oss << "\n  depth: " << depth;
    oss << "\n  rows: " << rows;
    oss << "\n  cols: " << cols;

    // 如果存在额外的信息，输出
    if (!extra_info.empty()) oss << "\n  " << extra_info;

    // 显示TMA映射的相关参数
    oss << "\ncuTensorMapEncodeTiled arguments:";
    oss << "\n  tma_map: " << reinterpret_cast<uintptr_t>(tma_map); // 显示TMA映射的地址
    oss << "\n  tma_format: " << tma_format;
    oss << "\n  tma_dim: " << tma_dim;
    oss << "\n  global_addr: " << reinterpret_cast<uintptr_t>(global_addr); // 显示全局地址的地址

    // 获取全局内存的属性信息
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, global_addr);   // 获取内存属性
    if (err == cudaSuccess) {   // 如果成功获取属性
        oss << "\n  global_addr memory type: ";
        if (attributes.type == cudaMemoryTypeDevice) oss << "valid device memory";  // 设备内存
        else if (attributes.type == cudaMemoryTypeHost) oss << "host memory (invalid for TMA)"; // 主机内存
        else if (attributes.type == cudaMemoryTypeManaged) oss << "managed memory"; // 管理内存
        else oss << "unknown memory type";  // 管理内存
    } else {
        oss << "\n  global_addr memory type: unable to determine (error: " << cudaGetErrorString(err) << ")";
    }
    
    // 显示全局内存形状、步长
    oss << "\n  gmem_shape: " << reinterpret_cast<uintptr_t>(gmem_shape) << " [";
    for (size_t i = 0; i < gmem_shape_size; ++i) oss << gmem_shape[i] << (i < gmem_shape_size - 1 ? ", " : "");
    oss << "]";
    oss << "\n  gmem_stride: " << reinterpret_cast<uintptr_t>(gmem_stride) << " [";
    for (size_t i = 0; i < gmem_stride_size; ++i) oss << gmem_stride[i] << (i < gmem_stride_size - 1 ? ", " : "");
    oss << "]";

    // 显示共享内存形状、步长    
    oss << "\n  smem_shape: " << reinterpret_cast<uintptr_t>(smem_shape) << " [";
    for (size_t i = 0; i < smem_shape_size; ++i) oss << smem_shape[i] << (i < smem_shape_size - 1 ? ", " : "");
    oss << "]";
    oss << "\n  smem_stride: " << reinterpret_cast<uintptr_t>(smem_stride) << " [";
    for (size_t i = 0; i < smem_stride_size; ++i) oss << smem_stride[i] << (i < smem_stride_size - 1 ? ", " : "");
    oss << "]";

    // 显示TMA的交错方式、置换方式、L2提升方式和超出范围填充方式
    oss << "\n  tma_interleave: " << tma_interleave;
    oss << "\n  tma_swizzle: " << tma_swizzle;
    oss << "\n  tma_l2Promotion: " << tma_l2Promotion;
    oss << "\n  tma_oobFill: " << tma_oobFill;
    
    // 返回构建好的错误信息字符串
    return oss.str();
}

} // namespace tma
#endif // defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
} // namespace detail

// 下面是坐标相关的代码
namespace ducks {
namespace coord {

// 定义一个名为 identifier 的空结构体，用于标识坐标类型
struct identifier {};
}
}

// 定义一个模板结构体 coord，用于表示张量坐标
template<typename _T=ducks::default_type> struct coord { // essentially a named int4 for tensor coordinates.
    using identifier = ducks::coord::identifier;    // 坐标类型的标识符
    using BASE = _T; // 坐标基础类型，单位类型
    static_assert(std::is_same_v<BASE, ducks::default_type> || detail::tile<BASE> || detail::vec<BASE>); // ensure BASE is a valid type
    int b, d, r, c; // 坐标的四个维度：batch、depth、rows、columns

    // 构造函数
    __device__ inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    __device__ inline coord(        int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {} // 默认batch为0
    __device__ inline coord(                int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {} // 默认depth和batch为0
    __device__ inline coord(                        int _c) : b( 0), d( 0), r( 0), c(_c) {} // 默认坐标为(0, 0, 0, 0)
    __device__ inline coord(                              ) : b( 0), d( 0), r( 0), c( 0) {} // 拷贝构造函数
    template<typename U> __device__ inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    __device__ inline coord(const int4 &other)  : b(other.x), d(other.y), r(other.z), c(other.w) {} // 从int4转换构造

    // 将coord转换为int4
    __device__ inline operator int4() const { return int4(b, d, r, c); }

    // 将坐标转换为单位坐标，支持不同的行列轴
    template<int row_axis, int col_axis> __device__ inline coord<ducks::default_type> unit_coord() const {
        if constexpr (detail::tile<BASE>) {// 如果是tile类型的坐标
            static_assert(row_axis != col_axis, "row and column axes must be different");
            static_assert(row_axis >= 0 && row_axis <= 3, "row axis must be between 0 and 3");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(
                row_axis == 0 ? b*BASE::rows : b,
                row_axis == 1 ? d*BASE::rows : d,
                row_axis == 2 ? r*BASE::rows : r,
                c*BASE::cols
            );
        }
        else if constexpr (detail::vec<BASE>) { // 如果是vec类型的坐标
            static_assert(row_axis == -1, "row axis must be be -1 for a vector coordinate to be converted to a unit coordinate");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(b, d, r, c*BASE::length);
        }
        else {  // 默认返回原始坐标
            return coord<ducks::default_type>(*this);
        }
    }
    // 获取指定轴的维度
    template<int axis> __device__ inline int dim() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be between 0 and 3");
        if constexpr      (axis == 0) { return b; }
        else if constexpr (axis == 1) { return d; }
        else if constexpr (axis == 2) { return r; }
        else                          { return c; }
    }
};

// 定义用于验证坐标类型的概念
namespace ducks {
namespace coord {
/**
* @brief 检查类型T是否是有效的坐标类型
* @tparam T 需要检查的类型
*
* 要求：
* - T有一个嵌套类型identifier，并且该类型必须是ducks::coord::identifier
*/
template<typename T> concept all = requires {
    typename T::identifier; // 检查T是否有identifier类型
} && std::is_same_v<typename T::identifier, identifier>; // 检查T::identifier是否等于ducks::coord::identifier
template<typename T> concept tile = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::tile<typename T::BASE>);    // 确保T是tile类型
template<typename T> concept vec  = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::vec<typename T::BASE>);     // 确保T是vector类型
}
}
}