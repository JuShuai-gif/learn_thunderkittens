/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include <utility>
#include "tma.cuh"
#endif

namespace kittens {


/**
 * @brief 表示全局内存布局的维度
 * 
 * 该结构体定义了全局布局中的四个常见轴：
 * - BATCH：批次维度（通常用于深度学习中的批处理）
 * - DEPTH：深度维度
 * - ROW：行维度
 * - COL：列维度
 * 
 * 这些常量值在全局布局中用于指示不同维度的索引。
 */
struct dim {
    static constexpr int BATCH = 0;     // 批次维度的索引
    static constexpr int DEPTH = 1;     // 深度维度的索引
    static constexpr int ROW   = 2;     // 行维度的索引
    static constexpr int COL   = 3;     // 列维度的索引
};

/* ----------   Associative dictionary for global layouts  ---------- */

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
namespace ducks {
namespace tma {
namespace descriptor {

/**
 * @brief 空标识符结构，用于 TMA 描述符的标识
 * 
 * 该标识符用于区别不同类型的描述符。在模板编程中，这个结构体作为标识符使用。
 */
struct identifier {};

/**
 * @brief 概念（Concept）用于检查所有复杂全局内存布局类型
 * 
 * 该概念用于检查模板参数 T 是否符合特定的要求：
 * - 必须包含一个名为 `identifier` 的嵌套类型。
 * - `identifier` 类型必须与 `ducks::tma::descriptor::identifier` 相同。
 */
template<typename T> concept all = requires {
    typename T::identifier;     // 检查类型 T 是否包含名为 identifier 的嵌套类型
} && std::is_same_v<typename T::identifier, identifier>;        // 检查 T::identifier 是否等于 ducks::tma::descriptor::identifier
} // namespace descriptor
} // namespace tma
} // namespace ducks
namespace detail {
namespace tma {

/**
 * @brief 用于处理描述符复制的辅助模板
 * 
 * 这个模板类的作用是帮助从 TMA 描述符类型中提取轴和数据类型。
 * 根据类型 T 的不同，`descriptor_copy_helper` 会继承不同的轴值和数据类型。
 */
template<typename T> struct descriptor_copy_helper {};

// 如果 TMA 描述符符合特定条件，提取其轴值和类型
template<kittens::ducks::tma::descriptor::all _T> struct descriptor_copy_helper<_T> { 
    static constexpr int value = _T::axis; // 提取轴值
    using T = _T::T;    // 提取数据类型
};

// 对其他类型的处理，如共享类型（`st` 类型和 `sv` 类型）
template<kittens::ducks::st::all _T> struct descriptor_copy_helper<_T> { 
    static constexpr int value = 2;     // 默认轴值
    using T = _T;   // 数据类型为 _T
};


template<kittens::ducks::sv::all _T> struct descriptor_copy_helper<_T> { 
    static constexpr int value = -1;    // 特殊轴值
    using T = _T;   // 数据类型为 _T
};

// 提供便捷的类型别名
template<typename T> using descriptor_copy_helper_t = descriptor_copy_helper<T>::T;


template<typename T> static constexpr int descriptor_copy_helper_v = descriptor_copy_helper<T>::value;
} // namespace tma
} // namespace detail

/**
 * @brief TMA 描述符结构体
 * 
 * 该模板结构体用于定义一个全局内存布局描述符，支持不同类型和维度的布局。
 * 
 * @tparam _T 布局类型
 * @tparam _axis 轴索引（默认值为 -9999，表示使用类型默认轴值）
 * 
 * 该结构体通过模板概念确保传入的类型符合一定要求，并根据类型提取数据类型和轴值。
 */
namespace tma {
template<typename _T, int _axis=-9999> struct descriptor {
    using identifier = ducks::tma::descriptor::identifier;  // 使用 TMA 描述符的标识符
    using T = detail::tma::descriptor_copy_helper_t<_T>;    // 提取数据类型
    // 确保类型符合要求
    static_assert(ducks::st::all<T> || ducks::sv::all<T> || ducks::tma::descriptor::all<T>, "Must be a shared TK type to generate a TMA descriptor.");
    // 根据类型确定轴值
    static constexpr int axis = (
        ducks::tma::descriptor::all<_T> ? detail::tma::descriptor_copy_helper_v<_T> : // if a copy, inherit the axis from the original descriptor. 
        (_axis != -9999) ? _axis : detail::tma::descriptor_copy_helper_v<_T>); // if a default value was provided, use it.
    // 确保类型和轴值符合预期
    static_assert((kittens::ducks::st::all<T> && axis >= 0 && axis <= 2) || (kittens::ducks::sv::all<T> && axis == -1), "Internal template error detected.");
};
} // namespace tma
#endif

namespace detail {
/**
 * @brief 描述符字典结构，用于存储不同布局的描述符
 * 
 * 该结构体实现了一个描述符字典，用于管理多个描述符（如 TMA 描述符、共享类型描述符等）。
 * 支持不同的模板参数类型，并提供了与描述符相关的操作。
 */
template<typename... Args>
struct descriptor_dict {
    __host__ descriptor_dict() {}   // 默认构造函数
    template<typename T> __host__ descriptor_dict(T _, int b, int d, int r, int c) {}   // 模板构造函数
    __host__ __device__ descriptor_dict(const descriptor_dict &other) {}    // 拷贝构造函数
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    /**
     * @brief 获取 TMA 描述符
     * 
     * 该函数返回与类型 T 相关联的 TMA 描述符。
     * 如果请求的类型未在全局布局中初始化，则抛出静态断言错误。
     */
    template<typename T, int U> __device__ const CUtensorMap* get() const {
        static_assert(
            std::is_same_v<T, std::true_type> && std::is_same_v<T, std::false_type>,
            "SKILL ISSUE: Requested a TMA descriptor for a type not initialized in the global layout."
        );
        return nullptr;
    }
#endif
};

/**
 * @brief 特化的描述符字典结构，处理多个描述符
 * 
 * 当传入多个参数时，`descriptor_dict` 结构体会递归地创建并存储每个描述符。
 * 例如，可以创建多个 TMA 描述符、共享描述符等。
 */
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<typename _T, typename... Args>
struct descriptor_dict<_T, Args...> {
    static_assert(ducks::sv::all<_T> || ducks::st::all<_T> || ducks::tma::descriptor::all<_T>, "Must be a shared TK type to generate a TMA descriptor.");
    using DESC = kittens::tma::descriptor<_T>; // 获取 TMA 描述符
    CUtensorMap tma_desc;// 存储 TMA 描述符
    descriptor_dict<Args...> other_descs;// 递归存储其他描述符
    __host__ descriptor_dict() {}

    /**
     * @brief 构造函数，初始化 TMA 描述符和其他描述符
     */
    __host__ descriptor_dict(typename DESC::T::dtype *data, int b, int d, int r, int c): other_descs(data, b, d, r, c) {
        kittens::detail::tma::create_tensor_map<typename DESC::T, DESC::axis>(&tma_desc, data, b, d, r, c);
    }
    __host__ __device__ inline descriptor_dict(const descriptor_dict &other) :
        tma_desc(other.tma_desc), other_descs(other.other_descs) {}// 拷贝构造函数

    /**
     * @brief 获取 TMA 描述符
     * 
     * 根据类型和轴值，返回对应的 TMA 描述符。
     */
    template<typename U, int axis> __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<typename DESC::T, U> && DESC::axis == axis) { return &tma_desc; }
        else                                                                     { return other_descs.template get<U, axis>(); }
    }
};
#endif
}

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {

/**
 * @brief 空标识符结构体，用于标识全局内存布局
 * 
 * 该结构体用于区分全局内存布局的不同类型。在模板编程中，`identifier` 起到了类型标识符的作用，防止不同布局类型冲突。
 */
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
#ifdef DF_BLACKWELL
    /**
     * @brief 静态断言，用于检测 FP4 类型的处理
     * 
     * 如果 `_T` 类型是 `fp4e2m1`，则要求使用打包类型（如 `fp4e2m1_2` 或 `fp4e2m1_4`），
     * 这是针对特定硬件（如 Blackwell）进行优化的检查。
     */
    static_assert(!std::is_same_v<_T, fp4e2m1>, "For FP4 types, you must use a packed type (i.e., fp4e2m1_2 or fp4e2m1_4).");
#endif
    using identifier = ducks::gl::identifier;   // 使用 ducks::gl::identifier 作为类型标识符
    
    // 使用 base_types::packing 来定义 `_T` 的解包和打包类型
    using T     = base_types::packing<_T>::unpacked_type;   // 解包后的基本类型
    using T2    = base_types::packing<_T>::packed_type;     // 打包后的类型
    using dtype = T;    // 数据类型是 T 的解包类型

    T* raw_ptr; // 存储数据指针
    
    // 这些常量不会被用户修改，是用于内部计算和表示的维度信息
    static constexpr int __b__ = b, __d__ = d, __r__ = r, __c__ = c; // 批次、深度、行、列维度
    
    // 使用 ducks::gl::make_dim_t 定义内存布局的各个维度
    ducks::gl::make_dim_t<b> batch_internal;
    ducks::gl::make_dim_t<d> depth_internal;
    ducks::gl::make_dim_t<r> rows_internal;
    ducks::gl::make_dim_t<c> cols_internal;
    
    // 一些 getter 方法，用于获取布局的维度，支持静态和动态查询
    template <int B=__b__> __device__ __host__ static constexpr std::enable_if_t<(B > 0), int> batch() { return B; }
    template <int B=__b__> __device__ __host__ std::enable_if_t<(B == -1), int> batch() const { return batch_internal; }
    template <int D=__d__> __device__ __host__ static constexpr std::enable_if_t<(D > 0), int> depth() { return D; }
    template <int D=__d__> __device__ __host__ std::enable_if_t<(D == -1), int> depth() const { return depth_internal; }
    template <int R=__r__> __device__ __host__ static constexpr std::enable_if_t<(R > 0), int> rows() { return R; }
    template <int R=__r__> __device__ __host__ std::enable_if_t<(R == -1), int> rows() const { return rows_internal; }
    template <int C=__c__> __device__ __host__ static constexpr std::enable_if_t<(C > 0), int> cols() { return C; }
    template <int C=__c__> __device__ __host__ std::enable_if_t<(C == -1), int> cols() const { return cols_internal; }

    /**
     * @brief 计算总元素数
     * 
     * 计算该布局中总的元素数量，即批次、深度、行和列的乘积。
     * 
     * @return 元素数量
     */
    __device__ __host__ inline size_t numel() const { return static_cast<size_t>(batch()) * depth() * rows() * cols(); }
    
    // TMA 描述符，用于处理张量映射
    detail::descriptor_dict<TMA_Types...> tma_descs;

    /**
     * @brief 构造函数，用于初始化全局内存布局
     * 
     * @param _data 数据指针
     * @param _batch 批次维度
     * @param _depth 深度维度
     * @param _rows 行维度
     * @param _cols 列维度
     */
    __host__ inline gl(T *_data,
                        ducks::gl::make_arg_t<b> _batch,
                        ducks::gl::make_arg_t<d> _depth,
                        ducks::gl::make_arg_t<r> _rows,
                        ducks::gl::make_arg_t<c> _cols) :
            raw_ptr(_data), batch_internal(_batch), depth_internal(_depth), rows_internal(_rows), cols_internal(_cols) {
        tma_descs = detail::descriptor_dict<TMA_Types...>(raw_ptr, batch_internal, depth_internal, rows_internal, cols_internal);
    }

    /**
     * @brief 拷贝构造函数，用于复制全局内存布局
     * 
     * @param other 另一个 `gl` 对象
     */
    __host__ __device__ inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch_internal(other.batch_internal), depth_internal(other.depth_internal), rows_internal(other.rows_internal), cols_internal(other.cols_internal), tma_descs(other.tma_descs) {}
#if defined(DFS_HOPPER) || defined(DF_BLACKWELL)
    /**
     * @brief 获取指定 TMA 描述符
     * 
     * @tparam U 目标类型
     * @tparam axis 轴的索引
     * 
     * @return 指向 TMA 描述符的指针
     */
    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U, axis>();
    }

    /**
     * @brief 预取指定的 TMA 描述符
     * 
     * 通过 `prefetch.tensormap` 指令将数据预取到设备内存中，优化后续计算。
     * 
     * @tparam U 目标类型
     * @tparam axis 轴索引（默认为 2）
     */
    template<typename U, int axis=2> __device__ inline void prefetch_tma() const {
        const CUtensorMap *tma_desc = tma_descs.template get<U, axis>();
        asm volatile ("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(tma_desc)) : "memory"); // must be called by a single thread
    }
#endif
    /**
     * @brief 重载下标运算符，用于访问指定坐标的元素
     * 
     * 该运算符通过坐标索引来访问存储的数据。
     * 
     * @param idx 坐标类型
     * @return 指向元素的引用
     */
    __device__ inline T& operator[](const coord<ducks::default_type> &idx) const { // yes I am abusing the const qualifier here a bit.
        return raw_ptr[(((size_t)idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c];
    }

    /**
     * @brief 获取指定轴的形状（维度大小）
     * 
     * @tparam axis 轴索引（0：批次、1：深度、2：行、3：列）
     * @return 指定轴的大小
     */
    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else if constexpr (axis==3) { return size_t(cols()); }
    }
    /**
     * @brief 获取指定轴的步长（跨度）
     * 
     * @tparam axis 轴索引（0：批次、1：深度、2：行、3：列）
     * @return 步长大小
     */
    template<int axis> __device__ inline size_t stride() const { 
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return (size_t)depth()*rows()*cols(); }
        else if constexpr (axis==1) { return (size_t)rows()*cols(); }
        else if constexpr (axis==2) { return (size_t)cols(); }
        else if constexpr (axis==3) { return 1; }
    }
};

// 创建不同维度的 `gl` 类型别名
template<typename _T, int d, int r, int c, typename... TMA_Types> using gl3 = gl<_T, 1, d, r, c, TMA_Types...>;
template<typename _T, int r, int c, typename... TMA_Types>        using gl2 = gl<_T, 1, 1, r, c, TMA_Types...>;
template<typename _T, int c, typename... TMA_Types>               using gl1 = gl<_T, 1, 1, 1, c, TMA_Types...>;

// 全局布局概念，用于约束模板类型
namespace ducks {
namespace gl {
/**
* @brief Concept for all global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // 检查 T 是否有名为 identifier 的嵌套类型
} && std::is_same_v<typename T::identifier, identifier>; // 检查 T::identifier 是否等于 ducks::gl::identifier
}
}

// 辅助函数，用于创建不安全的全局布局参数
template<int N> auto make_unsafe_gl_arg(int param) { // typename std::conditional_t<(N < 0), std::nullptr_t, int>
    if constexpr (N > 0) { return nullptr; }
    else                 { return param;   }
}

/**
 * @brief 创建全局布局对象
 * 
 * 创建一个全局内存布局对象 `GL`，并进行维度匹配检查，确保参数维度与 `GL` 类型的维度一致。
 * 
 * @tparam GL 全局布局类型
 * @param data 数据指针
 * @param b 批次
 * @param d 深度
 * @param r 行数
 * @param c 列数
 * @return 一个 GL 类型的对象
 */
template<ducks::gl::all GL, bool safe=true> __host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if(GL::__b__ > 0 && b != GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(GL::__b__) + ", Got: " + std::to_string(b));
        }
        if(GL::__d__ > 0 && d != GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(GL::__d__) + ", Got: " + std::to_string(d));
        }
        if(GL::__r__ > 0 && r != GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(GL::__r__) + ", Got: " + std::to_string(r));
        }
        if(GL::__c__ > 0 && c != GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(GL::__c__) + ", Got: " + std::to_string(c));
        }
    }
    return GL(
        reinterpret_cast<typename GL::dtype*>(data),
        make_unsafe_gl_arg<GL::__b__>(b),
        make_unsafe_gl_arg<GL::__d__>(d),
        make_unsafe_gl_arg<GL::__r__>(r),
        make_unsafe_gl_arg<GL::__c__>(c)
    );
}

} // namespace kittens
