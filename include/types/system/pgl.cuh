/**
 * @file
 * @brief Templated layouts for parallel global memory.
 */

#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "../global/global.cuh"

namespace kittens {

/* ----------  Parallel global layout descriptor  ---------- */

namespace ducks {
namespace pgl {

// 结构体 identifier 用于标识 PGL 类型    
struct identifier {};

// 概念（Concept）定义：检查 T 类型是否符合所有并行全局布局的要求
/**
 * @brief Concept for all parallel global layouts.
 * @tparam T The type to check against the concept requirements.
 *
 * Requires:
 * - T has a nested type identifier that is the same as ducks::pgl::identifier.
 */
template<typename T> concept all = requires {
    typename T::identifier; // 要求 T 类型有一个嵌套类型 identifier
} && std::is_same_v<typename T::identifier, identifier>;    // 并且这个类型必须是 ducks::pgl::identifier

} // namespace pgl
} // namespace ducks

/**
 * @brief Parallel global layout. Represents a region of data spread across multiple devices.
 * @tparam GL The underlying global layout on each device.
 * @tparam NUM_DEVICES The number of GPU devices.
 * @tparam MULTICAST Whether the multicast object should be initialized by the caller.
 * @tparam TMA_Types The types of TMA descriptors to use for the multicast locations. 
           Only valid if MULTICAST is true.
 */
template<kittens::ducks::gl::all _GL, int NUM_DEVICES = 8, bool MULTICAST = true, typename... TMA_Types>
struct pgl {
    using identifier = ducks::pgl::identifier;  // pgl 中的标识符类型
    using GL = _GL;         // 使用传入的全局布局类型
    using T = GL::dtype;    // 数据类型，来自全局布局的 dtype
    using dtype = T;        // 类型别名

    static constexpr int num_devices = NUM_DEVICES; // 设备数量
    static constexpr bool multicast = MULTICAST;    // 是否启用多播

    T *mc_ptr; // 多播指针；如果未启用多播，则为 nullptr
    GL gls[NUM_DEVICES];    // 存储每个设备的全局布局

    detail::descriptor_dict<TMA_Types...> tma_descs;    // 存储 TMA 描述符

    // 提供索引操作符，访问指定设备的全局布局
    __host__ __device__ const GL &operator[](int idx) const { return gls[idx]; }

    // 多播指针获取函数
    __device__ inline T* mc_ptr_at(const coord<ducks::default_type> &idx) const {
        static_assert(MULTICAST, "Multicast is not enabled for this PGL.");
        const GL &gl = gls[0]; // 所有设备的布局应该是相同的
        return &mc_ptr[((idx.b * static_cast<uint64_t>(gl.depth()) + idx.d) * gl.rows() + idx.r) * gl.cols() + idx.c];
    }

    // 构造函数（没有多播指针）
    __host__ inline pgl(T **_data,  // 存储每个设备数据的指针数组
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
        pgl(std::make_index_sequence<NUM_DEVICES>{}, _data, _batch, _depth, _rows, _cols) { }

    // 构造函数（有多播指针）
    __host__ inline pgl(T *_mc_ptr, // 多播指针，由调用者初始化
                        T **_data,  // 存储每个设备数据的指针数组
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
        pgl(std::make_index_sequence<NUM_DEVICES>{}, _mc_ptr, _data, _batch, _depth, _rows, _cols) { }
    
    // 构造函数：使用 index_sequence 来展开 NUM_DEVICES 次
    template<size_t... I>
    __host__ inline pgl(std::index_sequence<I...>,
                        T **_data,
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
            mc_ptr(nullptr), gls{GL(_data[I], _batch, _depth, _rows, _cols)...} {
        static_assert(!MULTICAST, "Multicast pointer not passed to multicast-enabled PGL.");
    }

    // 构造函数：使用 index_sequence 来展开 NUM_DEVICES 次
    template<size_t... I>
    __host__ inline pgl(std::index_sequence<I...>,
                        T *_mc_ptr,
                        T **_data,
                        ducks::gl::make_arg_t<GL::__b__> _batch,
                        ducks::gl::make_arg_t<GL::__d__> _depth,
                        ducks::gl::make_arg_t<GL::__r__> _rows,
                        ducks::gl::make_arg_t<GL::__c__> _cols) : 
            mc_ptr(_mc_ptr), gls{GL(_data[I], _batch, _depth, _rows, _cols)...} {
        static_assert(MULTICAST, "Multicast pointer passed to multicast-disabled PGL.");
        // 初始化 TMA 描述符
        tma_descs = detail::descriptor_dict<TMA_Types...>(
            mc_ptr, gls[0].batch_internal, gls[0].depth_internal, gls[0].rows_internal, gls[0].cols_internal);
    }
    
    // 获取指定类型和维度的 TMA 描述符
    template<typename U, int axis> 
    __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U, axis>();   // 获取 TMA 描述符
    }
    
    // 获取批次、深度、行列数等属性
    __host__ __device__ inline auto batch() const { return gls[0].batch(); }
    __host__ __device__ inline auto depth() const { return gls[0].depth(); }
    __host__ __device__ inline auto rows() const { return gls[0].rows(); }
    __host__ __device__ inline auto cols() const { return gls[0].cols(); }
    __host__ __device__ inline auto numel() const { return gls[0].numel(); }
    
    // 获取指定轴的形状和步幅
    template<int axis> __device__ inline size_t shape() const { return gls[0].template shape<axis>(); }
    template<int axis> __device__ inline size_t stride() const { return gls[0].template stride<axis>(); }
};

// 用于创建 PGL 的辅助函数，传入数据、批次、深度、行和列信息
template<ducks::pgl::all PGL, bool safe=true> __host__ inline PGL make_pgl(
    uint64_t *data, int b, int d, int r, int c
) {
    if constexpr (safe) {   // 检查维度是否匹配
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(PGL::GL::__b__) + ", Got: " + std::to_string(b));
        }
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(PGL::GL::__d__) + ", Got: " + std::to_string(d));
        }
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(PGL::GL::__r__) + ", Got: " + std::to_string(r));
        }
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(PGL::GL::__c__) + ", Got: " + std::to_string(c));
        }
    }
    return PGL(
        reinterpret_cast<typename PGL::dtype**>(data),
        make_unsafe_gl_arg<PGL::GL::__b__>(b),
        make_unsafe_gl_arg<PGL::GL::__d__>(d),
        make_unsafe_gl_arg<PGL::GL::__r__>(r),
        make_unsafe_gl_arg<PGL::GL::__c__>(c)
    );
}

// 用于创建具有多播指针的 PGL 的辅助函数
template<ducks::pgl::all PGL, bool safe=true> __host__ inline PGL make_pgl(
    uint64_t mc_ptr, uint64_t *data, int b, int d, int r, int c
) {
    if constexpr (safe) {   // 检查维度是否匹配
        if (PGL::GL::__b__ > 0 && b != PGL::GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch. Expected: " + std::to_string(PGL::GL::__b__) + ", Got: " + std::to_string(b));
        }
        if (PGL::GL::__d__ > 0 && d != PGL::GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch. Expected: " + std::to_string(PGL::GL::__d__) + ", Got: " + std::to_string(d));
        }
        if (PGL::GL::__r__ > 0 && r != PGL::GL::__r__) {
            throw std::runtime_error("Row dimension mismatch. Expected: " + std::to_string(PGL::GL::__r__) + ", Got: " + std::to_string(r));
        }
        if (PGL::GL::__c__ > 0 && c != PGL::GL::__c__) {
            throw std::runtime_error("Column dimension mismatch. Expected: " + std::to_string(PGL::GL::__c__) + ", Got: " + std::to_string(c));
        }
    }
    return PGL(
        reinterpret_cast<typename PGL::dtype*>(mc_ptr),
        reinterpret_cast<typename PGL::dtype**>(data),
        make_unsafe_gl_arg<PGL::GL::__b__>(b),
        make_unsafe_gl_arg<PGL::GL::__d__>(d),
        make_unsafe_gl_arg<PGL::GL::__r__>(r),
        make_unsafe_gl_arg<PGL::GL::__c__>(c)
    );
}

// 用于跨设备同步的便利类型别名
template <int NUM_DEVICES>
using barrier_t = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

} // namespace kittens