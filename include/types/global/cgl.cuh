/**
 * @file
 * @brief Templated layouts for complex global memory.
 * 
 * 本文件定义了用于复杂全局内存布局的模板结构，特别是针对全局内存的复杂数据类型（如复数）的处理。
 * 使用 CUDA 编程中常见的模式，如 `kittens` 和 `ducks` 命名空间，来组织数据布局。
 * 
 * 这个文件使用了多种技巧，如模板元编程、概念（concepts）等。
 */

#pragma once

#include "../../common/common.cuh"
#include "../shared/cst.cuh"    // 引入共享的常量和类型定义
#include "gl.cuh"               // 引入全局内存相关操作的头文件
#include "util.cuh"
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "tma.cuh"              // 如果是特定平台（Hopper 或 Blackwell），引入 TMA 相关的头文件
#endif

namespace kittens {


/**
 * @brief 空结构体，用作复数全局内存布局的标识符。
 * 
 * 这个标识符结构体用于区分不同的全局内存布局。它不包含数据成员，
 * 但作为一个类型标识符存在，用于模板编程中匹配特定布局类型。
 */
namespace ducks {
namespace cgl {
struct identifier {};
}
}

// 注释掉的部分，可能涉及特定的类型检查，当前暂时没有启用
// namespace detail {
// template<typename T> concept tile = ducks::cst::all<T> || ducks::crt::all<T>;
// template<typename T> concept vec  = ducks::csv::all<T> || ducks::crv::all<T>;
// }

/// 定义一个模板结构 `cgl`，用于表示复杂的全局内存布局
/**
 * @brief 复杂全局内存布局模板
 * @tparam _GL 全局布局组件类型
 * 
 * 这个结构体代表了一种全局内存布局，它包含一个复数组件（real 和 imag），用于存储复杂数据类型（例如复数）的实部和虚部。
 * 它会依赖于 `_GL` 类型，该类型定义了全局内存布局的具体细节。
 */
template<kittens::ducks::gl::all _GL>
struct cgl {
    using identifier = ducks::cgl::identifier;  // 关联类型标识符，指示该布局类型是复数布局
    using component  = _GL;                     // 组件类型，决定了实际的数据布局和存储方式
    using T          = component::T;            // 元素类型 T，代表存储的基本数据类型
    using T2         = component::T2;           // 另一个元素类型 T2，可能代表其他数据类型
    using dtype      = component::dtype;        // 数据类型 dtype，通常指代元素的基础数据类型
    component real, imag;
};

namespace ducks {
namespace cgl {
/**
 * @brief 概念（Concept）用于检查所有复杂全局内存布局类型
 * @tparam T 要检查的类型
 * 
 * 这个概念要求类型 T 必须包含一个名为 `identifier` 的嵌套类型，并且该类型必须等于 `ducks::cgl::identifier`。
 * 这是用于限制模板的机制，确保只有符合特定要求的类型才能被使用。
 */
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::cgl::identifier
}
}

}