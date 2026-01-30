/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens{
namespace ducks{

/**
 * @namespace rv_layout
 * 
 * 该命名空间用于模板元编程，主要目的是定义不同的寄存器向量布局（vector layouts）类型。
 */
namespace rv_layout {

/**
 * 该结构体是一个虚拟类型，用于标识对齐布局（aligned layout），通常是 8x 重复布局。
 * 这里的 inner_dim 被定义为 2，表示每个维度在对齐布局中的大小。
 */
struct align { constexpr static int inner_dim = 2; };// inner_dim 表示布局中的内维度大小
/**
 * @brief A dummy type used to identify an orthogonal (4x replicated) layout.
 * 
 * 该结构体是一个虚拟类型，用于标识正交布局（orthogonal layout），通常是 4x 重复布局。
 * inner_dim 被定义为 1，表示每个维度在正交布局中的大小。
 */
struct ortho { constexpr static int inner_dim = 1; };// inner_dim 表示布局中的内维度大小

/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 * 
 * 该结构体是一个虚拟类型，用于标识非重复布局（unreplicated layout）。这种布局适合进行更高效的协同加载（coalesced loads）
 * 和向量运算，如 layernorm（层归一化）。
 * inner_dim 被定义为 1，表示布局中没有额外的内维度重复。
 */
struct naive { constexpr static int inner_dim = 1; }; // inner_dim 表示布局中的内维度大小

/**
 * @brief A concept to check if a type is a register tile layout.
 * 
 * 该模板概念用于检查一个类型是否是寄存器瓦片布局类型。支持三种类型：align、ortho、naive。
 * 如果类型是这三者之一，则该概念成立。
 */
template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;


}

}



}



























