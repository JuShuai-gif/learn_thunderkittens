/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rt_layout
 * 
 * @brief 用于寄存器瓷砖布局的模板元编程的命名空间。
 */
namespace rt_layout {

/**
 * @brief 用于标识寄存器瓷砖的行优先布局（row-major layout）。
 * 
 * 行优先布局是大多数矩阵的默认布局，其中数据按行顺序排列。
 */
struct row {}; // 用于大多数矩阵
/**
 * @brief 用于标识寄存器瓷砖的列优先布局（col-major layout）。
 * 
 * 列优先布局通常用于矩阵乘法运算中的 B 矩阵，数据按列顺序排列。
 */
struct col {}; // 用于 MMA 运算中的 B 矩阵

/**
 * @brief 一个概念，用于检查一个类型是否是寄存器瓷砖的布局类型。
 * 
 * @tparam T 需要检查的类型
 * 
 * 要求：
 * - 类型 T 必须是 `row` 或 `col` 类型之一。
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;


/**
 * @brief 用于生成转置布局的结构体。
 * 
 * @tparam L 输入的布局类型（`row` 或 `col`）
 * 
 * 根据输入布局类型生成对应的转置布局类型：
 * - 如果输入是 `row`，则输出 `col`；
 * - 如果输入是 `col`，则输出 `row`。
 */
template<all L> struct transpose      { using type = col; };    // 默认将 row 转置为 col
template<>      struct transpose<col> { using type = row; };    // 将 col 转置为 row

} // namespace rt_layout
} // namespace ducks
} // namespace kittens