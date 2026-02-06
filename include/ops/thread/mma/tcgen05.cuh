/**
 * @file tcgen05.cuh
 * @brief Tensor Core Generation 5 (TCGEN05) 的矩阵乘积累加操作。
 *        针对存储在张量内存中的 tile 进行优化的 PTX 汇编级操作。
 *        支持多种混合精度计算（FP16, BF16, FP8, FP4等）。
 */

#pragma once

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {
namespace detail {
namespace tcgen05 {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor

// ============================================================================
// 指令描述符生成函数
// ============================================================================

/**
 * @brief 为标准的张量核心 MMA 操作生成 PTX 指令描述符
 * @tparam D 输出/累加矩阵的数据类型（如 float, half）
 * @tparam AB 输入矩阵 A 和 B 的数据类型（如 half, bf16, fp8e4m3 等）
 * @tparam M 矩阵 A 的维度（行数），通常是 16 的倍数
 * @tparam N 矩阵 B 的维度（列数），通常是 8 的倍数
 * @tparam trans_a 是否转置矩阵 A
 * @tparam trans_b 是否转置矩阵 B
 * @tparam neg 是否对矩阵 A 取负
 * @return uint32_t 32位指令描述符，用于 PTX 汇编指令
 * 
 * @note 根据 NVIDIA PTX ISA 文档的指令描述符格式生成
 *       支持 FP16/BF16/FP8 等精度，以及转置、取负等操作
 */
template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
__device__ static inline constexpr uint32_t instruction_descriptor() {
    uint32_t desc = 0;
    // 处理 16位数据类型（FP16/BF16/FP8）    
    if constexpr (sizeof(AB) == 2) { // kind::f16
        // 类型检查：输出必须是 float 或与输入相同的 half 类型
        static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
        desc |= 0b00      << 0;  // [1:0] 稀疏性位（当前未使用）
        desc |= 0b0       << 2;  // [2] 密集模式
        desc |= 0b0       << 3;  // [3] FP类型不使用饱和

        // [5:4] D矩阵（输出/累加）数据类型        
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D矩阵是FP32
        }
        else {
            desc |= 0b00  << 4; // D矩阵是FP16
        }
        desc |= 0b0       << 6;  // [6] 保留位

        // [9:7] A矩阵输入类型，[12:10] B矩阵输入类型        
        if constexpr (std::is_same_v<AB, half>) {
            desc |= 0b000 << 7;  // 16-bit A输入为FP16
            desc |= 0b000 << 10; // 16-bit B输入为FP16
        } else if constexpr (std::is_same_v<AB, bf16>) {
            desc |= 0b001 << 7;  // 16-bit A输入为BF16
            desc |= 0b001 << 10; // 16-bit B输入为BF16
        } else if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A输入为FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B输入为FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A输入为FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B输入为FP8 e5m2
        }
        // [13] 是否对A矩阵取负
        if constexpr (neg) {
            desc |= 0b1   << 13;  // 对A矩阵取负
        }
        else {
            desc |= 0b0   << 13; // 不对A矩阵取负
        }
        desc |= 0b0       << 14;  // [14] 不对B矩阵取负（始终为0）

        // [15] 是否转置A矩阵
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // 转置A矩阵
        }
        else {
            desc |= 0b0   << 15; // 不转置A矩阵
        }

        // [16] 是否转置B矩阵
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // 转置B矩阵
        }
        else {
            desc |= 0b0  << 16; // 不转置B矩阵
        }

        // [22:17] B矩阵维度N（编码为N>>3）        
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // [23] 保留位

        // [28:24] A矩阵维度M（编码为M>>4）        
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // [29] 保留位
        desc |= 0b00     << 30; // [31:30] B矩阵重用移位（无移位）
    } 
    // 处理 8位/4位数据类型（FP8/FP4）
    else if constexpr (sizeof(AB) == 1) { // kind::f8f6f4
        // 类型检查：FP8/6/4必须累加到float或half
        static_assert(std::is_same_v<D, float> || std::is_same_v<D, half>); // FP8/6/4 has to accumulate to float or half
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types

        // [5:4] D矩阵（输出/累加）数据类型
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D矩阵是FP32
        }
        else {
            desc |= 0b00  << 4; // D矩阵是FP16
        }
        desc |= 0b0       << 6;  // [6] 保留位

        // [9:7] A/B矩阵输入类型（FP8/FP4）        
        if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A输入为FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B输入为FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A输入为FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B输入为FP8 e5m2
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            desc |= 0b101 << 7;  // 4-bit A输入为FP4 e2m1
            desc |= 0b101 << 10; // 4-bit B输入为FP4 e2m1
        }

        // [13] 是否对A矩阵取负        
        if constexpr (neg) {
            desc |= 0b1   << 13; // 对A矩阵取负
        }
        else {
            desc |= 0b0   << 13; // 不对A矩阵取负
        }
        desc |= 0b0       << 14; // [14] 不对B矩阵取负（始终为0）

        // [15] 是否转置A矩阵        
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // 转置A矩阵
        }
        else {
            desc |= 0b0   << 15; // 不转置A矩阵
        }

        // [16] 是否转置B矩阵
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // 转置B矩阵
        }
        else {
            desc |= 0b0  << 16; // 不转置B矩阵
        }

        // [22:17] B矩阵维度N（编码为N>>3）        
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // [23] 保留位

        // [28:24] A矩阵维度M（编码为M>>4）
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // [29] 保留位
        desc |= 0b00     << 30; // [31:30] B矩阵重用移位（无移位）
    } else {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
    }
    return desc;
};

/**
 * @brief 为带缩放因子的低精度MMA操作生成PTX指令描述符
 * @tparam D 输出/累加矩阵的数据类型
 * @tparam AB 输入矩阵A和B的数据类型（FP8或FP4）
 * @tparam SAB 缩放因子的数据类型（FP8e4m3或FP8e8m0）
 * @tparam M 矩阵A的维度
 * @tparam N 矩阵B的维度
 * @tparam neg 是否对矩阵A取负
 * @tparam scale_factor_id 缩放因子ID（0-3用于MXFP8，0或2用于NVFP4）
 * @return uint32_t 32位指令描述符
 * 
 * @note 用于MXFP8和NVFP4等需要缩放因子的低精度格式
 */
template<typename D, typename AB, typename SAB, int M, int N, bool neg=false, int scale_factor_id=0>
__device__ static inline constexpr uint32_t instruction_descriptor() {
    // 只支持MXFP8和NVFP4类型
    static_assert(std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp4e2m1_2>, "AB must be fp8e4m3 for f4e2m1");
    static_assert(std::is_same_v<SAB, fp8e4m3> || std::is_same_v<SAB, fp8e8m0>, "SAB must be either fp8e4m3 or fp8e8m0");

    // 缩放类型：0表示ue4m3，1表示ue8m0
    constexpr int scale_type = std::is_same_v<SAB, fp8e4m3> ? 0 : std::is_same_v<SAB, fp8e8m0> ? 1 : -1;

    uint32_t desc = 0;
    desc |= 0b00 << 0; // [1:0] SBZ（应为零）
    desc |= 0b0 << 2; // [2] 密集模式
    desc |= 0b0 << 3; // [3] SBZ
    
    // [5:4] 矩阵B缩放因子ID（MXFP8: 0-3; NVFP4: 0或2）
    desc |= scale_factor_id << 4; // Matrix B scale Factor ID (0, 1, 2, 3 for MXFP8; 0, 2 for NVFP4)
    desc |= 0b0 << 6; // [6] SBZ

    // [9:7] A矩阵输入类型，[12:10] B矩阵输入类型    
    if constexpr (std::is_same_v<AB, fp8e4m3>) { // MXFP8
        desc |= (0b000 << 7); // 矩阵A是E4M3
        desc |= (0b000 << 10); // 矩阵B是E4M3
    } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) { // NVFP4
        desc |= 0b001 << 7; // 矩阵A是E2M1
        desc |= 0b01 << 10; // 矩阵B是E2M1
        desc |= 0b0 << 12;  // [12] SBZ
    } else {
        static_assert(sizeof(AB) == 999, "Invalid AB type.");
    }

    // [13] 是否对A矩阵取负
    if constexpr (neg) {
        desc |= 0b1 << 13; // 对A矩阵取负
    }
    else {
        desc |= 0b0 << 13; // 不对A矩阵取负
    }

    desc |= 0b0 << 14; // [14] 不对B矩阵取负
    desc |= 0b0 << 15; // [15] 不转置A（所有情况）
    desc |= 0b0 << 16; // [16] 不转置B（所有情况）

    // [22:17] B矩阵维度N（编码为N>>3）
    desc |= (N >> 3) << 17; 

    // [23] 缩放类型（0是ue4m3，1是ue8m0）
    desc |= scale_type   << 23; // Scale type (0 is ue4m3, 1 is ue8m0)


    desc |= 0b000 << 24; // [26:24] SBZ

    // [28:27] A矩阵维度M（编码为M>>7）
    desc |= (M >> 7) << 27; // A matrix has dimension M, encoded

    // [30:29] 矩阵A缩放因子ID
    desc |= scale_factor_id  << 29; // Matrix A scale Factor ID (0, 1, 2, 3 for MXFP8; 0, 2 for NVFP4)

    // [31] K维度（NVFP4: 0是K=64，1是K=96；MXFP8: 无选择，SBZ，K总是32）
    desc |= 0b0  << 31; // K dimension (NVFP4: 0 is K=64, 1 is K=96; MXFP8: no choice, SBZ, K is always 32)

    return desc;
}

// ============================================================================
// TTGEN05 MMA 汇编指令封装
// ============================================================================

/**
 * @brief 执行张量-张量MMA操作（一个矩阵在张量内存，另一个通过描述符）
 * @tparam AB 输入矩阵的数据类型
 * @tparam acc 是否累加（1=累加到D，0=覆盖D）
 * @tparam ncta CTA组大小（1或2）
 * @param d_tt_addr D矩阵（输出）的张量内存地址
 * @param a_tt_addr A矩阵（输入）的张量内存地址
 * @param b_desc B矩阵的描述符
 * @param idesc 指令描述符
 * 
 * @note 使用PTX内联汇编调用tcgen05.mma指令
 *       支持f8f6f4和f16两种计算类型
 */
template<typename AB, int acc, int ncta=1>
__device__ static inline void tt_st(uint32_t d_tt_addr, uint32_t a_tt_addr, uint64_t b_desc, uint32_t idesc) {

    // 处理低精度类型（FP8/FP4）
    if constexpr (std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp8e5m2> || std::is_same_v<AB, fp4e2m1_2>) {
        if constexpr (ncta == 1) {
            // 单CTA组的MMA操作（kind::f8f6f4）
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            // 双CTA组的MMA操作（kind::f8f6f4）
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {    // 处理FP16/BF16类型
        if constexpr (ncta == 1) {
            // 单CTA组的MMA操作（kind::f16）
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {            // 双CTA组的MMA操作（kind::f16）
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

/**
 * @brief 执行标量-张量MMA操作（两个矩阵都通过描述符）
 * @tparam AB 输入矩阵的数据类型
 * @tparam acc 是否累加（1=累加到D，0=覆盖D）
 * @tparam ncta CTA组大小（1或2）
 * @param d_tt_addr D矩阵（输出）的张量内存地址
 * @param a_desc A矩阵的描述符
 * @param b_desc B矩阵的描述符
 * @param idesc 指令描述符
 * 
 * @note 使用PTX内联汇编调用tcgen05.mma指令
 *       两个输入矩阵都通过描述符传递，而不是张量内存地址
 */
template<typename AB, int acc, int ncta=1>
__device__ static inline void st_st(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    // 处理低精度类型（FP8/FP4）
    if constexpr (std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp8e5m2> || std::is_same_v<AB, fp4e2m1_2>) {
        if constexpr (ncta == 1) {
            asm volatile(            // 单CTA组的MMA操作（kind::f8f6f4）
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(            // 双CTA组的MMA操作（kind::f8f6f4）
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {    // 处理FP16/BF16类型
        if constexpr (ncta == 1) {
            asm volatile(            // 单CTA组的MMA操作（kind::f16）
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {            // 双CTA组的MMA操作（kind::f16）
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

// SS（Shared-Shared）矩阵乘法内核函数，支持微缩放（microscaling）格式
// AB: 输入矩阵数据类型（fp8e4m3或fp4e2m1_2）
// SAB: 缩放因子数据类型
// acc: 累加标志（0：不累加，1：累加到现有结果）
// ncta: CTA（线程块）组大小（1或2个线程块协作）
// block_size: 缩放块大小（16或32）
template<typename AB, typename SAB, int acc, int ncta=1, int block_size=16>
__device__ static inline void st_st(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t sa_tt_addr, uint32_t sb_tt_addr, uint32_t idesc) {
    // 静态断言：确保AB是支持的微缩放格式
    static_assert(std::is_same_v<AB, fp8e4m3> || std::is_same_v<AB, fp4e2m1_2>, "AB must be fp8e4m3 for f4e2m1");
    if constexpr (ncta == 1) {// 单CTA模式
        if constexpr (std::is_same_v<AB, fp8e4m3>) { // Block size is always 32; alias is 1X
            // fp8e4m3数据类型，块大小固定为32，缩放向量别名为1X
            asm volatile(
                "{.reg .pred p;\n\t" \// 声明谓词寄存器p
                "setp.eq.u32 p, 1, %6;\n\t" \// 设置谓词：如果acc==1则p为真
                // 执行微缩放的张量核心指令
                // cta_group::1：单CTA组
                // kind::mxf8f6f4：微缩放fp8格式
                // block_scale.scale_vec::1X：块缩放，缩放向量格式为1X
                "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;}\n"
            ::  "r"(d_tt_addr), // %0: 输出张量地址
            "l"(a_desc), // %1: A矩阵描述符
            "l"(b_desc), // %2: B矩阵描述符
            "r"(idesc), // %3: 指令描述符
            "r"(sa_tt_addr), // %4: A缩放因子地址
            "r"(sb_tt_addr), // %5: B缩放因子地址
            "n"(acc)// %6: 累加标志（立即数）
            );
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            if constexpr (block_size == 32) { // E8M0缩放格式
                // fp4e2m1_2数据类型，块大小32，缩放向量别名为2X
                asm volatile(
                    "{.reg .pred p;\n\t" \
                    "setp.eq.u32 p, 1, %6;\n\t" \
                    // kind::mxf4nvf4：微缩放fp4格式
                    // scale_vec::2X：缩放向量格式为2X
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;}\n"
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
            else { // E4M3或E8M0缩放格式，块大小16别名为4X

                asm volatile( // block_size == 16 is an alias for scale_vec::4X
                "{.reg .pred p;\n\t" \
                    "setp.eq.u32 p, 1, %6;\n\t" \
                    // scale_vec::4X：缩放向量格式为4X
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;}\n"
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type.");
        }
    }
    else {// 双CTA模式
        if constexpr (std::is_same_v<AB, fp8e4m3>) { // Block size is always 32; alias is 1X
            // 双CTA组的fp8e4m3微缩放指令
            asm volatile(
                "{.reg .pred p;\n\t" \
                "setp.eq.u32 p, 1, %6;\n\t" \
                // cta_group::2：双CTA组
                "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
            );
        } else if constexpr (std::is_same_v<AB, fp4e2m1_2>) {
            if constexpr (block_size == 32) { // E8M0缩放格式
                asm volatile(
                "{.reg .pred p;\n\t" \
                    "setp.eq.u32 p, 1, %6;\n\t" \
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;}\n"
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
            else {  // E4M3或E8M0缩放格式
                asm volatile( // block_size == 16 is an alias for scale_vec::4X
                "{.reg .pred p;\n\t" \
                    "setp.eq.u32 p, 1, %6;\n\t" \
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;}\n"
                ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sa_tt_addr), "r"(sb_tt_addr), "n"(acc)
                );
            }
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type.");
        }
    }
}

// 提交指令并同步的辅助函数
// ncta: CTA组大小
template <int ncta>
__device__ static inline void commit(kittens::semaphore &sem, uint16_t dst_cta_mask = 0b11) {
    if constexpr (ncta == 1) {
        // 单CTA提交：在共享内存屏障上执行arrive操作
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
        ::  "l"(__cvta_generic_to_shared(&sem)));// 将信号量地址转换为共享内存地址
    }
    else {
        // 双CTA提交：在集群共享内存屏障上执行多播arrive操作
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        ::  "l"(__cvta_generic_to_shared(&sem)), "h"(dst_cta_mask));// dst_cta_mask指定目标CTA掩码
    }
}

} // namespace tcgen05
} // namespace detail

// 非微缩放格式的归约维度常量定义
// 根据数据类型确定归约维度：half为16，bf16为8，其他为32
template<typename T_AB> constexpr int reduction_dimension = sizeof(T_AB) == 2 ? 16 : sizeof(T_AB) == 4 ? 8 : 32;

// ==================== TT（Tensor-Tensor）矩阵乘法 ====================
// trans_a: A矩阵是否转置
// n_trans_b: B矩阵是否不转置（1-转置）
// D: 输出张量类型
// A: 输入张量A类型（寄存器张量）
// B: 输入张量B类型（共享内存描述符）
// acc: 累加标志
// ncta: CTA组大小
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;// 计算B的实际转置标志

    // 矩阵维度计算和验证
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;// 输出行数
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256))));  // 验证输出寄存器大小

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;// 输出列数
    static_assert(N == D::cols); // 验证输出寄存器大小

    constexpr int K = trans_a ? A::rows : A::cols;// 归约维度
    static_assert((trans_b ? B::rows : B::cols) == K);  // K维度必须匹配
    static_assert(std::is_same_v<typename A::T, typename B::T>);  // A和B必须类型相同

    // 类型别名和验证
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;// 获取归约维度
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");
    // 验证支持的数据类型组合
    static_assert(
        // half输出支持的类型
        (std::is_same_v<T_D, half> && (
            std::is_same_v<T_AB, half> || 
            std::is_same_v<T_AB, fp8e4m3> || 
            std::is_same_v<T_AB, fp8e5m2> || 
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // float输出支持的类型   
        (std::is_same_v<T_D, float> && (
            std::is_same_v<T_AB, bf16> || 
            std::is_same_v<T_AB, half> ||
            std::is_same_v<T_AB, fp8e4m3> ||
            std::is_same_v<T_AB, fp8e5m2> ||
            std::is_same_v<T_AB, fp4e2m1_2>
        )),
        "Currently unsupported type combination for matrix multiply."
    );
    // 生成指令描述符
    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    // 创建B矩阵的共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    // 内存一致性屏障：确保之前的异步操作完成
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

    // 执行第一个TT矩阵乘法指令（带累加标志）
    detail::tcgen05::template tt_st<T_AB, acc, ncta>(
        d.addr,
        a.template chunk_addr<trans_a>(0),
        b_desc.chunk_descriptor(0),
        idesc
    );
    // 循环处理剩余的K维度分块
    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template tt_st<T_AB, 1, ncta>(// 后续块总是累加
            d.addr,
            a.template chunk_addr<trans_a>(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}

// TT矩阵乘法的带信号量版本
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);// 提交并同步
}

// ==================== SS（Shared-Shared）矩阵乘法 ====================
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;

    // 矩阵维度计算和验证
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256)))); // output register is correctly sized

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    static_assert(N == D::cols); // output register is correctly sized

    // constexpr int K = std::is_same_v<typename A::T, fp4e2m1_2> ? (trans_a ? A::rows : A::cols) * 2 : (trans_a ? A::rows : A::cols);
    constexpr int K = (trans_a ? A::rows : A::cols);
    static_assert((trans_b ? B::rows : B::cols) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // 类型别名和验证
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");
    // 验证支持的数据类型组合
    static_assert(
        // Half output with supported input types
        (std::is_same_v<T_D, half> && (
            std::is_same_v<T_AB, half> || 
            std::is_same_v<T_AB, fp8e4m3> || 
            std::is_same_v<T_AB, fp8e5m2> || 
            std::is_same_v<T_AB, fp4e2m1_2>
        )) ||
        // Float output with supported input types  
        (std::is_same_v<T_D, float> && (
            std::is_same_v<T_AB, bf16> || 
            std::is_same_v<T_AB, half> ||
            std::is_same_v<T_AB, fp8e4m3> ||
            std::is_same_v<T_AB, fp8e5m2> ||
            std::is_same_v<T_AB, fp4e2m1_2>
        )),
        "Currently unsupported type combination for matrix multiply."
    );
    // 生成指令描述符
    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    // 创建A和B的共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);

    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);
    // 内存一致性屏障
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        // 执行第一个SS矩阵乘法指令
    detail::tcgen05::template st_st<T_AB, acc, ncta>(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        idesc
    );
    // 循环处理剩余的K维度分块
    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template st_st<T_AB, 1, ncta>(
            d.addr,
            a_desc.chunk_descriptor(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}

// SS矩阵乘法的带信号量版本
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);
}

// ==================== SS矩阵乘法（带微缩放） ====================
// 支持MXFP8和NVFP4格式的微缩放
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {

    // 静态断言：验证输入类型和缩放类型的兼容性
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.
    static_assert(
        (std::is_same_v<typename A::T, fp8e4m3> && std::is_same_v<typename B::T, fp8e4m3>) ||
        (std::is_same_v<typename A::T, fp4e2m1_2> && std::is_same_v<typename B::T, fp4e2m1_2>),
        "A and B must be fp8e4m3 or fp4e2m1_2"
    );
    static_assert(
        (std::is_same_v<typename A::T, fp8e4m3> && (
            std::is_same_v<typename SA::T, fp8e8m0> && std::is_same_v<typename SB::T, fp8e8m0>
        )) || 
        (std::is_same_v<typename A::T, fp4e2m1_2> && (
            (std::is_same_v<typename SA::T, fp8e8m0> && std::is_same_v<typename SB::T, fp8e8m0>) ||
            (std::is_same_v<typename SA::T, fp8e4m3> && std::is_same_v<typename SB::T, fp8e4m3>)
        )),
        "SAB must be fp8e8m0 for fp8e4m3 element type, or fp8e4m3 / fp8e8m0 for fp4e2m1_2 element type");
    // 微缩放格式只支持float32累加器
    static_assert(std::is_same_v<typename D::T, float>, "Only float32 accumulator is supported for microscaling formats");
    using T_AB = A::T;
    using T_SAB = SA::T;
    using T_D  = D::T;
    // 根据缩放类型确定块大小
    constexpr int block_size = std::is_same_v<typename SA::T, fp8e4m3> ? 16 : 32;
    constexpr int trans_b = 1 - n_trans_b;

    // 矩阵维度计算
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    constexpr int K = std::is_same_v<typename A::T, fp4e2m1_2> ? (trans_a ? A::rows : A::cols) * 2 : (trans_a ? A::rows : A::cols);
    constexpr int red_dim = std::is_same_v<typename A::T, fp8e4m3> ? 32 : 64; // TODO: 对于sm_103a和fp4e2m1，2个CTA时也可以是96
    static_assert(K % red_dim == 0, "K dimension must be divisible by red_dim.");

    // M is 128 for 1 CTA, 128 or 256 for 2 CTAs
    static_assert(M == D::rows*ncta && ((ncta == 1 && M == 128) || (ncta == 2 && (M == 128 || M == 256))));

    // valid N are steps of 8 for 1 CTA, steps of 16 for 2 CTAs
    static_assert(N == D::cols && ((ncta == 1 && N%8 == 0) || (ncta == 2 && N%16 == 0)));

    // 创建共享内存描述符
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    // 内存一致性准备
    kittens::tensor_after_thread_sync();// 线程同步
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

    // 预生成4个指令描述符（用于不同的SFID）
    constexpr uint32_t idescs[4] = {
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 0>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 1>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 2>(),
        detail::tcgen05::instruction_descriptor<T_D, T_AB, T_SAB, M, N, false, 3>()
    };
    // 执行第一个微缩放矩阵乘法指令
    detail::tcgen05::template st_st<T_AB, T_SAB, acc, ncta, block_size>(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        sa.addr,
        sb.addr,
        idescs[0]
    );

    // 计算缩放偏移量
    constexpr int N_offset = N / 32; // 当N=256时，偏移为8
    constexpr int M_offset = M / 32 / ncta; // 当M=256时，偏移为4
    // 根据不同的格式处理剩余的K维度分块
    if constexpr (std::is_same_v<typename A::T, fp8e4m3>) { // FP8E4M3 + FP8E8M0 缩放 (MXFP8)
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.chunk_descriptor(i),
                b_desc.chunk_descriptor(i),
                sa.addr + (i >> 2) * M_offset,  // i / 4，每4个K块共享同一组缩放因子
                sb.addr + (i >> 2) * N_offset, // i / 4
                idescs[i % 4]// 循环使用4个指令描述符
            );
        }
    } else if constexpr (std::is_same_v<typename A::T, fp4e2m1_2> && block_size == 16) {// FP4E2M1 + FP8E4M3 缩放 (NVFP4)
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.chunk_descriptor(i),
                b_desc.chunk_descriptor(i),
                sa.addr + i * M_offset,// 每个K块有独立的缩放因子
                sb.addr + i * N_offset,
                idescs[0] // SFID始终为0
            );
        }
    } else if constexpr (std::is_same_v<typename A::T, fp4e2m1_2> && block_size == 32) { // FP4E2M1 + FP8E8M0 缩放
        #pragma unroll
        for (int i = 1; i < K / red_dim; i++) {
            detail::tcgen05::template st_st<T_AB, T_SAB, 1, ncta, block_size>(
                d.addr,
                a_desc.chunk_descriptor(i),
                b_desc.chunk_descriptor(i),
                sa.addr + (i >> 1) * M_offset,// i / 2，每2个K块共享同一组缩放因子
                sb.addr + (i >> 1) * N_offset,
                (i & 1) ? idescs[2] : idescs[0]  // 交替使用0和2号指令描述符
            );
        }
    } else {
        static_assert(sizeof(T_AB) == 999, "Should not reach here.");
    }
}

// 带微缩放的SS矩阵乘法（带信号量版本）
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, SA, SB, acc, ncta>(d, a, b, sa, sb);
    detail::tcgen05::commit<ncta>(sem);
}

// Accumulator / numcta wrappers
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma<trans_a, trans_b, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    static_assert(!trans_a && trans_b, "Only ABt supported for microscaling formats currently");
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}

// Transpose wrappers
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 0>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}

} // namespace kittens
