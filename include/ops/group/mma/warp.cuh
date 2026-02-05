/**
 * @file
 * @brief Warp-level matrix multiply-accumulate operations for tiles stored in registers.
 * 该文件包含寄存器瓦片级的矩阵乘累加操作，用于warp级别的计算。
 */

/**
 * @brief Perform the HMMA.16816 operation with bf16 inputs and f32 accumulators.
 * 使用bf16输入和f32累加器执行HMMA.16816操作。
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
 * 此函数使用`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`指令执行半精度矩阵乘累加操作。
 *
 * Operation: D = A * B + C
 * 运算：D = A * B + C
 * Matrix dimensions: A[16x16], B[16x8], C[16x8], D[16x8] (in bf16/f32 precision)
 * 矩阵维度：A[16x16], B[16x8], C[16x8], D[16x8]（bf16/f32精度）
 * Each thread holds part of the matrices in registers.
 * 每个线程在寄存器中保存矩阵的一部分。
 *
 * @param[out] d0 The first half of the output float2 accumulator. (D矩阵的前半部分，包含4个float元素)
 * @param[out] d1 The second half of the output float2 accumulator. (D矩阵的后半部分，包含4个float元素)
 * @param[in] a0 The first half of the first input bf16_2 matrix. (A矩阵第一部分，每个bf16_2包含2个bf16元素)
 * @param[in] a1 The second half of the first input bf16_2 matrix. (A矩阵第二部分)
 * @param[in] a2 The first half of the second input bf16_2 matrix. (A矩阵第三部分)
 * @param[in] a3 The second half of the second input bf16_2 matrix. (A矩阵第四部分)
 * @param[in] b0 The first half of the bf16_2 matrix B. (B矩阵前半部分)
 * @param[in] b1 The second half of the bf16_2 matrix B. (B矩阵后半部分)
 * @param[in] c0 The first half of the float2 accumulator matrix C. (C累加器矩阵前半部分)
 * @param[in] c1 The second half of the float2 accumulator matrix C. (C累加器矩阵后半部分)
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    // 内联汇编实现HMMA操作
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        // PTX指令：同步的、对齐的矩阵乘累加指令
        // 计算: D = A * B + C
        // m16n8k16: 计算16x8矩阵，使用16x16的A和16x8的B
        // row.col: A按行主序访问，B按列主序访问
        // f32.bf16.bf16.f32: 输出f32，输入A为bf16，输入B为bf16，累加器C为f32
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        // 输出操作数：{%0-%3}对应d0.x, d0.y, d1.x, d1.y (共4个f32寄存器)
        "{%0, %1, %2, %3}, " \
        // 输入A操作数：{%4-%7}对应a0-a3 (4个bf16_2，每个包含2个bf16，共8个bf16值)
        "{%4, %5, %6, %7}, " \
        // 输入B操作数：{%8, %9}对应b0, b1 (2个bf16_2，共4个bf16值)
        "{%8, %9}, " \
        // 输入C操作数：{%10-%13}对应c0.x, c0.y, c1.x, c1.y (4个f32寄存器)
        "{%10, %11, %12, %13};"


        // D矩阵输出操作数列表（读写操作数）
    :   "+f"(d0.x), "+f"(d0.y),// %0, %1: d0的x和y分量，作为f32寄存器
        "+f"(d1.x), "+f"(d1.y)// %2, %3: d1的x和y分量

        // A矩阵输入操作数列表（只读）
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),// %4, %5: 将bf16_2转换为uint32传递
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),// %6, %7

        // B矩阵输入操作数列表（只读）
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),// %8, %9

        // C矩阵输入操作数列表（只读）
        "f"(c0.x), "f"(c0.y),// %10, %11: c0的x和y分量，作为f32寄存器
        "f"(c1.x), "f"(c1.y)// %12, %13: c1的x和y分量
    );
}

/**
 * @brief Perform the HMMA.16816 operation with fp16 inputs and fp32 accumulators.
 * 使用fp16输入和fp32累加器执行HMMA.16816操作。
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction.
 * 此函数使用`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`指令执行半精度矩阵乘累加操作。
 *
 * Operation: D = A * B + C (with fp16 inputs and fp32 accumulation)
 * 运算：D = A * B + C（使用fp16输入和fp32累加）
 *
 * @param[out] d0 The first half of the output float2 accumulator. (D矩阵的前半部分)
 * @param[out] d1 The second half of the output float2 accumulator. (D矩阵的后半部分)
 * @param[in] a0 The first half of the first input half_2 matrix. (A矩阵第一部分，每个half_2包含2个fp16)
 * @param[in] a1 The second half of the first input half_2 matrix. (A矩阵第二部分)
 * @param[in] a2 The first half of the second input half_2 matrix. (A矩阵第三部分)
 * @param[in] a3 The second half of the second input half_2 matrix. (A矩阵第四部分)
 * @param[in] b0 The first half of the half_2 matrix B. (B矩阵前半部分)
 * @param[in] b1 The second half of the half_2 matrix B. (B矩阵后半部分)
 * @param[in] c0 The first half of the float2 accumulator matrix C. (C累加器矩阵前半部分)
 * @param[in] c1 The second half of the float2 accumulator matrix C. (C累加器矩阵后半部分)
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
        // PTX指令：同步的、对齐的矩阵乘累加指令
        // m16n8k16: 计算16x8矩阵，使用16x16的A和16x8的B
        // row.col: A按行主序，B按列主序
        // f32.f16.f16.f32: 输出f32，输入A为f16，输入B为f16，累加器C为f32
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        // 输出操作数：{%0-%3}对应d0.x, d0.y, d1.x, d1.y
        "{%0, %1, %2, %3}, " \
        // 输入A操作数：{%4-%7}对应a0-a3 (4个half_2，每个包含2个fp16)
        "{%4, %5, %6, %7}, " \
        // 输入B操作数：{%8, %9}对应b0, b1
        "{%8, %9}, " \
        // 输入C操作数：{%10-%13}对应c0.x, c0.y, c1.x, c1.y
        "{%10, %11, %12, %13};"

        // D矩阵输出操作数列表
    :   "+f"(d0.x), "+f"(d0.y),// %0, %1
        "+f"(d1.x), "+f"(d1.y)// %2, %3

        // A矩阵输入操作数列表
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),// %4, %5: 将half_2转换为uint32传递
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),// %6, %7

        // B矩阵输入操作数列表
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),// %8, %9

        // C矩阵输入操作数列表
        "f"(c0.x), "f"(c0.y),// %10, %11
        "f"(c1.x), "f"(c1.y)// %12, %13
    );
}

/**
 * @brief Perform the HMMA.16816 operation with fp16 inputs and fp16 accumulators.
 * 使用fp16输入和fp16累加器执行HMMA.16816操作。
 *
 * This function performs the half-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` instruction.
 * 此函数使用`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`指令执行半精度矩阵乘累加操作。
 *
 * Operation: D = A * B + C (all in fp16 precision)
 * 运算：D = A * B + C（全部使用fp16精度）
 *
 * @param[out] d0 The first half of the output half_2 accumulator. (D矩阵前半部分，使用half_2格式)
 * @param[out] d1 The second half of the output half_2 accumulator. (D矩阵后半部分，使用half_2格式)
 * @param[in] a0 The first half of the first input half_2 matrix. (A矩阵第一部分)
 * @param[in] a1 The second half of the first input half_2 matrix. (A矩阵第二部分)
 * @param[in] a2 The first half of the second input half_2 matrix. (A矩阵第三部分)
 * @param[in] a3 The second half of the second input half_2 matrix. (A矩阵第四部分)
 * @param[in] b0 The first half of the half_2 matrix B. (B矩阵前半部分)
 * @param[in] b1 The second half of the half_2 matrix B. (B矩阵后半部分)
 * @param[in] c0 The first half of the half_2 accumulator matrix C. (C累加器矩阵前半部分)
 * @param[in] c1 The second half of the half_2 accumulator matrix C. (C累加器矩阵后半部分)
 */
__device__ static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        // PTX指令：同步的、对齐的矩阵乘累加指令
        // m16n8k16: 计算16x8矩阵，使用16x16的A和16x8的B
        // row.col: A按行主序，B按列主序
        // f16.f16.f16.f16: 输出f16，输入A为f16，输入B为f16，累加器C为f16
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        // 输出操作数：{%0, %1}对应d0, d1 (每个half_2包含2个fp16，共4个fp16值)
        "{%0, %1}, " \
        // 输入A操作数：{%2-%5}对应a0-a3
        "{%2, %3, %4, %5}, " \
        // 输入B操作数：{%6, %7}对应b0, b1
        "{%6, %7}, " \
        // 输入C操作数：{%8, %9}对应c0, c1
        "{%8, %9};"

        // D矩阵输出操作数列表（只写操作数）
        // 注意：这里使用"=r"而不是"+f"，因为输出是half_2类型，需要作为32位寄存器传递
    :   "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))// %0, %1: 输出到d0, d1的32位寄存器

        // A矩阵输入操作数列表
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),// %2, %3: a0, a1作为32位寄存器
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),// %4, %5: a2, a3

        // B矩阵输入操作数列表
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),// %6, %7: b0, b1
        
        // C矩阵输入操作数列表
        "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1)) // %8, %9: c0, c1
    );
}

// 检查是否定义了KITTENS_HOPPER或KITTENS_BLACKWELL宏（对应NVIDIA Hopper或Blackwell架构）
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 为FP8使用fp8e4m3_2数据类型执行HMMA.16816操作
 * 
 * 使用mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32指令，
 * 但使用fp8e4m3_2（2个FP8值）而不是fp8e4m3_4
 */
/**
 * @brief 为FP8执行HMMA.16816操作
 *
 * 此函数使用`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`指令执行fp8精度的矩阵乘加操作。
 * HMMA.16816表示：输出矩阵大小为16x16，使用8位浮点数（FP8），k维度为32。
 *
 * @param[out] d0 输出float2累加器的前半部分
 * @param[out] d1 输出float2累加器的后半部分
 * @param[in] a0,a1,a2,a3 输入FP8矩阵A的值
 * @param[in] b0,b1 输入FP8矩阵B的值
 * @param[in] c0,c1 输入float2累加器矩阵C的值
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                       const fp8e4m3_4 &a0, const fp8e4m3_4 &a1, 
                                       const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
                                       const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
                                       const float2 &c0, const float2 &c1) {
    // 内联汇编：使用Tensor Core执行FP8精度的矩阵乘加操作
    asm volatile(
        // 指令格式：mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        // m16n8k32: 计算16x8的矩阵块，内积维度为32
        // row.col: A矩阵按行主序，B矩阵按列主序
        // f32.e4m3.e4m3.f32: 输入为e4m3格式的FP8，累加器和输出为f32
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        // 输出矩阵D（4个32位浮点数，存储在2个float2中）
        "{%0, %1, %2, %3}, "
        // 输入矩阵A（4个FP8值，每个打包为32位）
        "{%4, %5, %6, %7}, "
        // 输入矩阵B（2个FP8值，每个打包为32位）
        "{%8, %9}, "
        // 输入累加器矩阵C（4个32位浮点数，存储在2个float2中）
        "{%10, %11, %12, %13};"
        
        // D矩阵（输出）操作数：4个32位浮点数
        // +表示读写操作数（输入并输出）
        : "+f"(d0.x), "+f"(d0.y),
          "+f"(d1.x), "+f"(d1.y)
        
        // A矩阵操作数：4个FP8值，每个打包为32位
        // r表示32位通用寄存器
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        
        // B矩阵操作数：2个FP8值，每个打包为32位
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        
        // C矩阵操作数：4个32位浮点数
        // f表示浮点寄存器
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
#endif

/**
 * @brief 行布局的基础矩阵乘加操作
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A * B + C，其中A和B都是BF16格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<bf16_2, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<bf16_2, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<bf16,  ducks::rt_layout::row> &a,
                                    const rt_base<bf16,  ducks::rt_layout::col> &b, // 列主序模式
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    // 第一次hmma16816调用：处理B矩阵的前两列
    hmma16816(
        d.data[0], d.data[1],   // 输出：D的前两列
        a.data[0], a.data[1], a.data[2], a.data[3],// 输入：A矩阵
        b.data[0], b.data[2],   // 输入：B矩阵的第0和第2列
        c.data[0], c.data[1]    // 输入：C的前两列
    );
    // 第二次hmma16816调用：处理B矩阵的后两列
    hmma16816(
        d.data[2], d.data[3],   // 输出：D的后两列
        a.data[0], a.data[1], a.data[2], a.data[3],// 输入：A矩阵（与第一次相同）
        b.data[1], b.data[3],   // 输入：B矩阵的第1和第3列
        c.data[2], c.data[3]    // 输入：C的后两列
    );
}

/**
 * @brief 行布局的基础矩阵乘加操作（fp16输入和fp32累加器）
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A * B + C，其中A和B都是half格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<half_2, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<half_2, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<half,  ducks::rt_layout::row> &a,
                                    const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}

// 再次检查Hopper或Blackwell架构，处理FP8数据类型
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 行布局的基础矩阵乘加操作（FP8输入）
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A * B + C，其中A和B都是FP8格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<fp8e4m3, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<fp8e4m3, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::row> &d,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                    const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief 行布局的基础矩阵乘加操作（half输入和输出）
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A * B + C，其中A、B、C和D都是half格式。
 *
 * @param[out] d 输出rt_base<half_2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<half_2, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<half_2, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<half_2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AB_base(rt_base<half, ducks::rt_layout::row> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<half, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}

/**
 * @brief 行布局的基础点积操作（计算A * B^T）
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础点积操作。
 * 计算：D = A * B^T + C，其中A和B都是BF16格式，C和D是float格式。
 * B^T表示B的转置，但在行主序模式下访问。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<bf16_2, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<bf16_2, row_layout>矩阵（行主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::row> &a,
                                     const rt_base<bf16,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    // 注意：B矩阵的索引顺序看起来是反的（0,2然后1,3）
    // 这可能是由于Tensor Core指令在计算A * B^T时的特殊数据布局要求
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // 这个看起来需要反向访问
        c.data[0], c.data[1]
    );

    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // 这个看起来需要反向访问
        c.data[2], c.data[3]
    );
}

/**
 * @brief 行布局的基础点积操作（计算A * B^T），使用fp16输入和fp32累加器
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础点积操作。
 * 计算：D = A * B^T + C，其中A和B都是half格式，C和D是float格式。
 * B^T表示B的转置，但在行主序模式下访问。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<half_2, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<half_2, row_layout>矩阵（行主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::row> &a,
                                     const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    // 注意：B矩阵的索引顺序看起来是反的（0,2然后1,3）
    // 这可能是由于Tensor Core指令在计算A * B^T时的特殊数据布局要求
    hmma16816(
        d.data[0], d.data[1], // 输出：D的前两列
        a.data[0], a.data[1], a.data[2], a.data[3],// 输入：A矩阵
        b.data[0], b.data[2], // 输入：B矩阵的第0和第2列（看起来需要反向访问）
        c.data[0], c.data[1]// 输入：C的前两列
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}

// 检查是否定义了KITTENS_HOPPER或KITTENS_BLACKWELL宏（对应NVIDIA Hopper或Blackwell架构）
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 行布局的基础点积操作（计算A * B^T），使用FP8输入
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础点积操作。
 * 计算：D = A * B^T + C，其中A和B都是FP8格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<fp8e4m3x4, row_layout>矩阵
 * @param[in] b 第二个输入rt_base<fp8e4m3x4, row_layout>矩阵（行主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2], // for some reason this one seems to need to be backwards
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3], // for some reason this one seems to need to be backwards
        c.data[2], c.data[3]
    );
}
#endif


/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B），使用转置的A矩阵
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B + C，其中A和B都是BF16格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<bf16_2, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<bf16_2, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<bf16,  ducks::rt_layout::col> &a,
                                     const rt_base<bf16,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],// 输出：D的前两列
        a.data[0], a.data[1], a.data[2], a.data[3],// 输入：A矩阵（转置，列布局）
        b.data[0], b.data[2],// 输入：B矩阵的第0和第2列
        c.data[0], c.data[1]// 输入：C的前两列
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B），使用fp16输入和fp32累加器
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B + C，其中A和B都是half格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<half_2, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<half_2, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<half,  ducks::rt_layout::col> &a,
                                     const rt_base<half,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}

// 再次检查Hopper或Blackwell架构，处理FP8数据类型
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B），使用FP8输入
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B + C，其中A和B都是FP8格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<fp8e4m3x4, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<fp8e4m3x4, col_layout>矩阵（列主序模式）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::row> &d,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                     const rt_base<fp8e4m3,  ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B^T），A和B都转置
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B^T + C，其中A和B都是BF16格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<bf16_2, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<bf16_2, col_layout>矩阵（列主序模式，但B也转置）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<bf16,  ducks::rt_layout::col> &a,
                                      const rt_base<bf16,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],// 输出：D的前两列
        a.data[0], a.data[1], a.data[2], a.data[3],// 输入：A矩阵（转置，列布局）
        b.data[0], b.data[2],// 输入：B矩阵的第0和第2列（B转置，但使用行布局？注释说列主序模式，但参数类型是row）
        c.data[0], c.data[1]// 输入：C的前两列
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}

/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B^T），使用fp16输入和fp32累加器
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B^T + C，其中A和B都是half格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<half_2, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<half_2, row_layout>矩阵（行主序模式，但B转置）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<half,  ducks::rt_layout::col> &a,
                                      const rt_base<half,  ducks::rt_layout::row> &b, // in row-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}

// 再次检查Hopper或Blackwell架构，处理FP8数据类型
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
/**
 * @brief 行布局的基础矩阵乘加操作（计算A^T * B^T），使用FP8输入
 *
 * 此函数使用`hmma16816`函数对行布局的矩阵执行基础矩阵乘加操作。
 * 计算：D = A^T * B^T + C，其中A和B都是FP8格式，C和D是float格式。
 *
 * @param[out] d 输出rt_base<float2, row_layout>累加器
 * @param[in] a 第一个输入rt_base<fp8e4m3x4, col_layout>矩阵（A的转置，因此使用列布局）
 * @param[in] b 第二个输入rt_base<fp8e4m3x4, col_layout>矩阵（列主序模式，但B转置）
 * @param[in] c 输入rt_base<float2, row_layout>累加器矩阵
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::row> &d,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::col> &a,
                                      const rt_base<fp8e4m3,  ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::row> &c) {
    hmma16816(
        d.data[0], d.data[1],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[0], b.data[2],
        c.data[0], c.data[1]
    );
    hmma16816(
        d.data[2], d.data[3],
        a.data[0], a.data[1], a.data[2], a.data[3],
        b.data[1], b.data[3],
        c.data[2], c.data[3]
    );
}
#endif

/**
 * @brief Matrix multiply-accumulate operation.
 * 矩阵乘累加操作。
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 * 此函数使用`hmma16816`函数执行矩阵乘累加操作。
 *
 * @tparam N The number of row tiles. (行瓦片数量)
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix. (A矩阵列瓦片数和B矩阵行瓦片数)
 * @tparam M The number of column tiles for the B matrix. (B矩阵列瓦片数)
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator. (输出累加器矩阵D，行主序)
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix. (输入矩阵A，行主序)
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode. (输入矩阵B，列主序)
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix. (输入累加器矩阵C，行主序)
 *
 * 执行操作: D = A × B + C
 * 其中A: N×K, B: K×M, C/D: N×M (以瓦片为单位)
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    // 检查是否在warp级别调用（确保所有线程同步执行）
    KITTENS_CHECK_WARP
    // 静态断言：检查维度一致性
    static_assert(D::rows == A::rows && D::cols == B::cols); // D的维度必须与A的行数和B的列数匹配
    static_assert(A::cols == B::rows); // A的列数必须等于B的行数（内积维度）
    static_assert(D::rows == C::rows && D::cols == C::cols); // D和C的维度必须相同
    
    // 检查数据类型兼容性（针对Hopper/Blackwell架构）
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(
        // 情况1: D=float32, A=bf16, B=bf16, C=float32 (混合精度)
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        // 情况2: D=fp16, A=fp16, B=fp16, C=fp16 (全fp16精度)
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        // 情况3: D=float32, A=fp8, B=fp8, C=float32 (fp8混合精度)
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    // 非Hopper/Blackwell架构的数据类型检查
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif
    // 外层循环：遍历输出矩阵D的行瓦片
    #pragma unroll      // 完全展开循环以提高性能
    for(int n = 0; n < D::height; n++) {
        // 内层循环：遍历输出矩阵D的列瓦片
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // 首次乘累加：D[n][m] = A[n][0] × B[0][m] + C[n][m]
            mma_AB_base(
                d.tiles[n][m],// 输出：D矩阵的(n,m)瓦片
                a.tiles[n][0], // 输入：A矩阵的第n行，第0列瓦片
                b.tiles[0][m],// 输入：B矩阵的第0行，第m列瓦片（注意B是列主序）
                c.tiles[n][m]// 输入：C矩阵的(n,m)瓦片
            );
            // 内积循环：累加K个瓦片的乘积结果
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                // 后续乘累加：D[n][m] += A[n][k] × B[k][m]
                mma_AB_base(
                    d.tiles[n][m],// 输入输出：使用D自身作为累加器
                    a.tiles[n][k],// 输入：A矩阵的第n行，第k列瓦片
                    b.tiles[k][m],// 输入：B矩阵的第k行，第m列瓦片
                    d.tiles[n][m]// 输入：当前的D值作为累加器
                );
            }
        }
    }
}


/**
 * @brief Dot product operation for row layout.
 * 行主序的点积操作（矩阵A乘以矩阵B的转置）。
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 * 此函数使用`hmma16816`函数执行点积操作。
 *
 * 执行操作: D = A × Bᵀ + C
 * 其中Bᵀ表示B的转置
 *
 * @tparam N The number of row tiles. (行瓦片数量)
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix. (A矩阵列瓦片数和B矩阵行瓦片数)
 * @tparam M The number of column tiles for the B matrix. (B矩阵列瓦片数)
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator. (输出累加器矩阵D)
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix. (输入矩阵A，行主序)
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode. (输入矩阵B，行主序)
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix. (输入累加器矩阵C)
 *
 * 注意：B是行主序且维度为(M, K)，但实际计算中使用Bᵀ，所以相当于K×M的矩阵
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // 注意：B是行主序且维度为(M, K)，而不是列主序的(K, M)
                                const C &c) {
    KITTENS_CHECK_WARP// 检查是否在warp内执行
    // 静态断言：检查维度匹配
    static_assert(D::rows == A::rows && D::cols == B::rows); // 检查D的维度与A的行数、B的行数匹配
    static_assert(A::cols == B::cols); // 检查约减维度（K）相同
    static_assert(D::rows == C::rows && D::cols == C::cols); // 检查D与C维度匹配

    // 数据类型检查，支持不同精度组合
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)// Hopper或Blackwell架构
    static_assert(
        // 支持三种精度组合：
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||     // bf16输入，float累加
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)  ||     // half输入输出
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)     // fp8输入，float累加
    );
    #else   // 非Hopper/Blackwell架构
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

    // 外层循环：遍历输出矩阵D的行瓦片（N维度）
    #pragma unroll // 循环展开优化
    for(int n = 0; n < D::height; n++) {
        // 内层循环：遍历输出矩阵D的列瓦片（M维度）        
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // 第一步：初始化累加，使用C的对应瓦片            
            mma_ABt_base(
                d.tiles[n][m],  // 输出瓦片
                a.tiles[n][0],  // A的第n行第0列瓦片
                b.tiles[m][0],  // B的第m行第0列瓦片（注意：B是行主序，这里取的是行索引m）
                c.tiles[n][m]   // C的对应瓦片作为初始累加值
            );

            // 后续步骤：K维度累加，从k=1开始
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}


/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 * 计算A转置乘以B的矩阵乘累加操作。
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 * 此函数使用`hmma16816`指令执行矩阵乘累加操作。
 *
 * 执行操作: D = Aᵀ × B + C
 *
 * @tparam N The number of row tiles. (行瓦片数量)
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix. (A矩阵列瓦片数和B矩阵行瓦片数)
 * @tparam M The number of column tiles for the B matrix. (B矩阵列瓦片数)
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator. (输出累加器矩阵D，行主序)
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix. (输入矩阵A，列主序，维度K×N)
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode. (输入矩阵B，列主序，维度K×M)
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix. (输入累加器矩阵C，行主序)
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行

    // 静态断言：检查维度匹配
    static_assert(D::rows == A::cols && D::cols == B::cols); // D的行数等于A的列数，D的列数等于B的列数
    static_assert(A::rows == B::rows); // 约减维度（K）相同：A的行数等于B的行数
    static_assert(D::rows == C::rows && D::cols == C::cols); // D与C维度匹配

    // 数据类型检查，支持不同精度组合
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

    // 外层循环：遍历输出矩阵D的行瓦片（N维度）
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        // 内层循环：遍历输出矩阵D的列瓦片（M维度）
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // 第一步：初始化累加，使用C的对应瓦片
            mma_AtB_base(
                d.tiles[n][m],  // 输出瓦片
                a.tiles[0][n],  // A的第0行第n列瓦片（注意：A是列主序，这里取的是列索引n）
                b.tiles[0][m],  // B的第0行第m列瓦片（注意：B是列主序，这里取的是列索引m）
                c.tiles[n][m]   // C的对应瓦片作为初始累加值
            );
            // 后续步骤：K维度累加，从k=1开始
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}


/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 * 计算A转置乘以B转置的矩阵乘累加操作。
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 * 此函数使用`hmma16816`指令执行矩阵乘累加操作。
 *
 * 执行操作: D = Aᵀ × Bᵀ + C
 *
 * @tparam N The number of row tiles. (行瓦片数量)
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix. (A矩阵列瓦片数和B矩阵行瓦片数)
 * @tparam M The number of column tiles for the B matrix. (B矩阵列瓦片数)
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator. (输出累加器矩阵D，行主序)
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix. (输入矩阵A，列主序，维度K×N)
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode. (输入矩阵B，行主序，维度M×K)
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix. (输入累加器矩阵C，行主序)
 */
template<ducks::rt::row_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行
    
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    #if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)   ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

    // 外层循环：遍历输出矩阵D的行瓦片（N维度）    
    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        // 内层循环：遍历输出矩阵D的列瓦片（M维度）
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // 第一步：初始化累加，使用C的对应瓦片
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            // 后续步骤：K维度累加，从k=1开始
            #pragma unroll
            for(int k = 1; k < A::height; k++) {    // 注意：对于列主序的A，height对应行数K
                mma_AtBt_base(
                    d.tiles[n][m],// 输出瓦片（同时作为输入累加器）
                    a.tiles[k][n],// A的第k行第n列瓦片
                    b.tiles[m][k],// B的第m行第k列瓦片
                    d.tiles[n][m]// 使用当前D瓦片作为累加器（更新累加结果）
                );
            }
        }
    }
}

/**
 * @brief Generalized matrix multiply-accumulate operation with optional transpositions.
 * 通用的矩阵乘累加操作，支持可选的转置标志。
 *
 * This function performs the matrix multiply-accumulate operation
 * with transposition flags for matrices A and B.
 * 此函数根据A和B的转置标志执行矩阵乘累加操作。
 *
 * @tparam trans_A Transposition flag for matrix A (transpose::T for transposed, transpose::N for not). (矩阵A转置标志)
 * @tparam trans_B Transposition flag for matrix B (transpose::T for transposed, transpose::N for not). (矩阵B转置标志)
 * @tparam D Output matrix type (must satisfy ducks::rt::all). (输出矩阵类型)
 * @tparam A First input matrix type (must satisfy ducks::rt::all). (第一个输入矩阵类型)
 * @tparam B Second input matrix type (must satisfy ducks::rt::all). (第二个输入矩阵类型)
 * @tparam C Accumulator matrix type (must satisfy ducks::rt::all). (累加器矩阵类型)
 * @param[out] d The output accumulator matrix. (输出累加器矩阵)
 * @param[in] a The first input matrix. (第一个输入矩阵)
 * @param[in] b The second input matrix. (第二个输入矩阵)
 * @param[in] c The input accumulator matrix. (输入累加器矩阵)
 */
template<int trans_A, int trans_B, ducks::rt::all D, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b,
                                  const C &c) {
    KITTENS_CHECK_WARP  // 检查是否在warp内执行
    
    // 根据转置标志选择对应的具体实现函数
    if constexpr(trans_A == transpose::T) { // A需要转置
        if constexpr(trans_B == transpose::T) { // B需要转置
            mma_AtBt(d, a, b, c);       // 计算 Aᵀ × Bᵀ + C
        } else {        // B不需要转置
            mma_AtB(d, a, b, c);        // 计算 Aᵀ × B + C
        }
    } else {    // A不需要转置
        if constexpr(trans_B == transpose::T) {// B需要转置
            mma_ABt(d, a, b, c);// 计算 A × Bᵀ + C
        } else { // B不需要转置
            mma_AB(d, a, b, c);// 计算 A × B + C（未在提供代码中显示，但假设存在）
        }
    }
}

/**
 * @brief Generalized matrix multiply-accumulate operation with optional transpositions (return version).
 * 通用的矩阵乘累加操作，支持可选的转置标志（返回结果版本）。
 *
 * This function performs the matrix multiply-accumulate operation
 * with transposition flags for matrices A and B and returns the result.
 * 此函数根据A和B的转置标志执行矩阵乘累加操作并返回结果。
 *
 * @tparam trans_A Transposition flag for matrix A (transpose::T for transposed, transpose::N for not). (矩阵A转置标志)
 * @tparam trans_B Transposition flag for matrix B (transpose::T for transposed, transpose::N for not). (矩阵B转置标志)
 * @tparam A First input matrix type (must satisfy ducks::rt::all). (第一个输入矩阵类型)
 * @tparam B Second input matrix type (must satisfy ducks::rt::all). (第二个输入矩阵类型)
 * @tparam C Accumulator matrix type (must satisfy ducks::rt::all). (累加器矩阵类型)
 * @param[in] a The first input matrix. (第一个输入矩阵)
 * @param[in] b The second input matrix. (第二个输入矩阵)
 * @param[in] c The input accumulator matrix. (输入累加器矩阵)
 * @return The result accumulator matrix of type C. (返回类型为C的结果累加器矩阵)
 */
template<int trans_A, int trans_B, ducks::rt::all A, ducks::rt::all B, ducks::rt::all C>
__device__ static inline C mma(const A &a,
                               const B &b,
                               const C &c) {
    KITTENS_CHECK_WARP // 检查是否在warp内执行
    C d; // 创建输出矩阵（默认构造）

    // 根据转置标志选择对应的具体实现函数    
    if constexpr(trans_A == transpose::T) {     // A需要转置
        if constexpr(trans_B == transpose::T) { // B需要转置
            mma_AtBt(d, a, b, c);   // 计算 Aᵀ × Bᵀ + C
        } else {    // B不需要转置
            mma_AtB(d, a, b, c);    // 计算 Aᵀ × B + C
        }
    } else {    // A不需要转置
        if constexpr(trans_B == transpose::T) {// B需要转置
            mma_ABt(d, a, b, c);// 计算 A × Bᵀ + C
        } else {// B不需要转置
            mma_AB(d, a, b, c);// 计算 A × B + C（未在提供代码中显示，但假设存在）
        }
    }
    return d;// 返回结果
}


//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  -------------------------------------------------- COMPLEX INPUTS --------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------



/**
 * @brief 复数矩阵乘加运算（半精度版本）
 *
 * 此函数使用半精度参数调用基础的mma_AB函数，实现复数矩阵的乘加运算
 * 复数矩阵乘法公式：(A_real + i*A_imag) * (B_real + i*B_imag) = 
 *                 (A_real*B_real - A_imag*B_imag) + i*(A_real*B_imag + A_imag*B_real)
 *
 * @tparam N A矩阵的行块数，也是结果矩阵的行块数
 * @tparam K A矩阵的列块数，同时也是B矩阵的行块数
 * @tparam M B矩阵的列块数，也是结果矩阵的列块数
 * @param[out] d 输出累加器矩阵，类型为rt_cmplx_hf<N, M, row_layout>，行主序
 * @param[in] a 第一个输入矩阵，类型为rt_cmplx_hf<N, K, row_layout>，行主序
 * @param[in] b 第二个输入矩阵，类型为rt_cmplx_hf<K, M, col_layout>，列主序
 * @param[in] c 输入累加器矩阵，类型为rt_cmplx_hf<N, M, row_layout>，行主序
 */
template<int N, int K, int M>
__device__ static inline void mma_AB(crt_hf<N, M, ducks::rt_layout::row> &d,
                               const crt_hf<N, K, ducks::rt_layout::row> &a,
                               const crt_hf<K, M, ducks::rt_layout::col> &b,
                               const crt_hf<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP// 检查是否在正确的warp中执行
    
    // 将输入累加器c的数据复制到输出累加器d中
    ::kittens::group<1>::copy(d.real, c.real);// 复制实部
    ::kittens::group<1>::copy(d.imag, c.imag);// 复制虚部

    // 创建临时矩阵用于存储A矩阵虚部的负值，以便使用单个累加寄存器
    rt_hf<N, K, ducks::rt_layout::row> tmp;
    // 定义float16类型的-1常量（0xFB80是half类型中-1的二进制表示）
    constexpr half factor = std::bit_cast<__half>(uint16_t(0xFB80));

    // 计算A_imag * (-1)，结果存入tmp
    ::kittens::group<1>::mul(tmp, a.imag, factor);

    // 计算实部：d.real = a.real * b.real + (-a.imag) * b.imag + c.real
    // 1. 计算 a.real * b.real，累加到d.real（此时d.real已包含c.real）
    mma_AB(d.real, a.real, b.real, d.real);
    // 2. 计算 (-a.imag) * b.imag，累加到d.real
    mma_AB(d.real, tmp, b.imag, d.real);

    // 计算虚部：d.imag = a.real * b.imag + a.imag * b.real + c.imag
    // 1. 计算 a.real * b.imag，累加到d.imag（此时d.imag已包含c.imag）
    mma_AB(d.imag, a.real, b.imag, d.imag);
    // 2. 计算 a.imag * b.real，累加到d.imag
    mma_AB(d.imag, a.imag, b.real, d.imag);
}

/**
 * @brief 复数矩阵乘加运算（bf16到fp32混合精度版本）
 *
 * 此函数使用bf16输入和fp32累加器，调用基础的mma_AB函数实现复数矩阵的乘加运算
 * 输入矩阵为bf16精度，累加器为fp32精度，可提高数值稳定性
 *
 * @tparam N A矩阵的行块数，也是结果矩阵的行块数
 * @tparam K A矩阵的列块数，同时也是B矩阵的行块数
 * @tparam M B矩阵的列块数，也是结果矩阵的列块数
 * @param[out] d 输出累加器矩阵，类型为rt_cmplx_fl<N, M, row_layout>，行主序，fp32精度
 * @param[in] a 第一个输入矩阵，类型为rt_cmplx_bf<N, K, row_layout>，行主序，bf16精度
 * @param[in] b 第二个输入矩阵，类型为rt_cmplx_bf<K, M, col_layout>，列主序，bf16精度
 * @param[in] c 输入累加器矩阵，类型为rt_cmplx_fl<N, M, row_layout>，行主序，fp32精度
 */
template<int N, int K, int M>
__device__ static inline void mma_AB(crt_fl<N, M, ducks::rt_layout::row> &d,
                               const crt_bf<N, K, ducks::rt_layout::row> &a,
                               const crt_bf<K, M, ducks::rt_layout::col> &b,
                               const crt_fl<N, M, ducks::rt_layout::row> &c) {
    KITTENS_CHECK_WARP  // 检查是否在正确的warp中执行
    
    // 将输入累加器c的数据复制到输出累加器d中
    ::kittens::group<1>::copy(d.real, c.real);
    ::kittens::group<1>::copy(d.imag, c.imag);

    // 创建临时矩阵用于存储A矩阵虚部的负值，以便使用单个累加寄存器
    kittens::rt_bf<N, K, ducks::rt_layout::row> tmp;
    // 定义bf16类型的-1常量（0xBF80是bfloat16类型中-1的二进制表示）
    constexpr bf16 factor = std::bit_cast<__nv_bfloat16>(uint16_t(0xBF80));
    // 计算A_imag * (-1)，结果存入tmp
    ::kittens::group<1>::mul(tmp, a.imag, factor);

    // 计算实部：d.real = a.real * b.real + (-a.imag) * b.imag + c.real
    // 注意：这里使用bf16输入和fp32累加器的混合精度乘法
    mma_AB(d.real, a.real, b.real, d.real); // a.real * b.real 累加到d.real
    mma_AB(d.real, tmp, b.imag, d.real);    // (-a.imag) * b.imag 累加到d.real
    
    // 计算虚部：d.imag = a.real * b.imag + a.imag * b.real + c.imag
    mma_AB(d.imag, a.real, b.imag, d.imag); // a.real * b.imag 累加到d.imag
    mma_AB(d.imag, a.imag, b.real, d.imag); // a.imag * b.real 累加到d.imag
}