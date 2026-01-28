#pragma once

#include <concepts>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <type_traits>

// 用于检查 CUDA 驱动 API 调用
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// 用于检查 CUDA 运行时 API 调用
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// Convenience utility
/*
便捷的实用工具，用于检查 CUDA 错误
使用示例:
CHECK_CUDA_ERROR(cudaMalloc(&ptr, size));
CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
*/
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// 模板函数 check，用于检查 CUDA 错误
template <typename T> void check(
    T err,                      // 错误码
    char const* const func,     // 调用的函数名称
    char const* const file,     // 错误发生的文件名
    int const line              // 错误发生的行号
) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


namespace kittens{

// Tile dimension constant
// 定义 tile 的维度常量
constexpr int BASE_TILE_DIM = 16;   // 基本 tile 维度为 16
// 定义模板常量 TILE_COL_DIM，根据类型 T 的大小决定 tile 的列维度
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
// TILE_ROW_DIM 为固定值，与列维度一样，始终等于 BASE_TILE_DIM
template<typename T> constexpr int TILE_ROW_DIM = BASE_TILE_DIM;

// 计算 tile 中元素的总数，TILE_COL_DIM 和 TILE_ROW_DIM 是列和行的维度
template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T> * TILE_ROW_DIM<T>};

// 常量，表示一个 warp 中的线程数
constexpr int WARP_THREADS{32};

// 常量，表示四个 warps 组成的 warpgroup 中的线程数
constexpr int WARPGROUP_THREADS{128};

// 常量，表示一个 warpgroup 中的 warp 数量（这里是四个）
constexpr int WARPGROUP_WARPS{4};

// 获取当前线程所在的 warp 的 ID，warp 是 32 个线程的一个集合
__device__ __forceinline__ int warpid(){
    return threadIdx.x >> 5;    // 线程 ID 右移 5 位，得到 warp ID
}

// 获取当前线程所在的 warpgroup 的 ID，warpgroup 是由四个 warp 组成的
__device__ __forceinline__ int warpgroupid(){
    return threadIdx.x >> 7;    // 线程 ID 右移 7 位，得到 warpgroup ID
}

// 获取当前线程在其所属 warp 中的 lane ID，lane 是 warp 中的线程 ID
__device__ __forceinline__ int laneid(){
    return threadIdx.x & 0x1f;  // 使用位掩码获取线程在 warp 中的 lane ID（32 位掩码）
}

// 根据不同的硬件架构，定义最大共享内存的大小
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
constexpr int MAX_SHARED_MEMORY = 227 * 1024;   // Hopper 或 Blackwell 架构下的共享内存大小为 227KB
#elif defined(DF_AMPERRE)
constexpr int MAX_SHARED_MEMORY = 164 * 1024;   // Amperre 架构下的共享内存大小为 164KB
#elif defined(DF_ADA)
constexpr int MAX_SHARED_MEMORY = 164 * 1024;   // Amperre 架构下的共享内存大小为 164KB
#endif



/**
 * @brief Query the number of SMs on a device.
 * @param device_id GPU device ordinal. If negative, uses the current device.
 * @return The number of streaming multiprocessors.
 */
// 查询设备上的 SM（流处理器）数量
__host__ inline int num_sms(int device_id = -1){
    if (device_id < 0){
        CUDACHECK(cudaGetDevice(&device_id));   // 获取当前设备 ID
    }
    int sm_count;
    // 获取设备的 SM 数量
    CUDACHECK(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device_id));
    return sm_count;
}

// 定义一个结构体，用于表示矩阵转置的状态
struct transpose
{
    static constexpr int N = 0; // 无转置
    static constexpr int T = 1; // 转置
};

// 定义一个结构体，用于表示 tile 的行和列轴
struct axis
{
    static constexpr int ROW = 0;   // 行轴
    static constexpr int COL = 1;   // 列轴
};

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 * 
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
// 定义命名空间 ducks，主要用于模板元编程
namespace ducks{
// 一个空的默认类型结构体，用于作为占位符
struct default_type{};
// typeof 宏，用于获取变量的去除常量和引用的类型
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type
}

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 * 定义了一个常量 MASK_ALL，表示 warp 中所有线程的活跃线程掩码。
 * 这个掩码值 0xFFFFFFFF 可以用于标识 warp 中的所有线程。
 */
static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;

/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 * 对给定类型 `T` 执行同步的向下 shuffle 操作，操作在同一个 warp 内的线程之间完成。
 * `mask` 是一个表示哪些线程是活跃的掩码，`f` 是要被 shuffle 的数据，`delta` 是移动的位数。
 
功能：向左看齐，把右边同学的数字传过来。

逻辑： 每个线程向比自己编号（Lane ID）大 delta 的线程要数据。

动作： * 线程 0 拿到 线程 0 + delta 的值。

线程 1 拿到 线程 1 + delta 的值。

...

线程 31 拿不到数据（因为右边没人了），通常会返回它自己的原值或未定义值。
 */

template <typename T>
__device__ static inline T packed_shfl_down_sync(uint32_t mask,const T& f,int delta){
    return __shfl_down_sync(mask,f,delta);
}


/**
 * @brief Specialization for float2 type (two floats).
 * 对 `float2` 类型进行 shuffle down 操作的特化版本。
 * 由于 `float2` 是一个结构体，包含 `x` 和 `y` 两个浮点数，所以需要分别对每个元素进行 shuffle。
 */
template<>
__device__ inline float2 packed_shfl_down_sync<float2>(uint32_t mask,const float2& f,int delta){
    float2 r;
    r.x = __shfl_down_sync(mask,f.x,delta);
    r.y = __shfl_down_sync(mask,f.y,delta);
    return r;
}

/**
 * @brief Perform a packed shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param src[in] The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 * 对给定类型 `T` 执行同步的 shuffle 操作。该操作会在 warp 内部的线程之间进行数据交换。
 * `mask` 是活跃线程的掩码，`f` 是要 shuffle 的数据，`src` 是源线程所在的 lane ID。
 功能：广播，全体看齐某一位同学。

逻辑： 强制所有线程去拿指定某一个线程（这里是 0 号线程）手里的数据。

动作：

不论你是线程 1、线程 10 还是线程 31，执行完这一行后，你的 shuffled_from_0 变量里存的全都是线程 0 当时的 my_value。

用途： 适合做配置广播。比如线程 0 计算出了一个全局比例因子，通过这个函数，其他 31 个线程瞬间就能同步这个参数，不需要写进内存再读出来。
 
 */
template<typename T>
__device__ static inline T packed_shfl_sync(uint32_t mask,const T& f,int src){
    return __shfl_sync(mask,f,src);
}

/**
 * @brief Specialization for float2 type (two floats).
 * 对 `float2` 类型进行 shuffle 操作的特化版本，类似于 `packed_shfl_down_sync`。
 */
template<>
__device__ inline float2 packed_shfl_sync<float2>(uint32_t mask,const float2& f,int src){
    float2 r;
    r.x = __shfl_sync(mask,f.x,src);
    r.y = __shfl_sync(mask,f.y,src);
    return r;
}

// Joyously stolen from https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/include/cute/container/alignment.hpp#L51
// 这部分定义了内存对齐相关的宏和工具
#if defined(__CUDACC__)
#define DF_ALIGN_AS(n) __align__(n)     // CUDA 编译器中使用的对齐宏
#else
#define DF_ALIGN_AS(n) alignas(n)       // 非 CUDA 环境下使用 C++11 alignas 关键字
#endif

// 根据不同的硬件架构，设置默认的内存对齐值
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
#define DF_DEFAULT_ALIGN DF_ALIGN_AS(128)   // Hopper 或 Blackwell 架构使用 128 字节对齐
#else
#define DF_DEFAULT_ALIGN DF_ALIGN_AS(16)    // 默认使用 16 字节对齐
#endif

/**
 * @brief Perform a ceiling division operation.
 * @tparam T The type of the value to be divided.
 * @param a[in] numerator.
 * @param b[in] denominator.
 * @return The result of the ceiling division.
 *  执行一个向上取整的除法操作。例如，3/2 的结果为 2，而不是 1。
 */
template<typename T>
__host__ __device__ constexpr T cdiv(T a,T b){
    return (a + b - 1) / b; // 向上取整实现
}

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 * 用于内存对齐的虚拟结构体，在 WGMMA 和 TMA 操作中需要特定的对齐要求。
 */
struct DF_DEFAULT_ALIGN alignment_dummy {    int dummy;};

/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * 简单的共享内存分配器，用于动态分配内存并跟踪内存对齐。
 * 该分配器将根据模板参数 `default_alignment` 强制执行对齐。
 */
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<int default_alignment = 1024>  // Hopper 和 Blackwell 架构下，默认对齐为 1024 字节
#else
template<int default_alignment = 16>    // 其他架构下，默认对齐为 16 字节
#endif

// 结构体：共享内存分配器
// 用途：在共享内存中进行动态内存分配，支持任意维度的数组
struct shared_allocator{
    int *ptr;   // 当前共享内存指针，以int为单位进行管理

private:
    // 内部递归模板，用于生成 N 维数组类型
    // 这个模板将维度列表转换为C风格的多维数组类型
    template<typename A, size_t... dims>
    struct variadic_array;

    // 递归生成 N 维数组类型 - 递归展开部分
    // 将维度列表 first_dim, rest_dims... 转换为 A[first_dim][rest_dims...]
    template<typename A, size_t first_dim, size_t... rest_dims>
    struct variadic_array<A, first_dim, rest_dims...> {
        using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        // 例如：variadic_array<float, 2, 3> -> float[2][3]
    };

    // 递归生成 N 维数组类型 - 递归终止条件
    // 当维度列表为空时，类型就是A本身
    template<typename A>
    struct variadic_array<A> {
        using type = A;     // 基础类型
    };

    // 定义 variadic_array_t 类型别名，方便使用
    // 用法：variadic_array_t<float, 2, 3> 表示 float[2][3]
    template<typename A, size_t... dims> 
    using variadic_array_t = typename variadic_array<A, dims...>::type;
    
    // 内存对齐函数 - 确保指针满足指定的对齐要求
    // alignment: 需要的对齐字节数（必须是2的幂次）
    template<int alignment>
    __device__ inline void align_ptr(){
        if constexpr (alignment > 0){   // 编译时判断，如果alignment>0才进行对齐
            uint64_t p = reinterpret_cast<uint64_t>(ptr);// 将指针转换为整数
            if (p % alignment != 0){// 如果当前地址不是alignment的倍数
                // 计算下一个对齐的地址：
                // 公式：((p + alignment - 1) & ~(alignment - 1))
                // 这里简化计算：p + (alignment - (p % alignment))                
                ptr = (int*)(p+(alignment - (p % alignment)));  // 例如：p=10, alignment=8 → 10%8=2 → 对齐到10+(8-2)=16
            }
        }
    }
    
public:
    // 构造函数：使用现有的共享内存指针初始化分配器
    __device__ shared_allocator(int *_ptr): ptr(_ptr){} 

    /**
    * @brief 为单个对象或N维数组分配共享内存
    * @tparam A 要分配的对象类型
    * @tparam dims... N维数组的维度列表（可为空）
    * @return 返回分配到对象的引用
    * 
    * 示例：
    *   allocate<float>()          // 分配一个float
    *   allocate<float, 10>()      // 分配float[10]
    *   allocate<float, 10, 20>()  // 分配float[10][20]
    */
    template<typename A, size_t... dims> 
    __device__ inline variadic_array_t<A, dims...>& allocate() {
        // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
        align_ptr<default_alignment>(); // 使用默认对齐进行对齐

        // 计算目标类型：如果dims为空就是A，否则是A[dims...]
        using at = variadic_array_t<A, dims...>;

        // 将当前指针转换为目标类型指针
        at*p = reinterpret_cast<at*>(ptr);
        
        // 移动指针：计算需要多少int大小的内存，然后移动ptr
        // sizeof(at) / sizeof(int)：计算需要的int数量
        // 注意：这里假设sizeof(at)能被sizeof(int)整除
        ptr += sizeof(at)/sizeof(int);
        return *p;  // 返回分配对象的引用
    }


    /**
    * @brief 为单个对象或N维数组分配共享内存（带自定义对齐）
    * @tparam alignment 自定义对齐要求（字节数）
    * @tparam A 要分配的对象类型
    * @tparam dims... N维数组的维度列表
    * @return 返回分配到对象的引用
    * 
    * 这个版本允许为特定对象指定对齐要求，常用于需要特殊对齐的数据类型
    * 例如：向量加载/存储、TMA操作等
    */
    template<int alignment, typename A, size_t... dims> 
    __device__ inline variadic_array_t<A, dims...>& allocate() {
        // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
        align_ptr<alignment>();// 使用指定的对齐要求
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr); // 将指针转换为目标类型
        ptr += sizeof(at)/sizeof(int);// 更新指针
        return *p;
    }

};

// 条件编译：仅在Hopper或Blackwell架构下编译以下代码
// DF_HOPPER/DF_BLACKWELL：架构定义宏
#if (defined(DF_HOPPER) || defined(DF_BLACKWELL))
/**
 * @brief 用于TMA（Tensor Memory Access）加载/存储的分配器包装器
 * TMA操作需要1024字节对齐，这个分配器强制使用1024字节对齐
 */
using tma_allocator = shared_allocator<1024>;   // 为TMA操作特化的分配器

// TMA的swizzle模式也需要1024字节对齐
using tma_swizzle_allocator = tma_allocator;    

/* 获取集群ID - 在GPU集群中识别当前集群 */
__device__ static inline int3 clusterIdx(){
    int3 cluster_idx;       // {x, y, z}三维集群ID

    // 使用内联汇编获取集群ID寄存器值
    // %clusterid.x/.y/.z：NVidia GPU的特殊寄存器
    asm volatile("mov.u32 %0,%clusterid.x;\n" : "=r"(cluster_idx.x));
    asm volatile("mov.u32 %0,%clusterid.y;\n" : "=r"(cluster_idx.y));
    asm volatile("mov.u32 %0,%clusterid.z;\n" : "=r"(cluster_idx.z));
    return cluster_idx;
}

/* 获取集群中的CTA（Cooperative Thread Array）ID */
__device__ static inline int cluster_ctarank() {
    uint32_t ctarank;

    // 使用内联汇编获取当前CTA在集群中的排名
    // %cluster_ctarank：特殊寄存器，表示CTA在集群中的索引
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(ctarank));
    return ctarank;
}
#endif


/* 
下面四个函数共同构成了一套高效的异步流水线管理机制（Pipelining / Ring Buffer Control），常用于 CUDA 核函数中对共享内存（Shared Memory）缓冲区的多级调度。

它们的设计思想非常接近 NVIDIA 官方 libcu++ 中的 cuda::pipeline。简单来说，它们是用来解决“生产者（从内存读数据）”和“消费者（计算数据）”之间如何同步的问题。

1. get_phasebit 与 update_phasebit：相位锁机制
这两个函数通过一个 32 位的整数（bitfield）来记录多个缓冲区的“状态”。

- 设计思路：将 32 位整数拆成两个 16 位的“账本”。
    - half == 0：低 16 位，存一套环形缓冲区的状态。
    - half == 1：高 16 位，存另一套（或者是相关的配对状态）。

- 什么是“相位位”（Phase Bit）？
    - 想象一个乒乓缓存（Double Buffer），相位位就像一个开关。
    - 作用：当生产者写完一轮回到起点时，它会翻转（update）这个位。消费者通过检查（get）这个位，就能知道“这一块数据是旧的还是刚写好的”。
    - 优势：不需要庞大的数组，只用一个 uint32_t 就能同时管理多达 16 个并发执行的流水线阶段。

2. ring_advance 与 ring_retreat：环形指针定位这是标准的环形缓冲区索引计算逻辑。
- ring_advance (前进)：计算下一个存储块的位置。比如你有 4 个共享内存块，现在在第 3 块，前进 1 步就回到第 0 块。

- ring_retreat (后退)：计算上一个存储块的位置。
    - 代码中的 16*N 妙处：在处理取模运算（%）时，如果 ring - distance 结果是负数，取模会出错。
       加上 16*N（只要是 $N$ 的倍数且足够大）可以保证计算结果始终为正数，
这是一种常见的位运算/嵌入式优化技巧。

3. 综合场景：它们是怎么配合工作的？
想象你在做一个 异步数据加载：

1. 生产者（Async Copy）：从全局显存读取数据到 Shared Memory[ring]。

2. 更新相位：数据搬完了，调用 update_phasebit。这意味着“第 ring 块已经填满了最新数据”。

3. 索引推进：调用 ring_advance，准备搬下一块数据。

4. 消费者（计算内核）：在计算前，调用 get_phasebit 检查。如果相位位变了，说明数据已经到位，可以开工。


代码细节点评（避坑指南）
- if constexpr：这是一个 C++17 特性。由于 half 是模板参数，编译器在编译阶段就会直接砍掉不符合条件的分支。这意味着在 GPU 运行时完全没有 if 判断开销，性能极高。
- asm volatile ("brkpt;\n");：这是一个硬核的调试手段。如果代码逻辑跑到了不该去的分支（比如 half 传了个 2），GPU 会直接触发断点停下来，方便你用调试器查找原因。
- 位宽限制：由于使用了 16 位偏移，这套逻辑暗示了你的流水线深度（或者说环形缓冲区的个数 $N$）最大不能超过 16。
*/


/**
 * @brief 从位字段中获取相位位（phase bit）
 * @tparam half 指定使用位字段的前半部分(0)还是后半部分(1)
 * @param bitfield 32位位字段，低16位用于前半部分，高16位用于后半部分
 * @param ring_id 环标识符，用于选择位字段中的哪一位
 * @return 相位位的值（0或1）
 * 
 * 用途：在流水线或环形缓冲中跟踪阶段/相位信息
 */
template<int half>
__device__ static inline int get_phasebit(uint32_t bitfield, int ring_id) {
    if constexpr (half == 0)        // 前半部分：使用低16位
        return (bitfield >> (ring_id)) & 0b1;   // 获取第ring_id位的值
    else if constexpr (half == 1)               // 后半部分：使用高16位
        return (bitfield >> (ring_id + 16)) & 0b1;  // 获取第(ring_id+16)位的值
    else
        asm volatile ("brkpt;\n");  // 非法参数，触发断点（调试用）
    return -1;
}

/**
 * @brief 更新相位位（翻转指定位）
 * @tparam half 指定操作位字段的前半部分还是后半部分
 * @param bitfield 要更新的位字段（引用）
 * @param ring_id 环标识符，指定要翻转的位
 * 
 * 用途：在阶段推进时翻转相位位
 */
template<int half> 
__device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
    if constexpr (half == 0)
        bitfield ^= (1 << (ring_id));   // 异或操作翻转指定位
    else if constexpr (half == 1)
        bitfield ^= (1 << (ring_id + 16));
    else
        asm volatile ("brkpt;\n");
}

/**
 * @brief 环形缓冲区前进操作
 * @tparam N 环形缓冲区大小
 * @param ring 当前环索引
 * @param distance 前进的距离（默认1）
 * @return 新的环索引
 * 
 * 计算：(ring + distance) % N
 */
template<int N> __device__ static inline int ring_advance(int ring, int distance=1) { return (ring + distance) % N; }

/**
 * @brief 环形缓冲区后退操作
 * @tparam N 环形缓冲区大小
 * @param ring 当前环索引
 * @param distance 后退的距离（默认1）
 * @return 新的环索引
 * 
 * 计算：(ring + N - distance) % N
 * 注意：16*N可能是为了确保正数，但可能有笔误，应该是N
 */
template<int N> __device__ static inline int ring_retreat(int ring, int distance=1) { 
    // 注意：这里可能是16*N，但逻辑上应该是N
    // 应该是 (ring + N - distance) % N
    return (ring + 16*N - distance) % N; 
}
}


















