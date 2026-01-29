#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>

#include "kittens.cuh"
#include "parallel_tensor.cuh"

// 定义检查CUDA张量的宏：验证输入张量是否在CUDA设备上
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// 定义检查张量连续性的宏：验证输入张量是否是连续存储的
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// 组合宏：同时检查CUDA设备和连续性
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace kittens {
namespace py {

// 概念（C++20特性）：检查配置结构体是否包含MIN_BLOCKS_PER_SM成员
// 用于模板元编程，确定配置是否指定了每个SM的最小块数
template <typename Config>
concept has_min_blocks_per_sm = requires { std::integral_constant<int, int(Config::MIN_BLOCKS_PER_SM)>{}; };

// 编译时函数：获取配置的最小块数，如果配置未定义则返回默认值1
template <typename Config>
consteval int min_blocks_per_sm() {
    if constexpr(has_min_blocks_per_sm<Config>)
        return Config::MIN_BLOCKS_PER_SM;
    else
        return 1;
}

// 非聚集内核启动包装器：为不使用集群特性的内核提供启动边界优化
// __launch_bounds__：CUDA编译器指令，优化寄存器使用和线程块调度
template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
void global_kernel_unclustered(const __grid_constant__ Globals G) {
    Kernel(G);  // 调用实际的内核函数
}

// 聚集内核启动包装器：为使用集群特性的内核提供启动边界优化
// __cluster_dims__：CUDA 9.0+指令，定义集群维度（多个线程块协作）
template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
__cluster_dims__(Config::CLUSTER_SIZE)
void global_kernel_clustered(const __grid_constant__ Globals G) {
    Kernel(G);  // 调用实际的内核函数
}

// 张量检查函数：验证PyTorch张量是否符合特定布局要求
// Layout：模板参数，定义期望的数据类型和布局
// TypeCheck：布尔模板参数，控制是否进行类型检查
template <typename Layout, bool TypeCheck = true>
static inline void tensor_check(const at::Tensor &t) {
    // 基础检查：CUDA设备、连续性、维度
    TORCH_CHECK(t.is_cuda(), "Tensor must be on CUDA device")
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous")
    TORCH_CHECK(t.dim() <= 4, "Expected Tensor.dim() <= 4");

    // 如果禁用类型检查，直接返回
    if constexpr (!TypeCheck) {
        return;
    } 
    // 根据Layout指定的数据类型进行类型检查
    // 检查整数类型
    else if constexpr (std::is_same_v<typename Layout::dtype, char>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Char, "Tensor has invalid dtype (expected int8)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, short>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Short, "Tensor has invalid dtype (expected int16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, int>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Int, "Tensor has invalid dtype (expected int32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, long>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Long, "Tensor has invalid dtype (expected int64)");
    // 检查浮点类型，包括特殊浮点格式
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e4m3>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "Tensor has invalid dtype (expected fp8e4m3)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e5m2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e5m2, "Tensor has invalid dtype (expected fp8e5m2)");
#endif
#ifdef DF_BLACKWELL
// Blackwell架构特有的浮点格式
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e8m0>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e8m0fnu || t.dtype() == at::ScalarType::Byte, "Tensor has invalid dtype (expected fp8e8m0)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp4e2m1_2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Tensor has invalid dtype (expected fp4_2)");
#endif
    // 检查标准浮点类型
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::bf16>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::BFloat16, "Tensor has invalid dtype (expected bfloat16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::half>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Half, "Tensor has invalid dtype (expected float16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, float>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float, "Tensor has invalid dtype (expected float32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, double>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Double, "Tensor has invalid dtype (expected float64)");
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

// 并行张量检查函数：验证并行张量（跨多个GPU）的一致性
// PGL：并行全局布局类型
template <kittens::ducks::pgl::all PGL, bool TypeCheck = true>
static inline void parallel_tensor_check(const TKParallelTensor& t) {
    // 基础张量检查
    tensor_check<PGL, TypeCheck>(t.data_);
    // 验证形状一致性
    TORCH_CHECK(t.data_.sizes().vec() == t.shape_, "Shape mismatch between TKParallelTensor and the underlying tensor");
    // 验证数据类型一致性
    TORCH_CHECK(t.data_.dtype() == t.dtype_, "Dtype mismatch between TKParallelTensor and the underlying tensor");
    // 验证设备数量一致性
    TORCH_CHECK(t.raw_ptrs_.size() == PGL::num_devices, "Number of devices mismatch between PGL and TKParallelTensor");
    // 验证本地秩一致性
    TORCH_CHECK(t.local_rank_ == t.data_.device().index(), "Current tensor device index mismatch within TKParallelTensor");
    // 验证本地世界大小一致性
    TORCH_CHECK(t.local_world_size_ == PGL::num_devices, "Number of devices mismatch between PGL and TKParallelTensor");
    // 验证多播配置一致性
    TORCH_CHECK(t.multicast_ == PGL::multicast, "Multicast mismatch between PGL and TKParallelTensor");
    // 验证数据指针一致性
    TORCH_CHECK(t.raw_ptrs_[t.local_rank_] == reinterpret_cast<void *>(t.data_.data_ptr()), "Current tensor data pointer not found in TKParallelTensor's raw_ptrs_");
}

// 将PyTorch张量转换为全局布局（GL）对象
// GL：全局布局类型
template <kittens::ducks::gl::all GL, bool TypeCheck = true>
static inline GL tensor_to_gl(const at::Tensor &t) {
    tensor_check<GL, TypeCheck>(t);
    // 将张量形状转换为4D数组（补1以适应4D布局）
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.dim()); ++i)
        shape[4 - t.dim() + i] = static_cast<int>(t.size(i));

    uint64_t data_ptr = reinterpret_cast<uint64_t>(t.data_ptr());
    // 创建全局布局对象
    return ::kittens::make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
}

// 带显式形状参数的张量转全局布局函数
template <kittens::ducks::gl::all GL, bool TypeCheck = true>
static inline GL tensor_to_gl(const at::Tensor &t, int B, int D, int R, int C) {
    tensor_check<GL, TypeCheck>(t);

    return ::kittens::make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), B, D, R, C);
}

// 将并行张量转换为并行全局布局（PGL）对象
template <kittens::ducks::pgl::all PGL, bool TypeCheck = true>
static inline PGL parallel_tensor_to_pgl(TKParallelTensor &t) {
    parallel_tensor_check<PGL, TypeCheck>(t);
    // 获取张量形状
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < static_cast<int>(t.data_.dim()); ++i) {
        shape[4 - t.data_.dim() + i] = static_cast<int>(t.data_.size(i));
    }
    // 根据是否支持多播选择不同的构造函数
    if constexpr (PGL::multicast)
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t>(t.multicast_ptr_), reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), shape[0], shape[1], shape[2], shape[3]);
    else
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), shape[0], shape[1], shape[2], shape[3]);
}

// 带显式形状参数的并行张量转并行全局布局函数
template <kittens::ducks::pgl::all PGL, bool TypeCheck = true>
static inline PGL parallel_tensor_to_pgl(TKParallelTensor &t, int B, int D, int R, int C) {
    parallel_tensor_check<PGL, TypeCheck>(t);

    if constexpr (PGL::multicast)
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t>(t.multicast_ptr_), reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), B, D, R, C);
    else
        return ::kittens::make_pgl<PGL>(
            reinterpret_cast<uint64_t *>(t.raw_ptrs_.data()), B, D, R, C);
}

// 创建假全局布局对象（用于测试或占位）
template <kittens::ducks::gl::all GL>
static inline GL make_fake_gl(const int batch, const int depth, const int rows, const int cols) {
    return ::kittens::make_gl<GL>(reinterpret_cast<uint64_t>(nullptr), batch, depth, rows, cols);
}

// 辅助函数：检查两个张量是否在同一设备上
static inline void _device_check(const at::Tensor& first, const at::Tensor& second) {
    TORCH_CHECK(first.device() == second.device(), "All tensors must be on the same device");
}

// 可变参数模板函数：检查多个张量是否在同一设备上
template <typename T1, typename... Ts>
static inline void device_check(const T1& first, const Ts&... rest) {
    (_device_check(first, rest), ...);  // C++17折叠表达式
}

// 辅助函数：检查两个并行张量的一致性
static inline void _parallel_tensor_check(const TKParallelTensor& first, const TKParallelTensor& second) {
    TORCH_CHECK(first.local_rank_ == second.local_rank_, "All parallel tensors must have the same local_rank");
    TORCH_CHECK(first.local_world_size_ == second.local_world_size_, "All parallel tensors must have the same local_world_size");
}

// 可变参数模板函数：检查多个并行张量的一致性
template <typename T1, typename... Ts>
static inline void parallel_tensor_check(const T1& first, const Ts&... rest) {
    (_parallel_tensor_check(first, rest), ...); // C++17折叠表达式
}

// 概念：检查配置是否有静态网格大小
template <typename Config>
concept static_grid = requires { Config::NUM_BLOCKS; };

// 概念：检查配置是否有静态块大小
template <typename Config>
concept static_block = requires { Config::NUM_THREADS; };

// 概念：检查配置是否有动态共享内存大小
template <typename Config>
concept static_dynamic_shared_memory = requires { Config::DYNAMIC_SHARED_MEMORY; };

// 内核启动函数：根据配置启动CUDA内核
template <typename Config, typename Globals, auto Kernel>
static inline void launch_kernel(const Globals &G) {
    dim3 grid;
    // 如果配置定义了静态网格大小，使用静态值
    if constexpr (static_grid<Config>)
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();    // 否则从全局对象获取

    dim3 block;
    // 如果配置定义了静态块大小，使用静态值
    if constexpr (static_block<Config>)
        block = dim3{Config::NUM_THREADS, 1, 1};
    else
        block = G.block();  // 否则从全局对象获取

    int dynamic_shared_memory;
    // 如果配置定义了动态共享内存，使用静态值
    if constexpr (static_dynamic_shared_memory<Config>)
        dynamic_shared_memory = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();// 否则从全局对象获取

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();// 获取当前CUDA流
    
    // 根据集群大小选择内核启动方式
    if constexpr (Config::CLUSTER_SIZE <= 1) {
        // 设置非聚集内核的动态共享内存属性
        CUDACHECK(cudaFuncSetAttribute(global_kernel_unclustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        // 启动非聚集内核        
        global_kernel_unclustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    } else {
        // 架构特定的集群大小限制检查
#if defined(DF_HOPPER)
        static_assert(Config::CLUSTER_SIZE <= 8, "Cluster size must be less than or equal to 8 for Hopper");
#elif defined(DF_BLACKWELL)
        static_assert(Config::CLUSTER_SIZE <= 16, "Cluster size must be less than or equal to 16 for Blackwell");
        // Blackwell架构特定：对于大于8的集群，需要设置特殊属性
        if constexpr (Config::CLUSTER_SIZE > 8)
            CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
#endif
        // 设置聚集内核的动态共享内存属性
        CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        // 启动聚集内核
        global_kernel_clustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    }
}

} // namespace py
} // namespace kittens
