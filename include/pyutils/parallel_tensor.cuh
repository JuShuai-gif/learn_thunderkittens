#pragma once

#include <iostream>
#include <map>
#include <vector>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include "../types/system/vmm.cuh"
#include "../types/system/ipc.cuh"
#include "broker.cuh"

namespace kittens {
namespace py {

/**
 * @brief 分布式张量包装器，用于多GPU IPC共享和组播。
 *        可以在内核调用前轻松创建PGL（并行组）。
 *        设计为每个线程/进程使用单个对象。
 */
struct TKParallelTensor {
    // 静态成员：懒加载初始化的KittensBroker映射表
    // 键为(local_rank, local_world_size)，值为对应的KittensBroker实例
    inline static std::map<std::pair<int, int>, KittensBroker> brokers_; // lazily initialized
    
    // PyTorch张量，用于直接从PyTorch访问
    at::Tensor data_; // for direct access from PyTorch
    // 张量形状（维度大小）
    std::vector<int64_t> shape_;
    // 张量数据类型（如float32, int64等）
    at::ScalarType dtype_;

    // 存储所有进程的原始指针（用于IPC访问）
    std::vector<void *> raw_ptrs_;
    // 分配的内存大小（字节）
    size_t allocated_size_;
    
    // 本地进程排名（通常与设备索引相同）
    int local_rank_; // identical to device index
    // 本地进程总数
    int local_world_size_;

    // 是否启用组播模式
    bool multicast_;
    // 组播使用的指针（用于多进程共享同一块内存）
    void *multicast_ptr_;
    // 组播分配的内存大小
    size_t multicast_allocated_size_;
    
    // IPC共享内存的风格（LEGACY传统方式或VMM虚拟内存管理）
    detail::ipc::flavor ipc_flavor_;

    /**
     * @brief 构造函数1：从已有PyTorch张量创建TKParallelTensor
     * @param tensor 现有的PyTorch CUDA张量
     * @param local_rank 本地进程排名
     * @param local_world_size 本地进程总数
     * @param multicast 是否启用组播（此构造函数中不支持）
     * 
     * 说明：此构造函数用于包装已分配的PyTorch张量，适合已有数据需要共享的场景
     */
    __host__ inline TKParallelTensor(
        const at::Tensor &tensor,
        int local_rank,
        int local_world_size,
        bool multicast
    ) : data_(tensor),
        shape_(tensor.sizes().vec()),   // 获取张量形状并转换为vector
        dtype_(tensor.scalar_type()),   // 获取数据类型
        raw_ptrs_(local_world_size, nullptr),   // 初始化指针数组，大小为进程数
        allocated_size_(tensor.nbytes()),       // 计算张量占用的字节数
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(detail::ipc::flavor::LEGACY) {  // 使用传统IPC方式

        // 参数检查和验证
        TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device"); // 必须是CUDA张量
        TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");// 必须是连续内存布局
        TORCH_CHECK(tensor.dim() <= 4, "Only tensors with dim <= 4 are supported for TKParallelTensor");// 仅支持最多4维
        TORCH_CHECK(tensor.device().index() == local_rank_, "Tensor device index must match local_rank");// 设备索引必须匹配本地排名
        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");
        TORCH_CHECK(!multicast, "Multicast is not supported for pre-allocated tensors");// 预分配张量不支持组播

        // 懒初始化KittensBroker
        // try_emplace: 如果键不存在则插入，存在则不做操作        
        brokers_.try_emplace(
            {local_rank_, local_world_size_},   // 键：(local_rank, local_world_size)
            local_rank_, local_world_size_      // 值：新创建的KittensBroker
        );
        
        // 警告：同一进程中有多个KittensBroker实例可能不安全
        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 KittensBroker instances created in the same process. This is not safe." << std::endl;
        
        // 设置CUDA设备上下文
        c10::cuda::CUDAGuard device_guard(local_rank_);
        // 交换IPC句柄（使用传统IPC风格）
        exchange_ipc_handles<detail::ipc::flavor::LEGACY>();
    }

    /**
     * @brief 构造函数2：根据形状和数据类型创建新的TKParallelTensor
     * @param shape 张量形状
     * @param dtype 张量数据类型
     * @param local_rank 本地进程排名
     * @param local_world_size 本地进程总数
     * @param multicast 是否启用组播
     * 
     * 说明：此构造函数会分配新的共享内存，适合从头创建共享张量的场景
     */
    __host__ inline TKParallelTensor(
        const std::vector<int64_t> &shape,
        const at::ScalarType dtype,
        int local_rank,
        int local_world_size,
        bool multicast
    ) : shape_(shape),
        dtype_(dtype),
        raw_ptrs_(local_world_size, nullptr),   // 初始化指针数组
        allocated_size_(0),                     // 初始分配大小为0，将在create_shareable_cuda_tensor中设置
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(detail::ipc::flavor::VMM) { // 使用VMM（虚拟内存管理）IPC方式

        // 参数检查
        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");
        
        // 懒初始化KittensBroker
        brokers_.try_emplace(
            {local_rank_, local_world_size_},
            local_rank_, local_world_size_
        );
        
        // 警告检查
        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 KittensBroker instances created in the same process. This is not safe." << std::endl;
        // 设置CUDA设备上下文
        c10::cuda::CUDAGuard device_guard(local_rank_);
        // 创建可共享的CUDA张量（分配共享内存）
        create_shareable_cuda_tensor();
        // 交换IPC句柄（使用VMM风格）
        exchange_ipc_handles<detail::ipc::flavor::VMM>();
        // 如果启用组播，初始化组播
        if (multicast_)
            initialize_multicast();
    }
    // 禁止拷贝构造函数和拷贝赋值运算符（避免资源重复释放）
    TKParallelTensor(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(TKParallelTensor&& other) = delete;
    
    /**
     * @brief 移动构造函数
     * @param other 被移动的源对象
     * 
     * 说明：支持移动语义，将资源所有权从一个对象转移到另一个对象
     */
    __host__ inline TKParallelTensor(TKParallelTensor&& other) :
        data_(std::move(other.data_)),
        shape_(std::move(other.shape_)),
        dtype_(std::move(other.dtype_)),
        raw_ptrs_(std::move(other.raw_ptrs_)),
        allocated_size_(other.allocated_size_),
        local_rank_(other.local_rank_),
        local_world_size_(other.local_world_size_),
        multicast_(other.multicast_),
        multicast_ptr_(other.multicast_ptr_),
        multicast_allocated_size_(other.multicast_allocated_size_),
        ipc_flavor_(other.ipc_flavor_) {
        // 清空源对象状态，防止资源被重复释放
        other.data_ = at::Tensor();
        other.shape_.clear();
        other.dtype_ = at::ScalarType::Undefined;
        other.raw_ptrs_.clear();
        other.allocated_size_ = 0;
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
        other.multicast_ = false;
        other.multicast_ptr_ = nullptr;
        other.multicast_allocated_size_ = 0;
    }
    /**
     * @brief 析构函数
     */
    __host__ inline ~TKParallelTensor() {
        destroy();  // 清理资源
    }
    
    /**
     * @brief 获取内部的PyTorch张量
     * @return at::Tensor 返回PyTorch张量
     */
    __host__ inline at::Tensor data() const {
        return data_;
    }
    
    /**
     * @brief 创建可共享的CUDA张量
     * 
     * 说明：使用VMM（虚拟内存管理）分配可在多进程间共享的内存，
     * 并创建一个PyTorch张量包装这些内存。
     */
    __host__ inline void create_shareable_cuda_tensor() {
        // 设置CUDA设备上下文
        c10::cuda::CUDAGuard device_guard(local_rank_);
        // 参数检查
        TORCH_CHECK(!shape_.empty(), "Shape must be non-empty");
        TORCH_CHECK(shape_.size() <= 4, "Shape must have at most 4 dimensions for TKParallelTensor");
        // 计算所需内存大小
        size_t size = c10::elementSize(dtype_); // 获取数据类型的大小（字节）
        for (auto dim : shape_) {
            TORCH_CHECK(dim > 0, "Size dimensions must be positive");
            size *= static_cast<size_t>(dim);   // 累计计算总字节数
        }
        // 使用VMM分配可共享的内存
        void *raw_ptr;
        detail::vmm::vm_alloc_map_set_access(
            &raw_ptr, &allocated_size_, size, local_rank_, local_world_size_);

        // 创建局部变量用于lambda捕获（避免捕获this指针）
        int local_rank = local_rank_;
        size_t allocated_size = allocated_size_;
        
        // 自定义删除器，当PyTorch张量不再需要时释放VMM内存
        auto deleter = [local_rank, raw_ptr, allocated_size](void* p) mutable {
            if (!p) return; // 空指针检查
            // 设置设备上下文
            c10::cuda::CUDAGuard device_guard(local_rank);
            // 同步当前CUDA流，确保所有操作完成
            auto stream = c10::cuda::getCurrentCUDAStream().stream();

            CUDACHECK(cudaStreamSynchronize(stream));
            // 释放VMM映射的内存
            detail::vmm::vm_unmap(raw_ptr, allocated_size);
        };
        // 配置PyTorch张量选项
        at::TensorOptions options = at::TensorOptions()
            .dtype(dtype_)                      // 设置数据类型
            .device(at::kCUDA, local_rank_);    // 设置设备为CUDA和对应的设备索引
        // 从原始指针创建PyTorch张量，并指定自定义删除器
        data_ = at::from_blob(raw_ptr, shape_, std::move(deleter), options);
    }


    template <detail::ipc::flavor IPC_FLAVOR>
    __host__ inline void exchange_ipc_handles() {
        // 根据 IPC flavor（LEGACY / VMM）确定对应的 handle 类型
        using handle_t = detail::ipc::handle<IPC_FLAVOR>;

        /* =========================================================
       1. 导出本 rank 上 GPU 内存的 IPC handle
       ========================================================= */

        // 检查当前 rank / 设备是否支持所选 IPC 方式
        detail::ipc::check_support(local_rank_);
        // 获取当前 tensor 对应的原始 GPU 指针
        void *raw_ptr = reinterpret_cast<void *>(data_.data_ptr());
        // 用于保存导出的 IPC handle
        handle_t ipc_handle;
        
        // 从原始 GPU 指针导出 IPC handle
        // LEGACY: cudaIpcMemHandle_t
        // VMM   : CUDA 虚拟内存 / FD
        detail::ipc::export_handle(&ipc_handle, raw_ptr);

        /* =========================================================
       2. 通过 broker 在所有 local ranks 间交换 IPC handle
       ========================================================= */

        // 保存所有 rank 的 IPC handle（每个 rank 一个）
        std::vector<handle_t> all_ipc_handles(local_world_size_);
        if constexpr (IPC_FLAVOR == detail::ipc::flavor::LEGACY) {
            // LEGACY IPC：
            //  - handle 是一段普通内存结构
            //  - 直接通过共享内存 / socket 交换字节
            brokers_.at({local_rank_, local_world_size_}).exchange_data(
                reinterpret_cast<void *>(all_ipc_handles.data()),
                reinterpret_cast<void *>(&ipc_handle),
                sizeof(handle_t)
            );
        } else if constexpr (IPC_FLAVOR == detail::ipc::flavor::VMM) {
            // VMM IPC：
            //  - handle 内部本质是一个文件描述符（FD）
            //  - 必须使用 sendmsg / SCM_RIGHTS 方式传递
            brokers_.at({local_rank_, local_world_size_}).exchange_fds(
                reinterpret_cast<int *>(all_ipc_handles.data()),
                ipc_handle.handle_
            );
        } else {
            throw std::runtime_error("Invalid IPC flavor");
        }

        /* =========================================================
       3. 从其他 ranks 的 IPC handle 中导入 GPU 指针
       ========================================================= */
        for (int i = 0; i < local_world_size_; i++) {
            if (i == local_rank_)
                // 自己的 rank 直接使用原始指针
                raw_ptrs_[i] = raw_ptr;
            else
                // 通过 IPC handle 导入其他 rank 的 GPU 内存
                // 生成一个可访问的 GPU 虚拟地址
                detail::ipc::import_handle(&raw_ptrs_[i], all_ipc_handles[i], allocated_size_, local_world_size_);
        }
    }

    __host__ inline void initialize_multicast() {
        // multicast 仅支持 VMM IPC
        using handle_t = detail::ipc::handle<detail::ipc::flavor::VMM>;

        /* =========================================================
       1. 环境与能力检查
       ========================================================= */

        // 检查当前设备是否支持 VMM multicast
        detail::vmm::multicast_check(local_rank_);
        // 检查 IPC 支持情况
        detail::ipc::check_support(local_rank_);
        // multicast 虚拟内存句柄
        detail::vmm::handle multicast_handle;

            /* =========================================================
       2. rank 0 创建 multicast handle，并广播给其他 ranks
       ========================================================= */
        if (local_rank_ == 0) {
            // ⚠️ 只有 rank 0 负责创建 multicast handle
            detail::vmm::multicast_create_handle(
                &multicast_handle,
                &multicast_allocated_size_,
                allocated_size_,
                local_world_size_
            );

            // 当前实现要求 multicast 分配大小与原 tensor 大小完全一致
            if (allocated_size_ != multicast_allocated_size_)
                throw std::runtime_error("Multicast allocated size does not match memory allocated size");

            // 将 multicast handle 导出为 IPC handle（FD）
            handle_t ipc_handle;
            detail::ipc::export_handle(&ipc_handle, multicast_handle);

            // 通过 broker 向所有 ranks 广播 FD
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(nullptr, ipc_handle.handle_, 0);
        } else {
            // 非 rank 0：从 rank 0 接收 multicast IPC handle
            handle_t ipc_handle;
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(&ipc_handle.handle_, -1, 0);
            multicast_allocated_size_ = allocated_size_;
            // 通过 IPC handle 导入 multicast 虚拟内存
            detail::ipc::import_handle(&multicast_handle, ipc_handle, multicast_allocated_size_, local_world_size_);
        }

        /* =========================================================
       3. 将所有 GPU device 绑定到 multicast handle
       ========================================================= */

        // 将当前 rank 对应的 GPU 加入 multicast 组
        detail::vmm::multicast_bind_device(multicast_handle, local_rank_);

        // 必须等待所有 ranks 都完成 bind
        brokers_.at({local_rank_, local_world_size_}).sync(); // must ensure all devices are added

        /* =========================================================
       4. 将原始 GPU 内存绑定到 multicast handle
       ========================================================= */
        detail::vmm::handle memory_handle;

        // 从本 rank 的 raw_ptr 中获取对应的 VMM handle
        detail::vmm::vm_retrieve_handle(&memory_handle, raw_ptrs_[local_rank_]);
        // 将实际物理内存绑定到 multicast 虚拟地址空间
        detail::vmm::multicast_bind_memory(multicast_handle, memory_handle, allocated_size_);

        // 确保所有 ranks 完成内存绑定
        brokers_.at({local_rank_, local_world_size_}).sync();

        /* =========================================================
       5. 映射 multicast 虚拟地址并设置访问权限
       ========================================================= */

        // 将 multicast handle 映射为可访问的 GPU 虚拟地址
        detail::vmm::vm_map(&multicast_ptr_, multicast_handle, multicast_allocated_size_);

        // 设置所有 GPU rank 对该地址的访问权限
        detail::vmm::vm_set_access(multicast_ptr_, multicast_allocated_size_, local_world_size_);

        /* =========================================================
       6. 释放中间 handle（地址映射仍然有效）
       ========================================================= */
        detail::vmm::vm_free(multicast_handle);
        detail::vmm::vm_free(memory_handle);
    }

    __host__ inline void destroy() {
        /* =========================================================
       1. multicast 资源清理
       ========================================================= */
        if (multicast_ && multicast_ptr_) {
            // 确保所有 ranks 停止使用 multicast 内存
            brokers_.at({local_rank_, local_world_size_}).sync();
            detail::vmm::handle multicast_handle;
            // 从 multicast 虚拟地址中反查 handle
            detail::vmm::vm_retrieve_handle(&multicast_handle, multicast_ptr_);
            // 取消虚拟地址映射
            detail::vmm::vm_unmap(multicast_ptr_, multicast_allocated_size_);
            // 将当前 device 从 multicast 组中移除
            detail::vmm::multicast_unbind_device(multicast_handle, multicast_allocated_size_, local_rank_);
            // 等待所有 ranks 完成解绑
            brokers_.at({local_rank_, local_world_size_}).sync();
            // 释放 multicast handle
            detail::vmm::vm_free(multicast_handle);
        }

        /* =========================================================
       2. 通过 IPC 导入的 GPU 指针清理
       ========================================================= */
        for (int i = 0; i < local_world_size_; i++) {
            if (i != local_rank_ && i < raw_ptrs_.size()) {
                if (ipc_flavor_ == detail::ipc::flavor::LEGACY) {
                    // 释放 LEGACY IPC 映射
                    detail::ipc::free_handle<detail::ipc::flavor::LEGACY>(raw_ptrs_[i], allocated_size_);
                } else if (ipc_flavor_ == detail::ipc::flavor::VMM) {
                    // 释放 VMM IPC 映射
                    detail::ipc::free_handle<detail::ipc::flavor::VMM>(raw_ptrs_[i], allocated_size_);
                } else {
                    throw std::runtime_error("Invalid IPC flavor");
                }
            }
        }
        // 必须在 tensor 销毁前完成所有 IPC 解绑
        brokers_.at({local_rank_, local_world_size_}).sync(); // must sync before destroying the tensor

        // 3. Tensor 本身的释放
        if (data_.defined())
            data_.reset(); // 正确减少引用计数，释放 GPU 内存

        // 4. 成员变量复位
        shape_.clear();
        dtype_ = at::ScalarType::Undefined;
        raw_ptrs_.clear();
        allocated_size_ = 0;
        local_rank_ = -1;
        local_world_size_ = -1;
        multicast_ = false;
        multicast_ptr_ = nullptr;
        multicast_allocated_size_ = 0;
    }
};

} // namespace py
} // namespace kittens

// 定义一个用于 pybind11 绑定的宏
// 目的：将 C++ 中的 TKParallelTensor 类暴露为 Python 类
// 使用宏的原因：
//   - 便于在多个 pybind module 中复用
//   - 避免重复书写冗长的 pybind11 绑定代码
#define BIND_TK_PARALLEL_TENSOR(m) \
    pybind11::class_<kittens::py::TKParallelTensor>(m, "TKParallelTensor") \
            /* ------------------------------------------------------------------ */\
        /* 构造函数 1：从已有的 PyTorch Tensor 构造                           */\
        /*                                                                      */\
        /* Python 侧签名：                                                      */\
        /*   TKParallelTensor(tensor, local_rank, local_world_size, multicast) */\
        /*                                                                      */\
        /* 语义说明：                                                           */\
        /*   - tensor            : 已存在的 CUDA Tensor                         */\
        /*   - local_rank        : 当前进程在本节点内的 rank                    */\
        /*   - local_world_size  : 本节点内参与进程数（GPU 数）                 */\
        /*   - multicast         : 是否启用 CUDA VMM multicast                  */\
        /*                                                                      */\
        /* 典型用途：                                                           */\
        /*   - 多进程共享已有模型参数 / buffer                                  */\
        /*   - torch.distributed + fork / spawn 模式                            */\
        /* ------------------------------------------------------------------ */\
        .def(pybind11::init<const at::Tensor&, int, int, bool>(), \
             pybind11::arg("tensor"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
                /* ------------------------------------------------------------------ */\
        /* 构造函数 2：从 shape + dtype 新建一个 CUDA Tensor                    */\
        /*                                                                      */\
        /* Python 侧签名：                                                      */\
        /*   TKParallelTensor(shape, dtype, local_rank, local_world_size,       */\
        /*                    multicast)                                        */\
        /*                                                                      */\
        /* 语义说明：                                                           */\
        /*   - shape             : Tensor 形状                                  */\
        /*   - dtype             : PyTorch 标量类型（如 torch.float16）         */\
        /*   - local_rank        : 当前进程在本节点内的 rank                    */\
        /*   - local_world_size  : 本节点内参与进程数                            */\
        /*   - multicast         : 是否启用 CUDA VMM multicast                  */\
        /*                                                                      */\
        /* 典型用途：                                                           */\
        /*   - 创建新的共享参数 buffer                                          */\
        /*   - 模型初始化阶段                                                   */\
        /* ------------------------------------------------------------------ */\
        .def(pybind11::init<const std::vector<int64_t>&, const at::ScalarType&, int, int, bool>(), \
             pybind11::arg("shape"), \
             pybind11::arg("dtype"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        /* ------------------------------------------------------------------ */\
        /* 成员函数绑定                                                        */\
        /* ------------------------------------------------------------------ */\
                                                                                \
        /* 返回底层 Tensor（通常用于算子调用或调试） */                         \
        .def("data", &kittens::py::TKParallelTensor::data) \
        /* ------------------------------------------------------------------ */\
        /* 只读成员变量暴露（Python 侧不可修改）                               */\
        /* ------------------------------------------------------------------ */\
                                                                                \
        /* 底层实际持有的 PyTorch Tensor */                                     \
        .def_readonly("data_", &kittens::py::TKParallelTensor::data_) \
        /* 当前进程在本节点中的 rank */                                         \
        .def_readonly("local_rank_", &kittens::py::TKParallelTensor::local_rank_) \
        /* 本节点内参与通信的进程总数 */                                       \
        .def_readonly("local_world_size_", &kittens::py::TKParallelTensor::local_world_size_)