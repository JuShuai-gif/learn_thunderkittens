#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../../common/common.cuh"
#include "vmm.cuh"

namespace kittens {
namespace ducks {
namespace ipc {
namespace handle {

// 定义标识符结构体，用于标识IPC句柄
struct identifier {};

// 模板约束，检查T类型是否有成员 identifier 且类型为identifier
template<typename T> concept all = requires {
    typename T::identifier; // 检查是否有成员 identifier
} && std::is_same_v<typename T::identifier, identifier>;    // 检查成员类型是否为 identifier

} // namespace handle
} // namespace ipc
} // namespace ducks

namespace detail {
namespace ipc {

// 枚举类型flavor，用于区分不同类型的IPC句柄
enum flavor {
    LEGACY = 0,     // 旧版IPC句柄
    VMM = 1         // 虚拟内存管理IPC句柄
};

// 针对不同flavor定义不同的句柄结构体
template<flavor _flavor>    // 根据不同的flavor类型，定义对应的结构体
struct handle;

// 对于flavor::LEGACY类型的IPC句柄结构体
template<> 
struct handle<flavor::LEGACY> {
    using identifier = ducks::ipc::handle::identifier;  // 定义identifier类型
    static constexpr flavor flavor_ = flavor::LEGACY;   // 设置句柄类型为LEGACY
    cudaIpcMemHandle_t handle_ {};                      // CUDA内存句柄，初始化为空
};

// 对于flavor::VMM类型的IPC句柄结构体
template<>
struct handle<flavor::VMM> {
    using identifier = ducks::ipc::handle::identifier;  // 定义identifier类型
    static constexpr flavor flavor_ = flavor::VMM;      // 设置句柄类型为VMM
    int handle_;    // 用于存储句柄（文件描述符）
};

// 检查设备是否支持CUDA IPC
__host__ inline static void check_support(const int device_id) {
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));   // 获取CUDA设备

    int ipc_supported = 0;
    // 检查设备是否支持IPC事件
    CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, device_id));
    int ipc_handle_supported = 0;
    // 检查设备是否支持POSIX文件描述符类型的IPC句柄
    CUCHECK(cuDeviceGetAttribute(&ipc_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));

    // 如果设备不支持IPC或IPC句柄，抛出异常
    if (!ipc_supported || !ipc_handle_supported)
        throw std::runtime_error("CUDA IPC is not supported on this device");
}

// 导出IPC句柄到共享内存
template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void export_handle(
    IPC_HANDLE *ipc_handle, // IPC句柄
    void *ptr               // 内存指针
) {
    // 根据IPC句柄类型执行不同的导出操作
    if constexpr (IPC_HANDLE::flavor_ == flavor::LEGACY) {
        CUDACHECK(cudaIpcGetMemHandle(&ipc_handle->handle_, ptr));  // 导出内存句柄（Legacy）
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUmemGenericAllocationHandle memory_handle;
        detail::vmm::vm_retrieve_handle(&memory_handle, ptr);       // 获取VMM句柄
        // 重要：这个句柄（文件描述符）必须由用户手动关闭
        CUCHECK(cuMemExportToShareableHandle(&ipc_handle->handle_, memory_handle, detail::vmm::HANDLE_TYPE, 0));
        detail::vmm::vm_free(memory_handle);    // 释放VMM句柄
    } else {
        throw std::runtime_error("Invalid IPC handle type");    // 无效的IPC句柄类型
    }
}

// 导出IPC句柄到共享内存（使用内存句柄作为参数）
template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void export_handle(
    IPC_HANDLE *ipc_handle,                         // IPC句柄
    CUmemGenericAllocationHandle &memory_handle     // 内存句柄
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUCHECK(cuMemExportToShareableHandle(&ipc_handle->handle_, memory_handle, detail::vmm::HANDLE_TYPE, 0));
    } else {
        throw std::runtime_error("Invalid IPC handle type");    // 无效的IPC句柄类型
    }
}

// 导入IPC句柄到内存（通过指针参数）
template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void import_handle (
    void **ptr,             // 内存指针
    IPC_HANDLE &ipc_handle, // IPC句柄
    const size_t size,      // 内存大小
    int local_world_size    // 本地世界大小
) {
    // 根据IPC句柄类型执行不同的导入操作
    if constexpr (IPC_HANDLE::flavor_ == flavor::LEGACY) {
        CUDACHECK(cudaIpcOpenMemHandle(ptr, ipc_handle.handle_, cudaIpcMemLazyEnablePeerAccess)); // this is the only flag supported
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUmemGenericAllocationHandle memory_handle;
        CUCHECK(cuMemImportFromShareableHandle(&memory_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.handle_)), detail::vmm::HANDLE_TYPE));
        detail::vmm::vm_map(ptr, memory_handle, size);      // 映射内存
        detail::vmm::vm_set_access(*ptr, size, local_world_size);   // 设置内存访问
        detail::vmm::vm_free(memory_handle);                // 释放内存句柄
        close(ipc_handle.handle_);  // 关闭文件描述符
        ipc_handle.handle_ = -1;    // 设置句柄为空
    } else {
        throw std::runtime_error("Invalid IPC handle type");    // 无效的IPC句柄类型
    }
}


// 导入IPC句柄到内存（通过内存句柄参数）
template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void import_handle (
    CUmemGenericAllocationHandle *memory_handle,    // 内存句柄
    IPC_HANDLE &ipc_handle,                         // IPC句柄
    const size_t size,                              // 内存大小
    int local_world_size                            // 本地世界大小
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUCHECK(cuMemImportFromShareableHandle(memory_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.handle_)), detail::vmm::HANDLE_TYPE));
        close(ipc_handle.handle_);  // 关闭文件描述符
        ipc_handle.handle_ = -1;    // 设置句柄为空
    } else {
        throw std::runtime_error("Invalid IPC handle type");    // 无效的IPC句柄类型
    }
}

// 释放IPC句柄对应的内存
template<flavor _flavor>
__host__ inline static void free_handle(
    void *ptr,          // 内存指针
    const size_t size   // 内存大小
) {
    if constexpr (_flavor == flavor::LEGACY) {
        CUDACHECK(cudaIpcCloseMemHandle(ptr));      // 关闭内存句柄（Legacy）
    } else if constexpr (_flavor == flavor::VMM) {
        detail::vmm::vm_unmap(ptr, size);           // 解除内存映射
    } else {
        throw std::runtime_error("Invalid IPC handle type");    // 无效的IPC句柄类型
    }
}

} // namespace ipc
} // namespace detail
} // namespace kittens