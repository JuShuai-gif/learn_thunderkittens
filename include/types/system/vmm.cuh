




#pragma once

#include <cuda.h>
#include <vector>
#include <stdexcept>

#include "../../common/common.cuh"

namespace kittens {
namespace detail {
namespace vmm {

// Intra-node shareable handle type
// This makes the handle shareable with cuMemExportToShareableHandle/cuMemImportFromShareableHandle
static constexpr CUmemAllocationHandleType HANDLE_TYPE = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

typedef CUmemGenericAllocationHandle handle;
/**
 * @brief 使用虚拟内存管理API分配设备内存
 * @param handle 输出参数，返回分配的内存句柄
 * @param allocated_size 输出参数，返回实际分配的大小（向上对齐到粒度）
 * @param size 请求分配的内存大小
 * @param device_id 要在哪个设备上分配内存
 * @note 使用CU_MEM_ALLOCATION_TYPE_PINNED类型，可以导出/导入到其他进程
 */
__host__ inline static void vm_alloc(
    CUmemGenericAllocationHandle *handle,
    size_t *allocated_size,
    const size_t size,
    const int device_id
) {
    CUmemAllocationProp prop = {};
    prop.location.id = device_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = HANDLE_TYPE;// 请求可共享的句柄类型
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;// 固定内存，支持DMA操作

    // 获取推荐的内存分配粒度（对齐要求）
    size_t granularity;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    *allocated_size = (size + granularity - 1) / granularity * granularity; // 向上对齐到粒度

    // 创建虚拟内存分配
    CUCHECK(cuMemCreate(handle, *allocated_size, &prop, 0));
}

/**
 * @brief 将分配的内存映射到虚拟地址空间
 * @param ptr 输出参数，返回映射后的虚拟地址
 * @param handle 要映射的内存句柄
 * @param size 要映射的大小（应与分配的大小一致）
 */
__host__ inline static void vm_map(
    void **ptr,
    const CUmemGenericAllocationHandle &handle,
    const size_t size
) {
    CUdeviceptr device_ptr;
    // 保留虚拟地址范围
    CUCHECK(cuMemAddressReserve(&device_ptr, size, 0, 0, 0));
    // 将内存句柄映射到保留的地址范围
    CUCHECK(cuMemMap(device_ptr, size, 0, handle, 0));
    *ptr = (void *)device_ptr;
}

/**
 * @brief 设置内存的访问权限，允许多个设备访问该内存
 * @param ptr 内存的虚拟地址
 * @param size 内存大小
 * @param num_devices 要授予访问权限的设备数量
 * @note 默认授予读写权限
 */
__host__ inline static void vm_set_access(
    void *ptr,
    const size_t size,
    const int num_devices
) {
    std::vector<CUmemAccessDesc> descs(num_devices);
    for (int i = 0; i < num_devices; i++) {
        descs[i].location.id = i;// 设备ID
        descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;// 读写权限
    }
    // 设置内存访问权限
    CUCHECK(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, descs.data(), num_devices));
}

/**
 * @brief 从已映射的内存中检索句柄
 * @param handle 输出参数，返回检索到的句柄
 * @param ptr 已映射内存的虚拟地址
 * @note 每次调用都需要对应的cuMemRelease来释放句柄引用计数
 */
__host__ inline static void vm_retrieve_handle(
    CUmemGenericAllocationHandle *handle,
    void *ptr
) {
    // 从映射地址检索内存分配句柄
    // 注意：每次调用此函数都需要对应的cuMemRelease来释放句柄
    CUCHECK(cuMemRetainAllocationHandle(handle, ptr));
}

/**
 * @brief 取消内存映射并释放虚拟地址范围
 * @param ptr 要取消映射的内存虚拟地址
 * @param size 要取消映射的内存大小
 */
__host__ inline static void vm_unmap(
    void *ptr,
    const size_t size
) {
    // 取消内存映射
    CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size)); 
    // 释放虚拟地址范围    
    CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
}

/**
 * @brief 释放内存句柄
 * @param handle 要释放的内存句柄
 * @note 建议尽快释放句柄；只有当所有句柄和地址映射都释放后，底层内存才会真正释放
 */
__host__ inline static void vm_free(CUmemGenericAllocationHandle &handle) {
    // 释放内存句柄，减少引用计数
    // 建议尽快释放句柄；底层内存只有当所有句柄和地址映射都释放后才会真正释放
    CUCHECK(cuMemRelease(handle));
}

/**
 * @brief 分配、映射并设置内存访问权限的便捷函数
 * @param ptr 输出参数，返回映射后的虚拟地址
 * @param allocated_size 输出参数，返回实际分配的大小
 * @param size 请求分配的内存大小
 * @param device_id 要在哪个设备上分配内存
 * @param num_devices 要授予访问权限的设备数量
 */
__host__ inline static void vm_alloc_map_set_access(
    void **ptr,
    size_t *allocated_size,
    const size_t size,
    const int device_id,
    const int num_devices
) {
    CUmemGenericAllocationHandle handle;
    vm_alloc(&handle, allocated_size, size, device_id); // 分配内存
    vm_map(ptr, handle, *allocated_size);               // 映射到虚拟地址
    vm_set_access(*ptr, *allocated_size, num_devices);  // 设置访问权限
    vm_free(handle);                                    // 立即释放句柄（不影响已映射的内存）
}

/**
 * @brief 检查设备是否支持多播
 * @param device_id 要检查的设备ID
 * @throw std::runtime_error 如果设备不支持多播
 */
__host__ inline static void multicast_check(const int device_id) {
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));

    int multicast_supported;
    CUresult result = cuDeviceGetAttribute(
        &multicast_supported,
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        device
    );

    if (!multicast_supported)
        throw std::runtime_error("Device does not support multicast");
}

/**
 * @brief 创建多播内存对象
 * @param handle 输出参数，返回多播句柄
 * @param allocated_size 输出参数，返回实际分配的大小（向上对齐到粒度）
 * @param size 请求分配的内存大小
 * @param num_devices 要绑定到多播对象的设备数量
 * @note 创建后，句柄需要通过MPI、KittensBroker等方式共享给所有进程
 */
__host__ inline static void multicast_create_handle(
    CUmemGenericAllocationHandle *handle,
    size_t *allocated_size,
    const size_t size,
    const int num_devices
) {
    if (num_devices <= 1)
        throw std::runtime_error("Multicast requires at least 2 devices");

    CUmulticastObjectProp prop = {};
    prop.numDevices = num_devices;
    prop.handleTypes = HANDLE_TYPE;
    
    // 获取多播内存的分配粒度
    size_t granularity;
    CUCHECK(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    *allocated_size = (size + granularity - 1) / granularity * granularity;
    prop.size = *allocated_size;

    // 创建多播对象
    // 此后，句柄必须通过MPI、KittensBroker等方式共享给所有进程
    cuMulticastCreate(handle, &prop);
}

/**
 * @brief 将设备绑定到多播对象
 * @param handle 多播句柄
 * @param device_id 要绑定的设备ID
 * @note 所有进程必须在此操作后同步，然后才能绑定内存
 */
__host__ inline static void multicast_bind_device(
    const CUmemGenericAllocationHandle &handle,
    const int device_id
) {
    // 将设备添加到多播组
    // 所有进程必须在此操作后同步，然后才能绑定内存
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));
    CUCHECK(cuMulticastAddDevice(handle, device));
}

/**
 * @brief 将内存绑定到多播对象
 * @param multicast_handle 多播句柄
 * @param memory_handle 要绑定的内存句柄
 * @param size 要绑定的内存大小
 * @note 所有进程应在调用此函数前完成设备添加
 */
__host__ inline static void multicast_bind_memory(
    const CUmemGenericAllocationHandle &multicast_handle,
    const CUmemGenericAllocationHandle &memory_handle,
    const size_t size
) {
    // 将内存绑定到多播对象
    // 所有进程应在调用此函数前完成设备添加
    CUCHECK(cuMulticastBindMem(multicast_handle, 0, memory_handle, 0, size, 0));
}

/**
 * @brief 将已映射的内存地址绑定到多播对象
 * @param multicast_handle 多播句柄
 * @param ptr 已映射内存的虚拟地址
 * @param size 要绑定的内存大小
 * @note 所有进程应在调用此函数前完成设备添加
 */
__host__ inline static void multicast_bind_address(
    const CUmemGenericAllocationHandle &multicast_handle,
    void *ptr,
    const size_t size
) {
    // 从映射地址检索句柄，然后绑定到多播对象
    // 所有进程应在调用此函数前完成设备添加
    CUmemGenericAllocationHandle memory_handle;
    vm_retrieve_handle(&memory_handle, ptr);// 获取内存句柄
    multicast_bind_memory(multicast_handle, memory_handle, size);// 绑定到多播
    vm_free(memory_handle);// 释放临时句柄
}

/**
 * @brief 从多播对象解绑设备
 * @param handle 多播句柄
 * @param size 绑定的大小
 * @param device_id 要解绑的设备ID
 * @note 不需要显式解绑内存，解绑设备会自动处理
 */
__host__ inline static void multicast_unbind_device(
    const CUmemGenericAllocationHandle &handle,
    const size_t size,
    const int device_id
) {
    // 从多播组中移除设备
    // 不需要显式解绑内存
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));
    CUCHECK(cuMulticastUnbind(handle, device, 0, size));
}

} // namespace vmm
} // namespace detail
} // namespace kittens




















