#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic Python list -> std::vector conversion

namespace kittens {
namespace py {

/*******************************************************
 * from_object<T>
 *
 * 作用：
 *   统一定义「如何从 pybind11::object 构造 C++ 类型 T」
 *
 * 设计思想：
 *   - 默认情况：直接 obj.cast<T>()
 *   - 特殊类型（如 GL Tensor）：需要手写解析逻辑
 *******************************************************/
template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        // 默认行为：让 pybind11 负责类型转换
        return obj.cast<T>();
    }
};

/*******************************************************
 * from_object<GL> 的偏特化
 *
 * 约束：
 *   GL 必须满足 ducks::gl::all concept
 *
 * 用途：
 *   - 接收 Python 侧传入的 torch.Tensor
 *   - 校验其合法性
 *   - 提取 shape / data_ptr
 *   - 构造底层 GL 对象（CUDA 侧使用）
 *******************************************************/
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // ---------- 1. 判断是否为 torch.Tensor ----------
        // 这里不直接 include torch headers，而是用 Python 反射
        if (pybind11::hasattr(obj, "__class__") && 
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {
        
            // ---------- 2. contiguous 校验 ----------
            // CUDA kernel 通常假定线性内存
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }

            // ---------- 3. 设备校验 ----------
            // 这里只允许 CUDA Tensor（GPU Tensor）
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }
            
            // ---------- 4. shape 提取 ----------
            // GL 侧统一使用 4D 形状（NCHW / NHWC 等）
            // 若 Tensor 维度 < 4，则前面补 1
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }

            // 从后向前填充
            // e.g. Tensor shape = (H, W)
            // => shape = {1, 1, H, W}
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }
            
            // ---------- 5. 获取底层 CUDA 指针 ----------
            // torch.Tensor.data_ptr() → uint64
            // 本质是 device pointer
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            
            // ---------- 6. 构造 GL 对象 ----------
            // make_gl 是你自己的工厂函数
            // GL 通常是「轻量 wrapper，不 owning 内存」
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        // 不是 Tensor，直接报错
        throw std::runtime_error("Expected a torch.Tensor");
    }
};

/*******************************************************
 * trait：成员指针萃取器
 *
 * 目的：
 *   从 `MT T::*` 中提取：
 *     - 成员类型 MT
 *     - 所属类型 T
 *
 * 这是 bind_kernel 的核心元编程工具
 *******************************************************/
template<typename> struct trait;

// pybind11::object 的别名（为了语义清晰）
template<typename> using object = pybind11::object;

// 针对「成员指针类型」的偏特化
template<typename MT, typename T> struct trait<MT T::*> { 
    using member_type = MT;     // 成员的类型
    using type = T;             // 所属类
};

/*******************************************************
 * has_dynamic_shared_memory concept
 *
 * 用于在编译期判断：
 *   TGlobal 是否定义了：
 *     int dynamic_shared_memory();
 *
 * 若存在：
 *   - kernel launch 前设置 MaxDynamicSharedMemory
 *******************************************************/
template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };

/*******************************************************
 * bind_kernel
 *
 * 功能：
 *   - 将 CUDA kernel 绑定为 Python 函数
 *   - 自动完成：
 *       Python args → TGlobal 构造
 *       stream 提取
 *       kernel<<<grid, block, smem, stream>>>
 *
 * 模板参数：
 *   kernel    : CUDA __global__ kernel
 *   TGlobal   : kernel 参数结构体
 *
 * member_ptrs：
 *   - 只是“参数类型列表描述器”
 *   - 不访问成员，仅用于推导类型
 *******************************************************/
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args, pybind11::kwargs kwargs) {

        // ---------- 1. 构造 TGlobal ----------
        // 每个 args[i] 对应 member_ptrs[i] 的 member_type
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};

        // ---------- 2. CUDA stream 处理 ----------
        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            // PyTorch stream → cudaStream_t
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }

        // ---------- 3. kernel launch ----------
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();

            // 设置 kernel 最大可用动态 shared memory
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block(), 0, raw_stream>>>(__g__);
        }
    });
}

/*******************************************************
 * bind_function
 *
 * 与 bind_kernel 的区别：
 *   - 绑定的是普通 C++ 函数（host function）
 *   - 不涉及 CUDA launch / stream
 *******************************************************/
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}

} // namespace py
} // namespace kittens
