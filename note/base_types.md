# 1. concept
```c++
template<typename T>
concept T2 = std::is_same_v<T,float2> || std::is_same_v<T,bf16_2> || std::is_same_v<T,half_2>;
// 标量数据类型
template <typename T>
concept T1 = std::is_same_v<T,float> || std::is_same_v<T,bf16> || std::is_same_v<T,half>;
```
上面这段代码的使用场景，本质上是编译器类型系统做接口约束，不是运行时判断

上面的语义非常清晰：
- T1 → 标量数值类型
- T2 → SIMD/向量/packed 类型
**它们的用途是**：在模板层面限制类型合法性 + 编译期分派不同实现路径。

## 一、最直接的用法(模板参数约束)
### 用在函数模板参数上
```C++
template<T1 T>
__device__ inline T scalar_add(T a, T b) {
    return a + b;
}

template<T2 T>
__device__ inline T vector_add(T a, T b) {
    return a + b;   // float2 / half2 / bf16_2 的向量加法
}
```
调用时：
```C++
float a,b;
scalar_add(a,b);     // ✅

float2 v1,v2;
vector_add(v1,v2);   // ✅

int x,y;
scalar_add(x,y);     // ❌ 编译期直接报错（不满足 T1）
```
**非法类型直接编译失败，不进函数体（零 runtime 成本）**

## 二、算子泛型接口(自动分派 T1/T2)
```C++
template<typename T>
__device__ inline T add(T a, T b) {
    if constexpr (T1<T>) {
        // 标量路径
        return a + b;
    } else if constexpr (T2<T>) {
        // packed / SIMD 路径
        return a + b;
    } else {
        static_assert(T1<T> || T2<T>, "Unsupported type");
    }
}
```
使用：
```C++
add<float>(a,b);     // 走 T1 分支
add<float2>(v1,v2);  // 走 T2 分支
```
👉 这是编译期多态（static polymorphism）
👉 无分支指令、无 runtime overhead

## 三、在kernel模板中使用(GPU算子统一接口)
```C++
template<T1 T>
__global__ void kernel_scalar(const T* a, const T* b, T* out) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}

template<T2 T>
__global__ void kernel_vector(const T* a, const T* b, T* out) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];  // 向量并行
}
```
调用：
```C++
kernel_scalar<float><<<1,32>>>(...);
kernel_vector<float2><<<1,32>>>(...);
```
👉 数据布局不同 → kernel 结构相同 → 类型系统区分执行语义

## 四、用于 traits/算子系统(kittens的核心用途)
典型的设计模式是这样的：
```C++
template<typename T>
struct Op;

template<T1 T>
struct Op<T> {
    static __device__ inline T compute(T x) {
        // 标量算子路径
        return x * x;
    }
};

template<T2 T>
struct Op<T> {
    static __device__ inline T compute(T x) {
        // 向量算子路径
        return x * x;  // SIMD
    }
};
```
使用：
```C++
Op<float>::compute(x);
Op<float2>::compute(v);
```
👉 算子泛型化 + 数据布局抽象
👉 这正是 算子库 / kernel 模板系统 / AI kernel 框架 的典型设计

## 五、和 packing / constants / convertor 的协同关系
在你现在这套体系中：
```
T1 = 逻辑数值类型（scalar semantic）
T2 = 数据布局类型（packed layout）

constants<T>  → 数值语义
packing<T>    → 内存布局语义
convertor<A,B>→ 精度语义
```
组合就是完整抽象层：
```
算法层：    matmul / softmax / attention / norm
语义层：    float语义 / bf16语义 / fp8语义
布局层：    scalar / packed2 / packed4
后端层：    warp / shared / tensorcore
```
这就是算子级抽象设计（Operator Abstraction Architecture）。

## 六、没有concept的写法
```C++
template<typename T>
__device__ inline T add(T a, T b) {
    static_assert(
        std::is_same_v<T,float> ||
        std::is_same_v<T,half>  ||
        std::is_same_v<T,bf16>,
        "unsupported type"
    );
    return a + b;
}
```
现在：
```C++
template<T1 T>
__device__ inline T add(T a, T b) {
    return a + b;
}
```
这两个 concept 的作用不是“判断类型”，而是：

> 用 C++ 类型系统在编译期建立 数据语义层级模型把“数值语义 / 数据布局 / 算法路径选择”从 runtime 转移到 compile-time








