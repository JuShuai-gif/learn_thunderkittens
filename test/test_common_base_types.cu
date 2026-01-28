#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../include/common/base_types.cuh"

using namespace kittens;
using namespace kittens::base_types;
using namespace kittens::ducks::base_types;


template<typename T>
__device__ const char* get_type_name();

template<> __device__ const char* get_type_name<float>(){
    return "float";
}

template<> __device__ const char* get_type_name<float2>(){
    return "float2";
}

template<> __device__ const char* get_type_name<bf16>(){
    return "bf16";
}

template<> __device__ const char* get_type_name<half>(){
    return "half";
}

template<> __device__ const char* get_type_name<bf16_2>(){
    return "bf16_2";
}

template<> __device__ const char* get_type_name<half_2>(){
    return "half_2";
}



template<typename T>
__global__ void test_constants_kernel() {
    T zero = constants<T>::zero();
    T one  = constants<T>::one();
    T pinf = constants<T>::pos_infty();
    T ninf = constants<T>::neg_infty();

    printf("Constants test for type %s\n", get_type_name<T>());
#if defined(__CUDA_ARCH__)
    // GPU 上简单打印数值
#endif
}

// 通用 convertor 测试 kernel
template<typename T, typename U>
__global__ void test_convertor_kernel(const U u) {
    T t = convertor<T,U>::convert(u);

#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same<T,float>::value) {
        printf("Convertor result (float): %f\n", t);
    } else if constexpr (std::is_same<T,half>::value) {
        printf("Convertor result (half): %f\n", __half2float(t));
    } else if constexpr (std::is_same<T,bf16>::value) {
        printf("Convertor result (bf16): %f\n", __bfloat162float(t));
    } else if constexpr (std::is_same<T,float2>::value) {
        printf("Convertor result (float2): [%f, %f]\n", t.x, t.y);
    } else if constexpr (std::is_same<T,half_2>::value) {
        float2 f2 = __half22float2(t);
        printf("Convertor result (half2): [%f, %f]\n", f2.x, f2.y);
    } else if constexpr (std::is_same<T,bf16_2>::value) {
        float2 f2 = __bfloat1622float2(t);
        printf("Convertor result (bf16_2): [%f, %f]\n", f2.x, f2.y);
    }
#endif
}

template<typename T>
__global__ void test_packing_kernel(const typename packing<T>::unpacked_type val) {
    auto p = packing<T>::pack(val);
    printf("Packing test: num=%d\n", packing<T>::num());
}



int main() {
    // 1. 测试 constants
    std::cout << "=== Testing constants ===" << std::endl;

    test_constants_kernel<float><<<1,1>>>();
    test_constants_kernel<float2><<<1,1>>>();
    test_constants_kernel<half><<<1,1>>>();
    test_constants_kernel<half_2><<<1,1>>>();
    test_constants_kernel<bf16><<<1,1>>>();
    test_constants_kernel<bf16_2><<<1,1>>>();

    // 2. 测试 packing
    std::cout << "=== Testing packing ===" << std::endl;

    test_packing_kernel<float><<<1,1>>>(1.23f);
    test_packing_kernel<float2><<<1,1>>>(1.23f);
    test_packing_kernel<half><<<1,1>>>(__float2half(1.23f));
    test_packing_kernel<half_2><<<1,1>>>(__float2half(1.23f));
    test_packing_kernel<bf16><<<1,1>>>(__float2bfloat16_rn(1.23f));
    test_packing_kernel<bf16_2><<<1,1>>>(__float2bfloat16(1.23f));

    // 3. 测试 convertor
    std::cout << "=== Testing convertor ===" << std::endl;

    // bf16 <-> float
    // rn 表示在进行浮点运算时，使用的舍入方式是“四舍五入到最近的偶数”
    test_convertor_kernel<float,bf16><<<1,1>>>(__float2bfloat16_rn(1.23f));
    test_convertor_kernel<bf16,float><<<1,1>>>(1.23f);

    // half <-> float
    test_convertor_kernel<float,half><<<1,1>>>(__float2half(1.23f));
    test_convertor_kernel<half,float><<<1,1>>>(1.23f);

    // bf16_2 <-> float2
    test_convertor_kernel<float2,bf16_2><<<1,1>>>(__float22bfloat162_rn(make_float2(1.23f, 4.56f)));
    test_convertor_kernel<bf16_2,float2><<<1,1>>>(make_float2(1.23f, 4.56f));

    // half_2 <-> float2
    test_convertor_kernel<float2,half_2><<<1,1>>>(__float22half2_rn(make_float2(1.23f, 4.56f)));
    test_convertor_kernel<half_2,float2><<<1,1>>>(make_float2(1.23f, 4.56f));

    // bf16 <-> half
    test_convertor_kernel<bf16,half><<<1,1>>>(__float2half(1.23f));
    test_convertor_kernel<half,bf16><<<1,1>>>(__float2bfloat16_rn(1.23f));

    // bf16_2 <-> half_2
    test_convertor_kernel<bf16_2,half_2><<<1,1>>>(__float22half2_rn(make_float2(1.23f, 4.56f)));
    test_convertor_kernel<half_2,bf16_2><<<1,1>>>(__float22bfloat162_rn(make_float2(1.23f, 4.56f)));

    cudaDeviceSynchronize();

    return 0;
}





















