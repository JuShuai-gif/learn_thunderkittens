#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

struct exp
{
    template<typename T> 
    static __device__ inline T op(const T& x)
    {
        return exp(x);
    }
};

// Template specializations for specific types
template<> __device__ inline float exp::op<float>(const float &x) { 
    return __expf(x); 
}

template<> __device__ inline float2 exp::op<float2>(const float2 &x) { 
    return float2{__expf(x.x), __expf(x.y)}; 
}

template<> __device__ inline __nv_bfloat16 exp::op<__nv_bfloat16>(const __nv_bfloat16 &x) { 
    return hexp(x);  // For bfloat16
}

template<> __device__ inline __nv_bfloat162 exp::op<__nv_bfloat162>(const __nv_bfloat162 &x) { 
    return h2exp(x); // For bfloat16_2
}

template<> __device__ inline __half exp::op<__half>(const __half &x) { 
    return hexp(x);  // For __half
}

template<> __device__ inline __half2 exp::op<__half2>(const __half2 &x) { 
    return h2exp(x); // For __half2
}

// CUDA kernel to test exp operations
__global__ void test_exp_operations()
{
    // Test with float
    float f = 2.0f;
    printf("exp(float): %f\n", exp::op(f));

    // Test with float2 (two floats)
    float2 f2 = {1.0f, 2.0f};
    printf("exp(float2): (%f, %f)\n", exp::op(f2).x, exp::op(f2).y);

    // Test with __nv_bfloat16
    __nv_bfloat16 bf = __float2bfloat16(1.0f);  // Convert float to __nv_bfloat16
    printf("exp(__nv_bfloat16): %f\n", __bfloat162float(hexp(bf)));

    // Test with __nv_bfloat16_2 (two __nv_bfloat16 values)
    __nv_bfloat162 bf2 = {__float2bfloat16(1.0f), __float2bfloat16(2.0f)};  // Convert float to __nv_bfloat16
    printf("exp(__nv_bfloat16_2): (%f, %f)\n", __bfloat162float(hexp(bf2.x)), __bfloat162float(hexp(bf2.y)));

    // Test with half
    __half h = __float2half(1.0f);  // Convert float to half
    printf("exp(half): %f\n", __half2float(hexp(h)));

    // Test with half_2 (two half values)
    __half2 h2 = __floats2half2_rn(1.0f, 2.0f);  // Convert two floats to half_2
    printf("exp(half_2): (%f, %f)\n", __half2float(hexp(h2.x)), __half2float(hexp(h2.y)));
}

int main()
{
    // Launch the CUDA kernel to test exp operations
    test_exp_operations<<<1, 1>>>();

    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize the device to ensure all printf outputs are completed
    cudaDeviceSynchronize();

    return 0;
}
