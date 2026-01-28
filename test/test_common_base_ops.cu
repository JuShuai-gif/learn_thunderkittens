#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <assert.h>

#include "../include/common/base_ops.cuh"

using namespace kittens;

// 辅助宏：用于检查 GPU 上的结果并打印
#define CUDA_CHECK_RESULT(name, expected, actual) \
    if (fabsf((float)actual - (float)expected) > 0.001f) { \
        printf("[FAIL] %s: Expected %f, Got %f\n", name, (float)expected, (float)actual); \
    } else { \
        printf("[PASS] %s\n", name); \
    }

__global__ void test_ops_kernel(float* results) {
    // --- 1. 常量算子测试
    float z = kittens::base_ops::zero::op<float>();
    float o = kittens::base_ops::zero::op<float>();
    CUDA_CHECK_RESULT("zero::op", 0.0f, z);
    CUDA_CHECK_RESULT("one::op", 1.0f, o);    

    // --- 3. exp 算子测试 (half / bf16) ---
    float f_val = 2.0f;
    float f_exp = kittens::base_ops::exp::op<float>(f_val);
    printf("f_exp:%f\n",f_exp);

    float2 f2_val = make_float2(2.3f,3.4f);
    float2 f2_exp = kittens::base_ops::exp::op<float2>(f2_val);
    printf("f2_exp: x %f,y %f\n",f2_exp.x,f2_exp.y);

    bf16 bf_val = __float2bfloat16(2.0f);
    bf16 bf_exp = kittens::base_ops::exp::op<bf16>(bf_val);
    printf("bf_exp: bf_exp: %f\n",__bfloat162float(bf_exp));

    half h_val = __float2half(2.0f);
    half h_exp = kittens::base_ops::exp::op<half>(h_val);      // 调用 hexp
    CUDA_CHECK_RESULT("exp::half", 7.389f, __half2float(h_exp));

    // 1. 测试 Unary Ops: exp
    float val = 1.0f;
    results[0] = kittens::base_ops::exp::op<float>(val); // 应接近 2.718

    // 2. 测试 Binary Ops: sum
    float a = 10.5f, b = 5.5f;
    results[1] = kittens::base_ops::sum::op<float>(a, b); // 应为 16.0

    // 3. 测试 float2 (模拟向量化操作)
    float2 a2 = make_float2(1.0f, 2.0f);
    float2 b2 = make_float2(3.0f, 4.0f);
    float2 c2 = kittens::base_ops::sum::op<float2>(a2, b2);
    results[2] = c2.x; // 4.0
    results[3] = c2.y; // 6.0

    // 4. 测试 Ternary Ops: FMA (a*b + c)
    results[4] = kittens::base_ops::fma_AxBtC::op<float>(2.0f, 3.0f, 4.0f); // 2*3 + 4 = 10.0
    
    // 5. 测试 Relu
    results[5] = kittens::base_ops::relu::op<float>(-5.0f); // 应为 0.0


}

void run_test() {
    float *d_res, h_res[10];
    cudaMalloc(&d_res, 10 * sizeof(float));

    test_ops_kernel<<<1, 1>>>(d_res);

    cudaMemcpy(h_res, d_res, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "--- Test Results ---" << std::endl;
    std::cout << "Exp(1.0): " << h_res[0] << " (Expected: ~2.718)" << std::endl;
    std::cout << "Sum(10.5, 5.5): " << h_res[1] << " (Expected: 16.0)" << std::endl;
    std::cout << "Float2 Sum: (" << h_res[2] << ", " << h_res[3] << ") (Expected: 4.0, 6.0)" << std::endl;
    std::cout << "FMA(2*3+4): " << h_res[4] << " (Expected: 10.0)" << std::endl;
    std::cout << "ReLU(-5.0): " << h_res[5] << " (Expected: 0.0)" << std::endl;

    cudaFree(d_res);
}

int main() {
    run_test();
    return 0;
}











