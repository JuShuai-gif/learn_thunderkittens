// test_kittens.cu
#include "../include/common/util.cuh"  // 假设上述代码保存在这个头文件中
#include <iostream>
#include <cassert>

#define DF_ADA;

// 测试常量定义和类型特性
__global__ void test_constants_and_types() {
    if(threadIdx.x == 0){
    printf("[Test 1] Constants and Types:\n");
    
    // 测试瓦片维度
    printf("  BASE_TILE_DIM = %d\n", kittens::BASE_TILE_DIM);
    
    // 测试不同类型对应的瓦片列维度
    printf("  TILE_COL_DIM<int> = %d\n", kittens::TILE_COL_DIM<int>);
    printf("  TILE_COL_DIM<short> = %d\n", kittens::TILE_COL_DIM<short>);
    printf("  TILE_COL_DIM<char> = %d\n", kittens::TILE_COL_DIM<char>);
    
    // 测试瓦片元素数量
    printf("  TILE_ELEMENTS<int> = %d\n", kittens::TILE_ELEMENTS<int>);
    
    // 测试线程组织
    printf("  WARP_THREADS = %d\n", kittens::WARP_THREADS);
    printf("  WARPGROUP_THREADS = %d\n", kittens::WARPGROUP_THREADS);
    printf("  WARPGROUP_WARPS = %d\n", kittens::WARPGROUP_WARPS);
    }
    // 测试线程ID获取函数
    printf("  Thread %d: warp_id=%d, warpgroup_id=%d, lane_id=%d\n", 
           threadIdx.x, 
           kittens::warpid(), 
           kittens::warpgroupid(), 
           kittens::laneid());
}

// 测试shuffle操作
__global__ void test_shuffle_operations() {
    int lane_id = kittens::laneid();
    
    // 测试float类型的shuffle
    float my_value = lane_id + 0.5f;
    float shuffled_down = kittens::packed_shfl_down_sync(kittens::MASK_ALL, my_value, 1);
    // warp中的所有线程去向 线程16拿数据
    float shuffled_from_0 = kittens::packed_shfl_sync(kittens::MASK_ALL, my_value, 16);
    
    printf("[Test 2] Shuffle Operations (Thread %d):\n", threadIdx.x);

    printf("  Lane %d: my_value=%.1f, shuffled_down=%.1f, shuffled_from_0=%.1f\n", 
           lane_id, my_value, shuffled_down, shuffled_from_0);
    
    // 测试float2类型的shuffle
    if (lane_id < 32) {  // 只在第一个warp中测试
        float2 my_vec2 = {static_cast<float>(lane_id), static_cast<float>(lane_id) * 2.0f};
        float2 shuffled_vec2 = kittens::packed_shfl_down_sync(kittens::MASK_ALL, my_vec2, 2);
        
        printf("  Lane %d: vec2=(%.1f, %.1f), shuffled_vec2=(%.1f, %.1f)\n",
               lane_id, my_vec2.x, my_vec2.y, shuffled_vec2.x, shuffled_vec2.y);
    }
}

// 测试共享内存分配器
__global__ void test_shared_allocator() {
    // 声明动态共享内存
    extern __shared__ int shared_mem[];
    
    // 创建分配器
    kittens::shared_allocator<> allocator(shared_mem);
    
    printf("[Test 3] Shared Allocator (Thread %d in Warp %d):\n", threadIdx.x, kittens::warpid());
    
    // 只在第一个线程中进行分配操作
    if (threadIdx.x == 0) {
        // 测试分配单个变量
        int& single_int = allocator.allocate<int>();
        single_int = 42;
        printf("  Allocated single int: %d\n", single_int);
        
        // 测试分配一维数组
        int (&array_1d)[10] = allocator.allocate<int, 10>();
        for (int i = 0; i < 10; i++) {
            array_1d[i] = i * 2;
        }
        printf("  Allocated 1D array[10], sum = %d\n", 
               array_1d[0] + array_1d[9]);
        
        // 测试分配二维数组
        float (&array_2d)[5][8] = allocator.allocate<float, 5, 8>();
        array_2d[2][3] = 3.14f;
        printf("  Allocated 2D array[5][8], value at [2][3] = %.2f\n", array_2d[2][3]);
        
        // 测试带对齐的分配
        auto& aligned_data = allocator.allocate<16, int4>();
        aligned_data.x = 1;
        aligned_data.y = 2;
        aligned_data.z = 3;
        aligned_data.w = 4;
        printf("  Allocated aligned int4: (%d, %d, %d, %d)\n", 
               aligned_data.x, aligned_data.y, aligned_data.z, aligned_data.w);
    }
    
    __syncthreads();
}

// 测试相位位和环形缓冲区操作
__global__ void test_phasebits_and_ring() {
    __shared__ uint32_t phase_bitfield;
    
    // 初始化相位位字段
    if (threadIdx.x == 0) {
        phase_bitfield = 0;
    }
    __syncthreads();
    
    printf("[Test 4] Phase Bits and Ring Buffer (Thread %d):\n", threadIdx.x);
    
    // 只在第一个warp的第一个线程中测试
    if (kittens::laneid() == 0 && kittens::warpid() == 0) {
        int ring_id = 3;
        
        // 测试获取相位位
        int phase_bit = kittens::get_phasebit<0>(phase_bitfield, ring_id);
        printf("  Initial phase bit for ring %d: %d\n", ring_id, phase_bit);
        
        // 测试更新相位位
        kittens::update_phasebit<0>(phase_bitfield, ring_id);
        phase_bit = kittens::get_phasebit<0>(phase_bitfield, ring_id);
        printf("  After flip, phase bit for ring %d: %d\n", ring_id, phase_bit);
        
        // 再次翻转应该回到0
        kittens::update_phasebit<0>(phase_bitfield, ring_id);
        phase_bit = kittens::get_phasebit<0>(phase_bitfield, ring_id);
        printf("  After second flip, phase bit for ring %d: %d\n", ring_id, phase_bit);
    }
    
    // 测试环形缓冲区操作
    if (kittens::laneid() == 1 && kittens::warpid() == 0) {
        const int N = 8;
        int current_pos = 2;
        
        int next_pos = kittens::ring_advance<N>(current_pos, 3);
        int prev_pos = kittens::ring_retreat<N>(current_pos, 2);
        
        printf("  Ring buffer size %d, current position %d:\n", N, current_pos);
        printf("    After advancing 3: %d\n", next_pos);
        printf("    After retreating 2: %d\n", prev_pos);
        
        // 测试边界条件
        int wrap_around = kittens::ring_advance<N>(N-1, 2);
        printf("    Wrap-around from %d + 2 = %d (should be 1)\n", N-1, wrap_around);
    }
}

// 测试Hopper/Blackwell特定功能
#if (defined(DF_HOPPER) || defined(DF_BLACKWELL))
__global__ void test_hopper_features() {
    printf("[Test 5] Hopper/Blackwell Specific Features:\n");
    
    // 测试TMA分配器
    extern __shared__ int tma_shared_mem[];
    kittens::tma_allocator tma_allocator(tma_shared_mem);
    
    if (threadIdx.x == 0) {
        printf("  Using TMA allocator with 1024-byte alignment\n");
        
        // TMA分配器应该使用1024字节对齐
        auto& tma_buffer = tma_allocator.allocate<float, 256>();
        printf("  Allocated TMA buffer of size %ld bytes\n", sizeof(tma_buffer));
    }
    
    // 测试集群功能（如果可用）
    if (threadIdx.x == 0) {
        int3 cluster_idx = kittens::clusterIdx();
        int cta_rank = kittens::cluster_ctarank();
        
        printf("  Cluster ID: (%d, %d, %d)\n", cluster_idx.x, cluster_idx.y, cluster_idx.z);
        printf("  CTA rank in cluster: %d\n", cta_rank);
    }
}
#endif

// 测试cdiv函数
__global__ void test_cdiv_function() {
    printf("[Test 6] Ceiling Division:\n");
    
    // 每个线程测试不同的值
    int thread_value = threadIdx.x + 1;
    int result = kittens::cdiv(thread_value, 4);
    
    printf("  Thread %d: cdiv(%d, 4) = %d\n", threadIdx.x, thread_value, result);
    
    // 测试一些边界情况
    if (threadIdx.x == 0) {
        printf("  Edge cases:\n");
        printf("    cdiv(7, 3) = %d (expected 3)\n", kittens::cdiv(7, 3));
        printf("    cdiv(8, 4) = %d (expected 2)\n", kittens::cdiv(8, 4));
        printf("    cdiv(0, 5) = %d (expected 0)\n", kittens::cdiv(0, 5));
    }
}

// 主机端测试函数
void host_test_functions() {
    std::cout << "\n[Host Test] Host-side Functions:\n";
    
    // 测试cdiv在主机端
    std::cout << "  cdiv(10, 3) = " << kittens::cdiv(10, 3) << std::endl;
    std::cout << "  cdiv(100, 32) = " << kittens::cdiv(100, 32) << std::endl;
    
    // 测试SM数量查询
    int sm_count = kittens::num_sms();
    std::cout << "  Number of SMs on current device: " << sm_count << std::endl;
    
    // 测试最大共享内存
    #if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    std::cout << "  MAX_SHARED_MEMORY (Hopper/Blackwell): " 
              << kittens::MAX_SHARED_MEMORY << " bytes" << std::endl;
    #elif defined(DF_AMPERRE)
    std::cout << "  MAX_SHARED_MEMORY (Amperre): " 
              << kittens::MAX_SHARED_MEMORY << " bytes" << std::endl;
    #else
    std::cout << "  MAX_SHARED_MEMORY (Default): not defined" << std::endl;
    #endif
}

// 主测试函数
void run_all_tests() {
    std::cout << "=== Starting ThunderKittens Tests ===\n" << std::endl;
    
    // 运行主机端测试
    host_test_functions();
    
    // 设置CUDA设备
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\n[Device Info] Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    
    // 测试1：常量和类型
    std::cout << "\n--- Running Test 1: Constants and Types ---" << std::endl;
    test_constants_and_types<<<1, 64>>>();
    cudaDeviceSynchronize();
    
    // 测试2：Shuffle操作
    std::cout << "\n--- Running Test 2: Shuffle Operations ---" << std::endl;
    test_shuffle_operations<<<1, 32>>>();  // 一个warp
    cudaDeviceSynchronize();
    
    // 测试3：共享内存分配器
    std::cout << "\n--- Running Test 3: Shared Allocator ---" << std::endl;
    // 计算所需的共享内存大小
    size_t shared_mem_size = 1024;  // 1KB共享内存用于测试
    test_shared_allocator<<<1, 32, shared_mem_size>>>();
    cudaDeviceSynchronize();
    
    // 测试4：相位位和环形缓冲区
    std::cout << "\n--- Running Test 4: Phase Bits and Ring Buffer ---" << std::endl;
    test_phasebits_and_ring<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // 测试5：Hopper/Blackwell特定功能
    #if (defined(DF_HOPPER) || defined(DF_BLACKWELL))
    std::cout << "\n--- Running Test 5: Hopper/Blackwell Features ---" << std::endl;
    test_hopper_features<<<1, 32, 2048>>>();
    cudaDeviceSynchronize();
    #else
    std::cout << "\n--- Skipping Test 5 (Not Hopper/Blackwell) ---" << std::endl;
    #endif
    
    // 测试6：cdiv函数
    std::cout << "\n--- Running Test 6: Ceiling Division ---" << std::endl;
    test_cdiv_function<<<1, 16>>>();
    cudaDeviceSynchronize();
    
    // 检查是否有CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\n[ERROR] CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "\n=== All Tests Completed Successfully ===" << std::endl;
}

// 简化测试：只测试核心功能
void run_basic_test() {
    std::cout << "=== Running Basic Test ===\n" << std::endl;
    
    // 只测试核心功能
    test_constants_and_types<<<1, 64>>>();
    test_shuffle_operations<<<1, 32>>>();
    test_phasebits_and_ring<<<1, 32>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "\nBasic test passed!" << std::endl;
    }
}

int main(int argc, char** argv) {
    // 检查CUDA设备可用性
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // 设置设备
    cudaSetDevice(0);
    
    // 根据命令行参数选择测试模式
    if (argc > 1 && std::string(argv[1]) == "--basic") {
        run_basic_test();
    } else {
        run_all_tests();
    }
    
    // 清理
    cudaDeviceReset();
    
    return EXIT_SUCCESS;
}