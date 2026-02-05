/**
 * @file
 * @brief 不同共享内存瓦片类型之间的组转换操作。
 */

/* ----------  拷贝操作  ---------- */

/**
 * @brief 将源共享内存瓦片复制到目标共享内存瓦片，可进行类型转换。
 *
 * 该函数使用线程组（GROUP_THREADS）中的所有线程协作完成复制操作。
 * 每个线程负责复制一部分元素，实现高效的并行复制。
 * 在复制过程中，如果源类型和目标类型不同，会自动进行类型转换。
 *
 * @tparam ST1 目标共享内存瓦片类型，必须满足 ducks::st::all 概念。
 * @tparam ST2 源共享内存瓦片类型，必须满足 ducks::st::all 概念。
 * @param dst[out] 目标共享内存瓦片，用于存储转换后的数据。
 * @param src[in] 源共享内存瓦片，提供要转换的数据。
 */
template<ducks::st::all ST1, ducks::st::all ST2>
__device__ static inline void copy(ST1 &dst, const ST2 &src) {
    // 静态断言：确保源和目标瓦片具有相同的行数和列数
    static_assert(ST1::rows == ST2::rows && ST1::cols == ST2::cols, 
                  "Tiles must have the same rows and cols");
    
    #pragma unroll  // 循环展开优化，提高指令级并行性
    // 将瓦片元素分配给线程组中的每个线程处理
    // 每个线程处理i, i+GROUP_THREADS, i+2*GROUP_THREADS...位置的元素
    for(int i = ::kittens::laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        // 计算元素在瓦片中的行索引和列索引
        // 行索引 = 一维索引 / 列数
        int row = i / dst.cols;
        // 列索引 = 一维索引 % 列数
        int col = i % dst.cols;
        
        // 从源瓦片读取元素，进行类型转换，然后存储到目标瓦片
        // base_types::convertor 提供类型转换功能，确保不同数据类型间的正确转换
        dst[{row, col}] = base_types::convertor<typename ST1::dtype, 
                                                 typename ST2::dtype>::convert(src[{row, col}]);
    }
}