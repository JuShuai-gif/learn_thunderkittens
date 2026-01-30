#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"


/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens{
namespace ducks{

namespace st{// 用于类型推断的标识结构体

struct identifier{};// 类型标识符基础结构

// 概念检查：检查类型T是否具有正确的标识符
template<typename T> concept all = requires {
    typename T::identifier; // 检查T::identifier是否存在
} && std::is_same_v<typename T::identifier, identifier>; // 检查T::identifier是否为ducks::st::identifier
}
} // namespace ducks

// st_subtile的前向声明，用于在st类中作为友元
template<
    typename ST,            // 父tile类型
    int _subtile_height,    // 子tile高度
    int _subtile_width      // 子tile宽度
>
struct st_subtile;


/**
 * @brief 支持不同数据类型和布局的共享内存tile结构
 *
 * @tparam _T 元素的数据类型（未打包的原始类型）
 * @tparam _rows tile的高度
 * @tparam _cols tile的宽度
 * @tparam _swizzle 是否启用内存重排（swizzle）优化
 * @tparam _swizzle_bytes swizzle的字节粒度，0表示自动计算
 */
template<typename _T, int _rows, int _cols, bool _swizzle=true, int _swizzle_bytes=0>
struct DF_DEFAULT_ALIGN st {    // 默认对齐的共享内存tile结构
#ifdef DF_BLACKWELL
    // Blackwell架构下，FP4类型必须使用打包类型
    static_assert(!std::is_same_v<_T, fp4e2m1>, "For FP4 types, you must use a packed type (i.e., fp4e2m1_2 or fp4e2m1_4).");
#endif
    using identifier = ducks::st::identifier;       // 类型标识符
    using T = base_types::packing<_T>::unpacked_type;// 未打包的数据类型
    using T2 = base_types::packing<_T>::packed_type;// 打包的数据类型
    using dtype = T; ///< tile中元素的数据类型

    static constexpr bool swizzle = _swizzle;// 是否启用swizzle优化

    // 底层数据定义，明确这不是子tile视图
    static constexpr int underlying_rows          = _rows;// 底层行数
    static constexpr int underlying_cols          = _cols;// 底层列数
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;// 底层元素总数

    static constexpr int rows                = _rows; ///< tile的总行数
    static constexpr int cols                = _cols; ///< tile的总列数
    static constexpr int num_elements        = rows * cols; ///< tile中的总元素数
    
    // 静态断言：确保维度符合要求
    static_assert(!swizzle || (rows % kittens::TILE_ROW_DIM<T> == 0), "Rows must be divisible by the tile dimension");
    static_assert((swizzle && (cols % kittens::TILE_COL_DIM<T> == 0)) || (!swizzle && (cols % kittens::BASE_TILE_DIM == 0)), "Cols must be divisible by the tile dimension");


#ifdef DF_BLACKWELL
    // Blackwell架构：必须是1-packed类型（如float, bf16等），除非是fp4类型
    static_assert(base_types::packing<dtype>::num() == 1 || std::is_same_v<dtype, fp4e2m1_2>); 
#else
    // 其他架构：必须是1-packed类型
    static_assert(base_types::packing<dtype>::num() == 1); 
#endif

    // 如果用户指定了swizzle_bytes值，列字节大小必须是swizzle_bytes的倍数
    static_assert(_swizzle_bytes == 0 || _swizzle_bytes == 32 || _swizzle_bytes == 64 || _swizzle_bytes == 128);
    // 计算swizzle_bytes：如果指定则使用指定值，否则根据数据类型和列数自动计算
    static constexpr int swizzle_bytes = _swizzle_bytes > 0 ? _swizzle_bytes : (
        sizeof(dtype) == 1 ? (  // 处理1字节数据类型（如FP8）
            (cols/kittens::TILE_COL_DIM<T>)%4 == 0 ? 128 :
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 2 ? (// 处理2字节数据类型（如half, bf16）
            (cols/kittens::TILE_COL_DIM<T>)%4 == 0 ? 128 :
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (// 处理4字节数据类型（如float）
            (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ? 128 : 64
        ) : -1
    );    

    dtype data[rows * cols]; // 实际存储数据的数组
    
    /**
     * @brief 根据坐标计算内存地址（支持swizzle优化）
     * @param ptr 指向数据的指针
     * @param coord 坐标(x,y)，其中x是行索引，y是列索引
     * @return 指向计算地址的指针
     */
    __device__ static inline T* idx(T* ptr,int2 coord){
        int r = coord.x,c = coord.y; // 坐标别名，提高可读性

        if constexpr(swizzle){// 如果启用了swizzle
            static constexpr int swizzle_repeat = swizzle_bytes * 8;    // swizzle重复周期
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);    // 每个子tile的列数
            const int outer_idx = c/subtile_cols;   // 外部索引
            // 计算原始地址
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;// 计算swizzle值
            return (T*)(addr ^ swizzle); // 应用swizzle：通过异或操作重排地址位
        } else {// 未启用swizzle
            return &ptr[r*cols + c];// 简单的行主序索引
        }
    }

    /**
     * @brief 根据坐标计算内存地址（32位版本）
     * @param ptr 基础地址（32位）
     * @param coord 坐标(x,y)
     * @return 计算后的32位地址
     */
    __device__ static inline uint32_t idx(uint32_t ptr, int2 coord) {
        int r = coord.x, c = coord.y; // 坐标别名
        if constexpr (swizzle) {// 如果启用了swizzle
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            // 计算原始地址（32位版本）
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (addr ^ swizzle);// 应用swizzle
        } else {// 未启用swizzle
            return ptr + sizeof(T)*(r*cols + c);
        }
    }
    // 下标运算符重载（支持2D坐标）
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }

    // 下标运算符重载（支持1D索引）
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }
    
    /**
     * @brief 创建子tile视图（支持任意行/列划分）
     * @tparam subtile_rows 子tile的行数
     * @tparam subtile_cols 子tile的列数
     * @param rowcol 子tile在原tile中的起始坐标（以子tile为单位）
     * @return 子tile视图对象
     */
    template<int subtile_rows, int subtile_cols>
    __device__ inline st_subtile<st<_T, _rows, _cols, _swizzle, _swizzle_bytes>, subtile_rows, subtile_cols> subtile(int2 rowcol);

    /**
     * @brief 返回真正的"子tile"（而非视图），仅支持列维度的划分
     * @tparam subtile_cols 子tile的列数
     * @param idx 子tile的索引
     * @return 对子tile的引用
     * 约束：只能划分列维度，且必须是swizzle_bytes的倍数
     */
    template<int subtile_cols>
    __device__ inline st<_T, _rows, subtile_cols, _swizzle, swizzle_bytes /*must not use _swizzle_bytes*/> &subtile(int idx) {
        static_assert(swizzle_bytes > 0, "Parent shared tile must have an explicit swizzle_bytes.");
        constexpr int swizzle_elements = swizzle_bytes / sizeof(T);// 每个swizzle块中的元素数
        static_assert(subtile_cols >= 0 && subtile_cols % swizzle_elements == 0);// 验证子tile列数是swizzle_elements的倍数
        // 计算并返回对子tile的引用
        return *reinterpret_cast<st<_T, _rows, subtile_cols, _swizzle, swizzle_bytes> *>(
            &data[rows*swizzle_elements*(subtile_cols/swizzle_elements)*idx]
        );
    }

    // 向量类型别名
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
};


/**
 * @brief 共享内存tile的引用（视图），指向原始tile的一部分
 *
 * st_subtile是st的替代品，内部引用适当的内存并进行最小的地址计算。
 * 不应直接创建此对象，而应使用subtile方法返回它。
 * 通常可以将其视为st，但不能用于wgmma操作。
 */
template<
    typename _ST,// 父tile类型
    int _subtile_rows,// 子tile行数
    int _subtile_cols// 子tile列数
>
struct st_subtile
{
    using identifier = ducks::st::identifier;
    using ST = _ST;
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T;    ///< tile中元素的数据类型

    static constexpr bool swizzle = ST::swizzle;// 继承父tile的swizzle设置
    // 父tile的底层维度
    static constexpr int underlying_rows          = ST::underlying_rows;
    static_assert(underlying_rows % kittens::TILE_ROW_DIM<T> == 0, "Underlying rows must be divisible by the tile dimension");
    static constexpr int underlying_cols          = ST::underlying_cols;
    static_assert(underlying_cols % kittens::TILE_COL_DIM<T> == 0, "Underlying cols must be divisible by the tile dimension");
    static constexpr int underlying_num_elements  = ST::underlying_num_elements;
    // 子tile的维度
    static constexpr int rows                = _subtile_rows;
    static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _subtile_cols;
    static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
    static constexpr int num_elements        = rows * cols;

    static constexpr int swizzle_bytes = ST::swizzle_bytes;// 继承父tile的swizzle_bytes
    
    dtype* data;// 指向父tile数据的指针
    int row_offset,col_offset;// 行偏移量 列偏移量

    /**
     * @brief 构造函数
     * @param src 父tile引用
     * @param rowcol 子tile在父tile中的起始坐标（以子tile为单位）
     */
    __device__ st_subtile(ST &src,int2 rowcol){
        data = &src.data[0];// 指向父tile数据
        row_offset = rowcol.x * rows;// 计算行偏移
        col_offset = rowcol.y * cols;   // 计算列偏移     
    }
    
    /**
     * @brief 根据坐标计算内存地址（考虑偏移量）
     * @param ptr 指向数据的指针
     * @param coord 相对坐标（相对于子tile）
     * @return 指向计算地址的指针
     */
    __device__ inline T* idx(T *ptr, const int2 coord) const { // naive row-major coord default
        int r = coord.x+row_offset, c = coord.y+col_offset; // 计算绝对坐标
        if constexpr (swizzle) {// 如果启用了swizzle
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            // 计算绝对地址
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (T*)(addr ^ swizzle);// 应用swizzle
        } else {// 未启用swizzle
            return &ptr[r*cols + c];
        }
    }

    /**
     * @brief 根据坐标计算内存地址（32位版本，考虑偏移量）
     * @param ptr 基础地址（32位）
     * @param coord 相对坐标
     * @return 计算后的32位地址
     */
    __device__ inline uint32_t idx(uint32_t ptr, const int2 coord) const { // naive row-major coord default
        int r = coord.x+row_offset, c = coord.y+col_offset;  // 计算绝对坐标
        if constexpr(swizzle) {// 如果启用了swizzle
            static constexpr int swizzle_repeat = swizzle_bytes * 8;
            static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
            const int outer_idx = c/subtile_cols;
            // 计算绝对地址（32位版本）
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
            return (addr ^ swizzle);// 应用swizzle
        } else {// 未启用swizzle
            return ptr + sizeof(T)*(r*cols + c);
        }
    }    

    /**
     * @brief 访问子tile元素（使用行和列坐标，如同tile是行主序的）
     * 这是访问共享内存tile的首选方式，抽象了swizzle布局的索引计算
     */
    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }

    // 注意：单坐标operator[]未定义，因为这不适用于st_subtile类型
    // 当然，可以直接通过.data来绕过这个限制

    // 向量类型别名
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
    /**
     * @brief 赋值运算符，用于将整个子tile设置为指定值（默认在warp范围内执行）
     * @param value 要设置的值
     */
    __device__ inline void operator=(const dtype &value) { // runs at warp scope by default
        #pragma unroll
        for(int i = kittens::laneid(); i < num_elements; i += WARP_THREADS) {
            data[i] = value;// 注意：这里直接访问data，不考虑偏移
        }
    }
};

// st类的subtile方法的实现（定义）
template <typename _T, int _rows, int _cols, bool _swizzle, int _swizzle_bytes> // 类模板参数
template <int subtile_rows, int subtile_cols> // 函数模板参数
__device__ inline st_subtile<st<_T, _rows, _cols, _swizzle, _swizzle_bytes>, subtile_rows, subtile_cols> // 返回类型
st<_T, _rows, _cols, _swizzle, _swizzle_bytes>::subtile(int2 rowcol) // 限定函数名和参数
{
    // 函数体内的类型别名
    using ST_t = st<_T, _rows, _cols>;  // 父tile类型的别名
    using dtype = typename ST_t::dtype; // 数据类型的别名

    // 静态断言：验证子tile维度有效性
    static_assert(subtile_rows > 0 && subtile_cols > 0, "Subtile dimensions must be positive.");
    static_assert(subtile_rows % kittens::TILE_ROW_DIM<dtype> == 0,
        "Subtile rows must be divisible by the base tile row dimension.");
    static_assert(subtile_cols % kittens::TILE_COL_DIM<dtype> == 0,
        "Subtile cols must be divisible by the base tile col dimension.");

    // 检查父tile维度是否能被子tile维度整除
    static_assert(ST_t::rows % subtile_rows == 0,
        "Parent tile rows must be divisible by subtile rows.");
    static_assert(ST_t::cols % subtile_cols == 0,
        "Parent tile cols must be divisible by subtile cols.");

    // 确保父st对象本身不是子tile视图
    static_assert(ST_t::rows == ST_t::underlying_rows && ST_t::cols == ST_t::underlying_cols,
        "Cannot create a subtile from an object that appears to be a subtile view (rows/cols mismatch underlying).");

    // 构造并返回st_subtile对象
    return st_subtile<ST_t, subtile_rows, subtile_cols>(*this, rowcol);
}


/* ----------  用于美观性的包装器类型 ---------- */
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_bf = st<bf16,  _height, _width, _swizzle, _swizzle_bytes>;// bfloat16类型的tile

template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_hf = st<half,  _height, _width, _swizzle, _swizzle_bytes>;// half类型的tile

template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_fl = st<float, _height, _width, _swizzle, _swizzle_bytes>;// float类型的tile

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_fp8e4m3 = st<fp8e4m3, _height, _width, _swizzle, _swizzle_bytes>;// FP8 E4M3类型的tile
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_fp8e5m2 = st<fp8e5m2, _height, _width, _swizzle, _swizzle_bytes>;// FP8 E5M2类型的tile
#endif
#if defined(DF_BLACKWELL)
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_fp8e8m0 = st<fp8e8m0, _height, _width, _swizzle, _swizzle_bytes>;// FP8 E8M0类型的tile
template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_fp4e2m1_2 = st<fp4e2m1_2, _height, _width, _swizzle, _swizzle_bytes>; // FP4 E2M1（打包为2
#endif


/* ----------  PRINTOUTS  ---------- */

/**
 * @brief Get a readable type name for shared tiles
 */
template<typename T, int rows, int cols>
__device__ constexpr const char* get_tile_type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "st_fl";
    } else if constexpr (std::is_same_v<T, half>) {
        return "st_hf";
    } else if constexpr (std::is_same_v<T, bf16>) {
        return "st_bf";
#if defined(DF_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp4e2m1_2>) {
        return "st_fp4_e2m1_2";
    } else if constexpr (std::is_same_v<T, fp8e8m0>) {
        return "st_fp8_e8m0";
#endif
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    } else if constexpr (std::is_same_v<T, fp8e4m3>) {
        return "st_fp8_e4m3";
    } else if constexpr (std::is_same_v<T, fp8e5m2>) {
        return "st_fp8_e5m2";
#endif  
    } else {
        return "st_unknown";
    }
}

#if defined(DF_BLACKWELL)
/**
 * @brief Print the contents of a shared tile as a formatted table.
 * 
 * This function should be called by a single thread in the warp.
 * It will print the entire tile atomically to avoid interleaved output.
 * 
 * @param tile The shared tile to print
 */
template<ducks::st::all ST>
__device__ inline void print_fp4(const ST& tile) {
    if (std::is_same_v<typename ST::dtype, fp4e2m1>) {

        constexpr int cols = ST::cols * 2;
        printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, cols, get_tile_type_name<typename ST::dtype, ST::rows, cols>(), ST::rows, cols);

        // Print column headers
        printf("     "); // Padding for row indices
        for (int c = 0; c < cols; c++) {
            printf("%8d ", c);
        }
        printf("\n");
        
        // Print separator line
        printf("     ");
        for (int c = 0; c < cols; c++) {
            printf("--------+");
        }
        printf("\n");
        
        // Print data rows
        for (int r = 0; r < ST::rows; r++) {
            printf("%3d |", r); // Row index
            for (int c = 0; c < cols; c += 2) {
                uint8_t *vals = reinterpret_cast<uint8_t*>(const_cast<fp4e2m1*>(&tile[{r,c/2}]));

                // Convert to fp4e2m1 and then to float
                float f1 = static_cast<float>(fp4e2m1(vals[0] & 0xF));
                float f2 = static_cast<float>(fp4e2m1((vals[0] >> 4) & 0xF));

                printf("%8.3f %8.3f ", f1, f2);

            }
            printf("\n");
        }
        printf("\n");
    } else {
        printf("Type must be FP4 in this function\n");
    }
}
#endif

/**
 * @brief Print the contents of a shared tile as a formatted table.
 * 
 * This function should be called by a single thread in the warp.
 * It will print the entire tile atomically to avoid interleaved output.
 * 
 * @param tile The shared tile to print
 */
template<ducks::st::all ST>
__device__ inline void print(const ST& tile) {
    printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, ST::cols, get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);
    
    // Print column headers
    printf("     "); // Padding for row indices
    for (int c = 0; c < ST::cols; c++) {
        printf("%8d ", c);
    }
    printf("\n");
    
    // Print separator line
    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("--------+");
    }
    printf("\n");
    
    // Print data rows
    for (int r = 0; r < ST::rows; r++) {
        printf("%3d |", r); // Row index
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", tile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(tile[{r,c}]));
            } else if constexpr (std::is_integral_v<typename ST::dtype>) {
                printf("%8d ", (int)tile[{r,c}]);
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e4m3>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e5m2>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#endif
#if defined(DF_BLACKWELL)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e8m0>) {
                printf("%8.3f ", static_cast<float>(tile[{r,c}]));
#endif
            } else {
                printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Print the contents of a shared subtile as a formatted table.
 * 
 * This function prints subtiles with additional information about their position
 * within the parent tile.
 * 
 * @param subtile The shared subtile to print
 */
template<typename ST, int subtile_rows, int subtile_cols>
__device__ inline void print(const st_subtile<ST, subtile_rows, subtile_cols>& subtile) {
    printf("Block %d: Shared Subtile %dx%d (offset: [%d,%d], Type: %s<%d,%d> from %s<%d,%d>):\n", 
            blockIdx.x, subtile.rows, subtile.cols, 
            subtile.row_offset, subtile.col_offset,
            get_tile_type_name<typename ST::dtype, subtile.rows, subtile.cols>(), subtile.rows, subtile.cols,
            get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);
    
    // Print column headers
    printf("     "); // Padding for row indices
    for (int c = 0; c < subtile.cols; c++) {
        printf("%8d ", c);
    }
    printf("\n");
    
    // Print separator line
    printf("     ");
    for (int c = 0; c < subtile.cols; c++) {
        printf("--------+");
    }
    printf("\n");
    
    // Print data rows
    for (int r = 0; r < subtile.rows; r++) {
        printf("%3d |", r); // Row index
        for (int c = 0; c < subtile.cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", subtile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(subtile[{r,c}]));
            } else if constexpr (std::is_integral_v<typename ST::dtype>) {
                printf("%8d ", (int)subtile[{r,c}]);
            } else {
                printf("%8.3f ", (float)subtile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
/**
 * @brief Fill a shared tile with ones.
 * 
 * This function should be called by a single thread in the warp.
 * It will fill the entire tile with 1s (value, not bits) in the tile.
 * 
 * @param tile The shared tile to fill with 1s
 */
template<ducks::st::all ST>
__device__ inline void fill_value(ST& tile, float value) {
    printf("Filling Tile %dx%d with %f:\n", ST::rows, ST::cols, value);

    // Fill tile with value
    for (int r = 0; r < ST::rows; r++) {
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                tile[{r,c}] = value;
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                tile[{r,c}] = __float2bfloat16(value);
#if defined(DF_BLACKWELL)  
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1>) {
                tile.data[r*ST::cols + c] = fp4e2m1(value);
            } else if constexpr (std::is_same_v<typename ST::dtype, fp8e8m0>) {
                tile.data[r*ST::cols + c] = fp8e8m0(value);
#endif
            } else {    
                tile[{r,c}] = value;
            }
        }
    }
}

/**
 * @brief Fill a shared tile with something!
 * 
 * This function should be called by a single thread in the warp.
 * It will fill the entire tile with 1s (value, not bits) in the tile.
 * 
 * @param tile The shared tile to fill with 1s
 */
template<ducks::st::all ST>
__device__ inline void fill_identity(ST& tile) {
    printf("Filling Tile %dx%d with identity:\n", ST::rows, ST::cols);
    
    // Print data rows
    for (int r = 0; r < ST::rows; r++) {
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                // printf("%8.3f ", tile[{r,c}]);
                tile[{r,c}] = 1.0f;
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                // printf("%8.3f ", __bfloat162float(tile[{r,c}]));
                tile[{r,c}] = __float2bfloat16(1.0f);
#if defined(DF_BLACKWELL)  
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1>) {
                if(r == c){
                    // tile[{r,c}] = std::bit_cast<fp4e2m1>(uint8_t(0xFF));
                    tile.data[r*ST::cols + c] = std::bit_cast<fp4e2m1>(uint8_t(0xFF));
                } else {
                    tile.data[r*ST::cols + c] = std::bit_cast<fp4e2m1>(uint8_t(0x00));
                }
#endif
            } else {    
                // printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
    }
}

/**
 * @brief Print the contents of a shared tile as a formatted table in bits.
 * 
 * This function should be called by a single thread in the warp.
 * It will print the entire tile atomically to avoid interleaved output.
 * Each element will be printed as a bitfield
 * 
 * @param tile The shared tile to print
 */
template<ducks::st::all ST>
__device__ inline void print_bits(const ST& tile, bool unswizzle = false) {
    printf("Block %d: Shared Tile %dx%d (Type: %s<%d,%d>):\n", blockIdx.x, ST::rows, ST::cols, get_tile_type_name<typename ST::dtype, ST::rows, ST::cols>(), ST::rows, ST::cols);
    
    // Print column headers
    printf(" "); // Padding for row indices
    for (int c = 0; c < ST::cols; c++) {
        printf("%11d ", c);
    }
    printf("\n");
    
    // Print separator line
    printf("     ");
    for (int c = 0; c < ST::cols; c++) {
        printf("-----------+");
    }
    printf("\n");
    
    // Print data rows
    for (int r = 0; r < ST::rows; r++) {
        printf("%3d |", r); // Row index
        for (int c = 0; c < ST::cols; c++) {
            if constexpr (std::is_same_v<typename ST::dtype, float>) {
                printf("%8.3f ", tile[{r,c}]);
            } else if constexpr (std::is_same_v<typename ST::dtype, __nv_bfloat16>) {
                printf("%8.3f ", __bfloat162float(tile[{r,c}]));
            // } else if constexpr (std::is_integral_v<typename ST::dtype>) {
            //     printf("%8d ", (int)tile[{r,c}]);
#if defined(DF_BLACKWELL)
            } else if constexpr (std::is_same_v<typename ST::dtype, fp4e2m1> || std::is_same_v<typename ST::dtype, fp8e8m0>) {
                // print as bitfield

                uint8_t bits;
                if(unswizzle){
                    bits = *reinterpret_cast<const uint8_t*>(&tile.data[r*ST::cols + c]); // Assuming 4-bit value
                } else {
                    bits = *reinterpret_cast<const uint8_t*>(&tile[{r,c}]); // Assuming 4-bit value
                }
                // Print all 32 bits with formatting for readability
                printf("0b");
                // Print in groups of 4 for readability
                for (int bit = 7; bit >= 0; bit--) {
                    printf("%d", (bits >> bit) & 0x1);
                    if (bit % 4 == 0 && bit > 0) printf("_");
                }
                printf(" ");
#endif
            } else {
                printf("%8.3f ", (float)tile[{r,c}]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace kittens


















