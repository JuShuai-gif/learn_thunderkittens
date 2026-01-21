#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens{

namespace ducks{


namespace sv{

struct identifier 
{
    
};

template <typename T>
concept all = requires{
    typename T::identifier;
} && std::is_same_v<typename T::identifier,identifier>;
}
}

template<typename _T,size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = length / TILE_ROW_DIM<T>; ///< Length in subtiles.'
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
#endif
#if defined(DF_BLACKWELL)
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4> || !std::is_same_v<T2, fp8e8m0_4>, "Unsupported type for fp8");
#endif

#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
    static constexpr int num_alloc_elements = ((length * sizeof(dtype) + 127) / 128) * (128 / sizeof(dtype)); // round up to the nearest 128-byte boundary
#else
    static constexpr int num_alloc_elements = length;
#endif
    dtype data[num_alloc_elements]; ///< The actual shared vector data.


    __device__ static inline T* idx(T* ptr,int idx){// useful for computations in shared address space, as silly as it sounds.
        return ptr[idx];
    }

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<int sub_length> __device__ inline sv<_T, sub_length> &subvec(int idx) {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }
    template<int sub_length> __device__ inline const sv<_T, sub_length> &subvec(int idx) const {
        return *(sv<dtype, sub_length>*)&data[idx * sub_length];
    }

    __device__ inline void operator=(const dtype &value) { // runs at warp scope by default
        #pragma unroll
        for(int i = kittens::laneid(); i < length; i += WARP_THREADS) {
            data[i] = value;
        }
    }


};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// vector types
template<size_t _length> using sv_bf = sv<bf16,  _length>;
template<size_t _length> using sv_hf = sv<half,  _length>;
template<size_t _length> using sv_fl = sv<float, _length>;
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
template<int _length> using sv_fp8e4m3 = sv<fp8e4m3, _length>;
template<int _length> using sv_fp8e5m2 = sv<fp8e5m2, _length>;
#endif
#if defined(DF_BLACKWELL)
template<int _length> using sv_fp8e8m0 = sv<fp8e8m0, _length>;
template<int _length> using sv_fp4e2m1_2 = sv<fp4e2m1_2, _length>;
#endif

/* ----------  PRINTOUTS  ---------- */

template<ducks::sv::all SV>
__device__ inline void print(const SV& sv) {
    printf("Shared Vector %d:\n", SV::length);
    for(int i = 0; i < SV::length; i++) {
        if constexpr (std::is_same_v<typename SV::dtype, bf16>) {
            printf("%f ", __bfloat162float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, half>) {
            printf("%f ", __half2float(sv[i]));
        } else if constexpr (std::is_same_v<typename SV::dtype, float>) {
            printf("%f ", sv[i]);
#if defined(DF_HOPPER) || defined(DF_BLACKWELL)
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e4m3>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
#ifdef DF_BLACKWELL
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e8m0>) {
            printf("%f ", static_cast<float>(sv[i]));
#endif
        } else {
            printf("%d ", (int)(sv[i]));
        }
    }
    printf("\n");
}
}






















