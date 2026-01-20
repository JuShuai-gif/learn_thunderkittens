#pragma once

namespace kittens{

enum class reduce_op{
    ADD = 0,
    MIN = 1,
    MAX = 2
};

enum class memory_model{
    WEAK = 0,
    STRONG = 1
};

template<typename T>
struct multimem;


template<>
struct multimem<int>
{
    template <reduce_op Op,memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(int& dst,const int* src){
        if constexpr(Op == reduce_op::ADD){
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }else if constexpr(Op == reduce_op::MIN){
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }else if constexpr(Op == reduce_op::MAX){
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }
    }

    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(int *dst, const int &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.s32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.s32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(int *dst, const int &src) {
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MIN) {
            asm volatile("multimem.red.release.sys.global.min.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MAX) {
            asm volatile("multimem.red.release.sys.global.max.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        }
    }
};

template <>
struct multimem<uint>
{
    
};























}




























