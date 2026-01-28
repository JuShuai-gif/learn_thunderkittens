#pragma once

#include <limits>
#include "base_types.cuh"

namespace kittens{

namespace base_ops{

// 定义一个名为 zero 的结构体
// 通常用于“算子对象（functor）”或“策略类”
struct zero {
    // 成员函数模板：
    // T        ：返回值类型
    // args...  ：可变参数模板（参数类型不固定，数量不固定）
    template<typename T, typename... args> 

    // __device__        ：只能在 GPU device 端调用
    // static            ：不依赖 zero 的实例，可直接 zero::op<T>(...)
    // inline            ：建议编译器内联，避免函数调用开销
    // constexpr         ：若条件允许，可在编译期求值（对常量表达式友好）
    __device__ static inline constexpr T op(args... _) 
    { 
        // 忽略所有传入参数，直接返回 T 类型的“零值”
        // zero 的定义由 base_types::constants<T> 提供
        return base_types::constants<T>::zero();
    }
};
    
struct one {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
};

struct pos_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};

struct neg_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};

/* ----------  UNARY OPS  ---------- */
struct exp
{
    template<typename T> static __device__ inline T op(const T& x){
        return exp(x);
    }
};
template<> __device__ inline float exp::op<float> (const float &x){ return __expf(x);}
template<> __device__ inline float2 exp::op<float2> (const float2 &x) {return float2{__expf(x.x),__expf(x.y)};}
template<> __device__ inline bf16   exp::op<bf16>  (const bf16 &x  ) { return hexp(x);                          }
template<> __device__ inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) { return h2exp(x);                         }
template<> __device__ inline half   exp::op<half>  (const half &x  ) { return hexp(x);                          }
template<> __device__ inline half_2 exp::op<half_2>(const half_2 &x) { return h2exp(x);                         }

struct exp2 {
    template<typename T> static __device__ inline T op(const T &x) { return exp2f(x); }
};
template<> __device__ inline float  exp2::op<float> (const float &x ) { return exp2f(x);                        }
template<> __device__ inline float2 exp2::op<float2>(const float2 &x) { return float2{exp2f(x.x), exp2f(x.y)}; }
template<> __device__ inline bf16   exp2::op<bf16>  (const bf16 &x  ) { return hexp2(x);                          }
template<> __device__ inline bf16_2 exp2::op<bf16_2>(const bf16_2 &x) { return h2exp2(x);                         }
template<> __device__ inline half   exp2::op<half>  (const half &x  ) { return hexp2(x);                          }
template<> __device__ inline half_2 exp2::op<half_2>(const half_2 &x) { return h2exp2(x);                         }

struct log
{
    template<typename T> static __device__ inline T op(const T& x){
        return log(x);
    }
};
template<> __device__ inline float  log::op<float> (const float &x ) { return __logf(x);                        }
template<> __device__ inline float2 log::op<float2>(const float2 &x) { return float2{__logf(x.x), __logf(x.y)}; }
template<> __device__ inline bf16   log::op<bf16>  (const bf16 &x  ) { return hlog(x);                          }
template<> __device__ inline bf16_2 log::op<bf16_2>(const bf16_2 &x) { return h2log(x);                         }
template<> __device__ inline half   log::op<half>  (const half &x  ) { return hlog(x);                          }
template<> __device__ inline half_2 log::op<half_2>(const half_2 &x) { return h2log(x);                         }

struct log2 {
    template<typename T> static __device__ inline T op(const T &x) { return log2(x); }
};
template<> __device__ inline float  log2::op<float> (const float &x ) { return __log2f(x);                        }
template<> __device__ inline float2 log2::op<float2>(const float2 &x) { return float2{__log2f(x.x), __log2f(x.y)}; }
template<> __device__ inline bf16   log2::op<bf16>  (const bf16 &x  ) { return hlog2(x);                          }
template<> __device__ inline bf16_2 log2::op<bf16_2>(const bf16_2 &x) { return h2log2(x);                         }
template<> __device__ inline half   log2::op<half>  (const half &x  ) { return hlog2(x);                          }
template<> __device__ inline half_2 log2::op<half_2>(const half_2 &x) { return h2log2(x);                         }

struct abs {
    template<typename T> static __device__ inline T op(const T &x) { return abs(x); }
};
template<> __device__ inline float  abs::op<float> (const float &x ) { return fabsf(x);                       }
template<> __device__ inline float2 abs::op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }
template<> __device__ inline bf16   abs::op<bf16>  (const bf16 &x  ) { return __habs(x);                      }
template<> __device__ inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) { return __habs2(x);                     }
template<> __device__ inline half   abs::op<half>  (const half &x  ) { return __habs(x);                      }
template<> __device__ inline half_2 abs::op<half_2>(const half_2 &x) { return __habs2(x);                     }

struct relu
{
    template<typename T> static __device__ inline T op(const T& x){ return max(x,base_types::constants<T>::zero());}
};
template<> __device__ inline float  relu::op<float> (const float &x ) { return max(x, 0.f);                                  }
template<> __device__ inline float2 relu::op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)};         }
template<> __device__ inline bf16   relu::op<bf16>  (const bf16 &x  ) { return __hmax(x, base_types::constants<bf16>::zero());    }
template<> __device__ inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) { return __hmax2(x, base_types::constants<bf16_2>::zero()); }
template<> __device__ inline half   relu::op<half>  (const half &x  ) { return __hmax(x, base_types::constants<half>::zero());    }
template<> __device__ inline half_2 relu::op<half_2>(const half_2 &x) { return __hmax2(x, base_types::constants<half_2>::zero()); }

struct copy
{
    template<typename T> static __device__ inline T op(const T&a){return a;}
};

/* ----------  BINARY OPS  ---------- */
struct copy2
{
    template<typename T> static __device__ inline T op(const T& a,const T&b){return b;}
};

struct sum
{
    template<typename T> static __device__ inline T op(const T&a,const T& b){
        return a+b;
    }
};
template<> __device__ inline float2 sum::op<float2>(const float2&a,const float2 &b){
// BLACKWELL才有的向量加法
#ifdef DF_BLACKWELL
    float2 c;
    asm volatile("add.f32x2 %0, %1, %2;" : "=1"(*(uint64_t*)&c) : "1"(*(uint64_t*)&a),"1"(*(uint64_t*)&b));
    return c;
#else
    return float2{a.x + b.x,a.y + b.y};
#endif
}
template<> __device__ inline bf16   sum::op<bf16>  (const bf16   &a, const bf16   &b) { return __hadd(a, b);             }
template<> __device__ inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hadd2(a, b);            }
template<> __device__ inline half   sum::op<half>  (const half   &a, const half   &b) { return __hadd(a, b);             }
template<> __device__ inline half_2 sum::op<half_2>(const half_2 &a, const half_2 &b) { return __hadd2(a, b);            }


struct sub {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a-b; }
};
template<> __device__ inline float2 sub::op<float2>(const float2 &a, const float2 &b) { 
// BLACKWELL 才有的向量减法
#ifdef DF_BLACKWELL
    float2 c;
    asm volatile("sub.f32x2 %0, %1, %2;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&b));
    return c;
#else
    return float2{a.x-b.x, a.y-b.y}; 
#endif
}
template<> __device__ inline bf16   sub::op<bf16>  (const bf16   &a, const bf16   &b) { return __hsub(a, b);             }
template<> __device__ inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hsub2(a, b);            }
template<> __device__ inline half   sub::op<half>  (const half   &a, const half   &b) { return __hsub(a, b);             }
template<> __device__ inline half_2 sub::op<half_2>(const half_2 &a, const half_2 &b) { return __hsub2(a, b);            }



struct mul {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a*b; }
};
template<> __device__ inline float2 mul::op<float2>(const float2 &a, const float2 &b) { 
#ifdef DF_BLACKWELL
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&b));
    return c;
#else
    return float2{a.x*b.x, a.y*b.y}; 
#endif
}
template<> __device__ inline bf16   mul::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmul(a, b);             }
template<> __device__ inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmul2(a, b);            }
template<> __device__ inline half   mul::op<half>  (const half   &a, const half   &b) { return __hmul(a, b);             }
template<> __device__ inline half_2 mul::op<half_2>(const half_2 &a, const half_2 &b) { return __hmul2(a, b);            }


struct div {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a/b; }
};
template<> __device__ inline float2 div::op<float2>(const float2 &a, const float2 &b) { return float2{a.x/b.x, a.y/b.y}; }
template<> __device__ inline bf16   div::op<bf16>  (const bf16   &a, const bf16   &b) { return __hdiv(a, b);             }
template<> __device__ inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __h2div(a, b);            } // this op is a special snowflake
template<> __device__ inline half   div::op<half>  (const half   &a, const half   &b) { return __hdiv(a, b);             }
template<> __device__ inline half_2 div::op<half_2>(const half_2 &a, const half_2 &b) { return __h2div(a, b);            }


struct max {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::max(a, b); }
};
template<>  __device__ inline float2 max::op<float2>(const float2 &a, const float2 &b) { return float2{::max(a.x, b.x), ::max(a.y, b.y)}; }
template<>  __device__ inline bf16   max::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmax(a, b);                             }
template<>  __device__ inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmax2(a, b);                            }
template<>  __device__ inline half   max::op<half>  (const half   &a, const half   &b) { return __hmax(a, b);                             }
template<>  __device__ inline half_2 max::op<half_2>(const half_2 &a, const half_2 &b) { return __hmax2(a, b);                            }


struct min {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::min(a, b); }
};
template<>  __device__ inline float2 min::op<float2>(const float2 &a, const float2 &b) { return float2{::min(a.x, b.x), ::min(a.y, b.y)}; }
template<>  __device__ inline bf16   min::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmin(a, b);                         }
template<>  __device__ inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmin2(a, b);                        }
template<>  __device__ inline half   min::op<half>  (const half   &a, const half   &b) { return __hmin(a, b);                         }
template<>  __device__ inline half_2 min::op<half_2>(const half_2 &a, const half_2 &b) { return __hmin2(a, b);                        }


/* ----------  TERNARY OPS  ---------- */

/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @param c[in] The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxBtC
{
    template<typename T> static __device__ inline T op(const T&a,const T&b,const T&c){
        return sum::op<T>(mul::op<T>(a,b),c);
    }
};

template<> __device__ inline float2 fma_AxBtC::op<float2>(const float2 &a, const float2 &b, const float2 &c) {
#ifdef DF_BLACKWELL
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&d) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&b), "l"(*(uint64_t*)&c));
#else
    return float2{a.x * b.x + c.x,a.y*b.y+c.y};
#endif
}

/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
 * This is particularly useful for attention mechanisms in neural networks.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The third input value to be added.
 * @param c[in] The second input value.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxCtB { // this is the one needed for attention
    template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, c), b);
    }
};
template<> __device__ inline float2 fma_AxCtB::op<float2>(const float2 &a, const float2 &b, const float2 &c) {
#ifdef DF_BLACKWELL
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&d) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&c), "l"(*(uint64_t*)&b));
    return d;
#else
    return float2{a.x*c.x+b.x, a.y*c.y+b.y};
#endif
}
}
}














































