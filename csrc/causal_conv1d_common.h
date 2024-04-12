/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#endif

#ifndef USE_ROCM
#define _SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
#define _SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
namespace rocm_utils {

template <typename T>
__host__ __device__ constexpr T min(T x, T y) 
{
    return x < y ? x : y;
}

template <typename T>
__host__ __device__ constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

} // namespace rocm_utils
#endif // USE_ROCM

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, _SHFL_XOR_SYNC(x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    x = op(x, _SHFL_XOR_SYNC(x, 1));
    return x;
}
};
