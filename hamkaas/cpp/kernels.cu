#include <cstdint>

#include <cstdio>

#include "helpers.h"
#include "tensor.h"

namespace NHamKaas {

template <typename T>
__global__ void SumTensorsBroadcastKernel(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShapeMultiplier,
    int64_t dimensions,
    int64_t outputSize)
{
    int64_t indices[MaxDimensions];

    int64_t lhsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (lhsIndex >= outputSize) {
        return;
    }

    int64_t lhsIndexCopy = lhsIndex;

    for (int64_t i = dimensions - 1; i >= 0; --i) {
        indices[i] = lhsIndexCopy % lhsShape[i];
        lhsIndexCopy /= lhsShape[i];
    }

    int64_t rhsIndex = 0;
    for (int64_t i = 0; i < dimensions; ++i) {
        rhsIndex = rhsIndex * lhsShape[i] + indices[i] * rhsShapeMultiplier[i];
    }

    output[lhsIndex] = lhs[lhsIndex] + rhs[rhsIndex];
}

template <typename T>
void SumTensorsBroadcast(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShapeMultiplier,
    int64_t dimensions,
    int64_t outputSize)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (outputSize + ThreadsPerBlock - 1) / ThreadsPerBlock;

    SumTensorsBroadcastKernel<T><<<blocks, ThreadsPerBlock>>>(
        lhs,
        rhs,
        output,
        lhsShape,
        rhsShapeMultiplier,
        dimensions,
        outputSize);
}

#define INSTANTIATE(T) \
    template void SumTensorsBroadcast( \
        const T* lhs, \
        const T* rhs, \
        T* output, \
        int64_t* lhsShape, \
        int64_t* rhsShapeMultiplier, \
        int64_t dimensions, \
        int64_t outputSize);
FOR_ALL_TYPES(INSTANTIATE)
#undef INSTANTIATE

} // namespace NHamKaas
