#pragma once

#include <cassert>
#include <cstdint>

namespace NHamKaas {

void SumTensorsBroadcast(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

void ReLU(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t size);

void SiLU(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t size);

void RMSNorm(
    cudaStream_t stream,
    const float* input,
    const float* weights,
    float* output,
    int64_t size,
    float epsilon);

void ComplexHadamardProductBroadcast(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

void HadamardProductBroadcast(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

void SlicedSoftmax(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize);

void ReplaceSlice(
    cudaStream_t stream,
    float* input,
    int64_t inputSize,
    const float* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end);

void Permute(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size);

} // namespace NHamKaas
