#pragma once

#include <cassert>
#include <cstdint>

namespace NHamKaas {

template <class T>
void SumTensorsBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

template <class T>
void ReLU(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t size);

template <class T>
void SiLU(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t size);

template <class T>
void RMSNorm(
    cudaStream_t stream,
    const T* input,
    const T* weights,
    T* output,
    int64_t size,
    T epsilon);

template <class T>
void ComplexHadamardProductBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

template <class T>
void HadamardProductBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize);

template <class T>
void SlicedSoftmax(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize);

template <class T>
void ReplaceSlice(
    cudaStream_t stream,
    T* input,
    int64_t inputSize,
    const T* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end);

template <class T>
void Permute(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size);

} // namespace NHamKaas
