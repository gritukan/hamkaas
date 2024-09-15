#pragma once

#include <cassert>
#include <cstdint>

namespace NHamKaas {

template <class T>
void SumTensorsBroadcast(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShapeMultiplier,
    int64_t dimensions,
    int64_t outputSize);

template <class T>
void ReLU(
    const T* input,
    T* output,
    int64_t size);

template <class T>
void SiLU(
    const T* input,
    T* output,
    int64_t size);

template <class T>
void RMSNorm(
    const T* input,
    const T* weights,
    T* output,
    int64_t size,
    T epsilon);

template <class T>
void ComplexHadamardProduct(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t size);

template <class T>
void HadamardProduct(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t size);

template <class T>
void SlicedSoftmax(
    const T* input,
    T* output,
    int64_t* prefixSizePtr,
    int64_t size);

template <class T>
void ReplaceSlice(
    T* input,
    int64_t inputSize,
    const T* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end);

template <class T>
void Permute(
    const T* input,
    T* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size);

} // namespace NHamKaas
