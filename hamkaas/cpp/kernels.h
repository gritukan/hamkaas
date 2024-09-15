#pragma once

#include <cassert>
#include <cstdint>

namespace NHamKaas {

template <EPointwiseOperation Operation>
void Pointwise(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* out,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    bool needBroadcasting,
    int64_t outputSize);

void RMSNorm(
    cudaStream_t stream,
    const float* input,
    const float* weights,
    float* output,
    int64_t size,
    float epsilon);

void SlicedSoftmax(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize);

} // namespace NHamKaas
