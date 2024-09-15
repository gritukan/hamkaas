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

} // namespace NHamKaas
