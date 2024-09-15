#pragma once

#include <cstdint>

namespace NHamKaas {

template <typename T>
void SumTensorsBroadcast(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShapeMultiplier,
    int64_t dimensions,
    int64_t outputSize);

} // namespace NHamKaas
