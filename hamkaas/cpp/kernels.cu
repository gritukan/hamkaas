#include <cstdint>

#include <cassert>
#include <cstdio>

#include "helpers.h"
#include "tensor.h"

namespace NHamKaas {

[[maybe_unused]] constexpr int64_t MaxThreadsPerBlock = 256;
[[maybe_unused]] constexpr int64_t MaxBlockCount = 65535;

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
    int64_t outputSize)
{
    // (lab4/03): Your code here.
}

#define INSTANTIATE_POINTWISE(Operation) \
    template void Pointwise<Operation>( \
        cudaStream_t stream, \
        const float* lhs, \
        const float* rhs, \
        float* out, \
        int64_t* lhsShape, \
        int64_t* rhsShape, \
        int64_t dimensions, \
        bool needBroadcasting, \
        int64_t outputSize);
INSTANTIATE_POINTWISE(EPointwiseOperation::Add)
INSTANTIATE_POINTWISE(EPointwiseOperation::HadamardProduct)
INSTANTIATE_POINTWISE(EPointwiseOperation::ReLU)
INSTANTIATE_POINTWISE(EPointwiseOperation::SiLU)
#undef INSTANTIATE_POINTWISE

} // namespace NHamKaas
