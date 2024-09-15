#pragma once

#include <cstdint>

namespace NHamKaas {

enum class EPointwiseOperation
{
    Add,
    HadamardProduct,
    ComplexHadamardProduct,
    ReLU,
    SiLU,
};

constexpr bool IsBinary(EPointwiseOperation operation)
{
    return
        operation == EPointwiseOperation::Add ||
        operation == EPointwiseOperation::HadamardProduct ||
        operation == EPointwiseOperation::ComplexHadamardProduct;
}

constexpr bool IsBroadcastingSupported(EPointwiseOperation operation)
{
    return
        operation == EPointwiseOperation::Add ||
        operation == EPointwiseOperation::HadamardProduct ||
        operation == EPointwiseOperation::ComplexHadamardProduct;
}

struct TNonCopyable
{
    TNonCopyable() = default;
    TNonCopyable(const TNonCopyable&) = delete;
    TNonCopyable& operator=(const TNonCopyable&) = delete;
};

int64_t Align(int64_t x);

} // namespace NHamKaas
