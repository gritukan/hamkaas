#pragma once

#include <cstdint>

namespace NHamKaas {

enum class EPointwiseOperation
{
    Add,
    HadamardProduct,
    ReLU,
    SiLU,
};

constexpr bool IsBinary(EPointwiseOperation operation)
{
    return
        operation == EPointwiseOperation::Add ||
        operation == EPointwiseOperation::HadamardProduct;
}

constexpr bool IsBroadcastingSupported(EPointwiseOperation operation)
{
    return
        operation == EPointwiseOperation::Add ||
        operation == EPointwiseOperation::HadamardProduct;
}

struct TNonCopyable
{
    TNonCopyable() = default;
    TNonCopyable(const TNonCopyable&) = delete;
    TNonCopyable& operator=(const TNonCopyable&) = delete;
};

} // namespace NHamKaas
