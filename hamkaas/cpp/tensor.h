#pragma once

#include <cstdint>
#include <vector>

namespace NHamKaas {

constexpr int64_t MaxDimensions = 3;

enum EValueType
{
    Float32,
    Float64,
    Int64,
};

struct TTensorMeta
{
    EValueType ValueType;

    // TODO: Replace it with compact vector.
    std::vector<int64_t> Shape;

    int64_t GetDimensions() const;
    int64_t GetElementCount() const;
    int64_t GetElementSize() const;
    int64_t GetCapacity() const;
};

} // namespace NHamKaas
