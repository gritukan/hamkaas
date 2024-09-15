#include "tensor.h"

#include <cassert>

namespace NHamKaas {

int64_t TTensorMeta::GetDimensions() const
{
    return static_cast<int64_t>(Shape.size());
}

int64_t TTensorMeta::GetElementCount() const
{
    int64_t count = 1;
    for (int64_t size : Shape) {
        count *= size;
    }

    return count;
}

int64_t TTensorMeta::GetElementSize() const
{
    switch (ValueType) {
    case EValueType::Float32:
        return 4;
    case EValueType::Float64:
        return 8;
    case EValueType::Int64:
        return 8;
    default:
        assert(false);
    }
}

int64_t TTensorMeta::GetCapacity() const
{
    return GetElementCount() * GetElementSize();
}

} // namespace NHamKaas
