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
    case EValueType::Float16:
        return 2;
    case EValueType::Float32:
        return 4;
    case EValueType::Float64:
        return 8;
    case EValueType::Int16:
        return 2;
    case EValueType::Int32:
        return 4;
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

TTensor::TTensor(TTensorMeta meta, char* data)
    : Meta_(meta)
    , Data_(data)
{ }

const TTensorMeta& TTensor::GetMeta() const
{
    return Meta_;
}

EValueType TTensor::GetValueType() const
{
    return Meta_.ValueType;
}

int64_t TTensor::GetDimensions() const
{
    return Meta_.GetDimensions();
}

const std::vector<int64_t>& TTensor::GetShape() const
{
    return Meta_.Shape;
}

int64_t TTensor::GetElementCount() const
{
    return Meta_.GetElementCount();
}

int64_t TTensor::GetElementSize() const
{
    return Meta_.GetElementSize();
}

int64_t TTensor::GetCapacity() const
{
    return Meta_.GetCapacity();
}

const char* TTensor::GetData() const
{
    return Data_;
}

char* TTensor::GetData()
{
    return Data_;
}

} // namespace NHamKaas
