#pragma once

#include <vector>

namespace NHamKaas {

constexpr int MaxDimensions = 3;

enum EValueType
{
    Float16,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
};

struct TTensorMeta
{
    EValueType ValueType;

    // TODO: Replace it with compact vector.
    std::vector<int> Shape;

    int GetDimensions() const;
    int GetElementCount() const;
    int GetElementSize() const;
    int GetCapacity() const;
};

class TTensor
{
public:
    TTensor(TTensorMeta meta, void* data);

    const TTensorMeta& GetMeta() const;
    EValueType GetValueType() const;
    int GetDimensions() const;
    const std::vector<int>& GetShape() const;
    int GetElementCount() const;
    int GetElementSize() const;
    int GetCapacity() const;

    const void* GetData() const;
    void* GetData();

private:
    TTensorMeta Meta_;
    void* Data_;
};

} // namespace NHamKaas
