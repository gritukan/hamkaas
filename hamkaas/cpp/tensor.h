#include <vector>

constexpr int MaxDimensions = 3;

enum EValueType
{
    Float16,
    Float32,
    Float64,
};

struct TTensorMeta
{
    EValueType ValueType;

    // TODO: Replace it with compact vector.
    std::vector<int> Shape;

    int GetDimensions() const;
};
