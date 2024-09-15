#include "tensor.h"

int TTensorMeta::GetDimensions() const
{
    return static_cast<int>(Shape.size());
}
