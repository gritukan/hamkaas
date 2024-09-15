#pragma once

#include "helpers.h"

#include <cstdint>

namespace NHamKaas {

class TAllocator
{
public:
    int64_t Allocate(int64_t size);
    void Free(int64_t ptr, int64_t size);

    // Returns the amount of memory required to satisfy all allocations.
    int64_t GetWorkingSetSize() const;

private:
    int64_t Offset_ = 0;
};

} // namespace NHamKaas