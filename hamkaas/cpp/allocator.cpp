#include "allocator.h"

namespace NHamKaas {

int64_t TAllocator::Allocate(int64_t size)
{
    if (size % 64) {
        size += (64 - size % 64);
    }

    int64_t ptr = Offset_;
    Offset_ += size;
    return ptr;
}

void TAllocator::Free(int64_t /*ptr*/, int64_t /*size*/)
{
    // Do nothing.
}

int64_t TAllocator::GetWorkingSetSize() const
{
    return Offset_;
}

} // namespace NHamKaas
