#include "helpers.h"

namespace NHamKaas {

int64_t Align(int64_t x)
{
    if (x % 64) {
        x += (64 - x % 64);
    }
    return x;
}

} // namespace NHamKaas
