#pragma once

#include <cstdint>

namespace NHamKaas {

struct TNonCopyable
{
    TNonCopyable() = default;
    TNonCopyable(const TNonCopyable&) = delete;
    TNonCopyable& operator=(const TNonCopyable&) = delete;
};

int64_t Align(int64_t x);

#define FOR_ALL_TYPES(XX) \
    XX(float) \
    XX(double) \

} // namespace NHamKaas