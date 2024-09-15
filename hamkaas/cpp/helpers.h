#pragma once

namespace NHamKaas {

struct TNonCopyable
{
    TNonCopyable() = default;
    TNonCopyable(const TNonCopyable&) = delete;
    TNonCopyable& operator=(const TNonCopyable&) = delete;
};

} // namespace NHamKaas