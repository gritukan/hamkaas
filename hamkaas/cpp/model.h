#pragma once

#include "node.h"

#include <unordered_map>

namespace NHamKaas {

class TModel
{
public:
    explicit TModel(TNodeBasePtr node);

    void Evaluate(const std::unordered_map<std::string, const void*>& inputs, void* output) const;

private:
    const TNodeBasePtr Node_;
};

} // namespace NHamKaas
