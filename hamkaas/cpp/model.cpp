#include "model.h"

#include <cstring>

namespace NHamKaas {

TModel::TModel(TNodeBasePtr node)
    : Node_(std::move(node))
{ }

void TModel::Evaluate(const std::unordered_map<std::string, const void*>& inputs, void* output) const
{
    auto result = Node_->EvaluateCpu(inputs);
    memcpy(output, result.data(), result.size());
}

} // namespace NHamKaas
