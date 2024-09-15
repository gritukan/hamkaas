#pragma once

#include "node.h"

#include <unordered_map>

namespace NHamKaas {

class TModel
{
public:
    explicit TModel(TNodeBasePtr rootNode);

    void Compile(const std::unordered_map<std::string, const void*>& constants);

    void Evaluate(const std::unordered_map<std::string, const void*>& inputs, void* output) const;

private:
    const TNodeBasePtr RootNode_;

    // Memory allocated for the model output.
    void* OutputBuffer_;

    std::vector<TNodeBase*> EvaluationOrder_;

    void BuildEvaluationOrder();
    void AllocateMemory();
    void FillConstants(const std::unordered_map<std::string, const void*>& constants);
};

} // namespace NHamKaas
