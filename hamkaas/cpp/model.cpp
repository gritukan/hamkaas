#include "model.h"

#include "allocator.h"
#include "error.h"

#include <cassert>
#include <cstring>
#include <unordered_set>
#include <unordered_map>

namespace NHamKaas {

TModel::TModel(TNodeBasePtr rootNode)
    : RootNode_(std::move(rootNode))
{ }

void TModel::Compile(const std::unordered_map<std::string, const void*>& constants)
{
    // NB: After this point, the model cannot be further modified.
    BuildEvaluationOrder();
    AllocateMemory();
    FillConstants(constants);
}

void TModel::Evaluate(const std::unordered_map<std::string, const void*>& inputs, void* output) const
{
    for (auto* node : EvaluationOrder_) {
        if (auto* inputNode = dynamic_cast<TInputNode*>(node)) {
            auto it = inputs.find(inputNode->GetName());
            if (it == inputs.end()) {
                THROW("Missing input", VAR(inputNode->GetName()));
            }
            auto* buffer = inputNode->GetOutput();
            std::memcpy(buffer, it->second, inputNode->GetOutputSize());
        } else {
            node->EvaluateCpu();
        }
    }

    std::memcpy(output, OutputBuffer_, RootNode_->GetOutputSize());
}

void TModel::BuildEvaluationOrder()
{
    std::vector<TNodeBase*> stack;
    std::unordered_set<TNodeBase*> visited;

    stack.push_back(RootNode_.get());
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (visited.count(node)) {
            EvaluationOrder_.push_back(node);
            continue;
        }
        visited.insert(node);
        stack.push_back(node);
        for (const auto& input : node->GetInputs()) {
            stack.push_back(input);
        }
    }
}

void TModel::AllocateMemory()
{
    // For each non-output node, stores a node such after its evaluation
    // the memory for its output can be freed.
    std::unordered_map<TNodeBase*, TNodeBase*> lastNodeOccurence;
    for (auto* node : EvaluationOrder_) {
        for (auto* input : node->GetInputs()) {
            // Input node may not be the owner of the output.
            // Let's find the real owner.
            while (input->GetOutputOwner() != input) {
                input = input->GetOutputOwner();
            }
            lastNodeOccurence[input] = node;
        }
    }

    // For node stores the list of outputs that should be freed after its evaluation.
    std::unordered_map<TNodeBase*, std::vector<TNodeBase*>> outputsToFree;
    for (const auto& [output, owner] : lastNodeOccurence) {
        outputsToFree[owner].push_back(output);
    }

    std::unordered_map<TNodeBase*, int64_t> bufferMemory;
    std::unordered_map<TNodeBase*, int64_t> outputMemory;

    TAllocator allocator;
    for (auto* node : EvaluationOrder_) {
        auto bufferSize = node->GetBufferSize();
        auto bufferPtr = allocator.Allocate(bufferSize);
        assert(bufferMemory.emplace(node, bufferPtr).second);

        auto outputSize = node->GetOutputSize();
        auto outputPtr = allocator.Allocate(outputSize);
        assert(outputMemory.emplace(node, outputPtr).second);

        // Buffer may be released immediately after evaluation.
        allocator.Free(bufferPtr, bufferSize);

        // Free outputs that are not needed anymore.
        for (auto* output : outputsToFree[node]) {
            allocator.Free(outputMemory[output], output->GetOutputSize());
        }
    }

    auto* baseAddress = static_cast<char*>(malloc(allocator.GetWorkingSetSize()));
    for (auto* node : EvaluationOrder_) {
        node->SetBuffer(baseAddress + bufferMemory[node]);
        node->SetOutput(baseAddress + outputMemory[node]);
    }
    OutputBuffer_ = baseAddress + outputMemory[RootNode_.get()];
}

void TModel::FillConstants(const std::unordered_map<std::string, const void*>& constants)
{
    for (auto* node : EvaluationOrder_) {
        if (auto* constantNode = dynamic_cast<TConstantNode*>(node)) {
            auto it = constants.find(constantNode->GetName());
            if (it == constants.end()) {
                THROW("Missing constant", VAR(constantNode->GetName()));
            }
            auto* buffer = constantNode->GetOutput();
            std::memcpy(buffer, it->second, constantNode->GetOutputSize());
        }
    }
}

} // namespace NHamKaas
