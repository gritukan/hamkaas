#include "model.h"

#include "allocator.h"
#include "error.h"

#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace NHamKaas {

TModel::TModel(TNodeBasePtr rootNode)
    : RootNode_(std::move(rootNode))
{ }

void TModel::Compile(const std::unordered_map<std::string, const char*>& constants)
{
    // NB: After this point, the model cannot be further modified.
    BuildEvaluationOrder();
    AllocateMemory();
    FillConstants(constants);
}

void TModel::Evaluate(const std::unordered_map<std::string, const char*>& inputs, char* output) const
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

    std::memcpy(output, RootNode_->GetOutput(), RootNode_->GetCapacity());
}

void TModel::BuildEvaluationOrder()
{
    std::unordered_set<TNodeBase*> visited;
    std::function<void(TNodeBase*)> dfs = [&] (TNodeBase* node) -> void {
        if (visited.count(node)) {
            return;
        }
        visited.insert(node);

        for (const auto& input : node->GetInputs()) {
            dfs(input);
        }

        EvaluationOrder_.push_back(node);
    };

    dfs(RootNode_.get());
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
            while (input && input->GetOutputOwner() != input) {
                input = input->GetOutputOwner();
            }
            if (input) {
                lastNodeOccurence[input] = node;
            }
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

    // Output buffers for input nodes should be available at the beginning
    // in order to copy input data before execution. At first pass, we allocate
    // output buffers for input nodes, on the second pass for all other nodes.
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (auto* node : EvaluationOrder_) {
            auto isInputNode = dynamic_cast<TInputNode*>(node) != nullptr;
            if (isInputNode != (iteration == 0)) {
                continue;
            }

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
    }

    auto* baseAddress = static_cast<char*>(malloc(allocator.GetWorkingSetSize()));
    memset(baseAddress, 0, allocator.GetWorkingSetSize());
    for (auto* node : EvaluationOrder_) {
        node->SetBuffer(baseAddress + bufferMemory[node]);
        node->SetOutput(baseAddress + outputMemory[node]);
    }
    OutputBuffer_ = baseAddress + outputMemory[RootNode_.get()];
}

void TModel::FillConstants(const std::unordered_map<std::string, const char*>& constants)
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
