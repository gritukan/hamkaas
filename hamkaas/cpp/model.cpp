#include "model.h"

#include "allocator.h"
#include "device.h"
#include "error.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace NHamKaas {

TModel::TModel(const TBootstrap* bootstrap, TNodeBasePtr rootNode)
    : Bootstrap_(bootstrap)
    , RootNode_(std::move(rootNode))
{ }

TModel::~TModel()
{
    if (MemoryPool_) {
        if (UseGpu_) {
            CUDA_ASSERT(cudaFree(MemoryPool_));
        } else {
            free(MemoryPool_);
        }
    }
}

void TModel::Compile(
    const TCompilationOptions& options,
    const std::unordered_map<std::string, const char*>& constants)
{
    UseGpu_ = options.UseGpu;
    if (UseGpu_) {
        Device_ = CreateCudaDevice();
    } else {
        Device_ = CreateCpuDevice();
    }

    // NB: After this point, the model cannot be further modified.
    BuildEvaluationOrder();
    AllocateMemory();
    FillConstants(constants);
}

void TModel::Evaluate(
    const std::unordered_map<std::string, const char*>& inputs,
    char* output) const
{
    // Copy input tensors to input nodes.
    for (auto* inputNode : InputNodes_) {
        auto it = inputs.find(inputNode->GetName());
        if (it == inputs.end()) {
            THROW("Missing input", VAR(inputNode->GetName()));
        }

        auto* buffer = inputNode->GetOutput();
        Device_->CopyToDevice(buffer, it->second, inputNode->GetOutputSize());
    }

    TEvaluationContext context{
        .Bootstrap = Bootstrap_,
        .Device = Device_.get(),
    };

    // Now evaluate all the nodes in order.
    for (auto* node : EvaluationOrder_) {
        if (UseGpu_) {
            node->EvaluateGpu(context);
        } else {
            node->EvaluateCpu();
        }
    }

    // And copy the output tensor back.
    Device_->CopyToHost(output, RootNode_->GetOutput(), RootNode_->GetCapacity());
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

        // Input nodes are kinda special in terms of memory management
        // and evaluation, so we keep them separately.
        if (auto* inputNode = dynamic_cast<TInputNode*>(node)) {
            InputNodes_.push_back(inputNode);
        } else {
            EvaluationOrder_.push_back(node);
        }
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
    // of the model evaluation.    
    for (auto* inputNode : InputNodes_) {
        auto inputSize = inputNode->GetOutputSize();
        auto inputPtr = allocator.Allocate(inputSize);
        assert(outputMemory.emplace(inputNode, inputPtr).second);
    }

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

    MemoryPool_ = Device_->DeviceMalloc(2 * allocator.GetWorkingSetSize());
    std::cerr << "Allocated memory start = " << reinterpret_cast<int64_t>(MemoryPool_) << " end = " << reinterpret_cast<int64_t>(MemoryPool_ + 2 * allocator.GetWorkingSetSize()) << std::endl;
    for (auto* node : InputNodes_) {
        node->SetOutput(MemoryPool_ + outputMemory[node]);
    }
    for (auto* node : EvaluationOrder_) {
        node->SetBuffer(MemoryPool_ + bufferMemory[node]);
        node->SetOutput(MemoryPool_ + outputMemory[node]);
    }
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
            Device_->CopyToDevice(buffer, it->second, constantNode->GetOutputSize());
        }
    }
}

} // namespace NHamKaas
