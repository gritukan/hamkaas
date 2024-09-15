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

#ifdef USE_CUDNN
#include "cudnn_optimizer.h"
#endif

namespace NHamKaas {

TModel::TModel(const TBootstrap* bootstrap, TNodeBasePtr rootNode)
    : Bootstrap_(bootstrap)
    , RootNode_(std::move(rootNode))
{ }

TModel::~TModel()
{
    if (Stream_) {
        CUDA_ASSERT(cudaStreamDestroy(Stream_));
    }
    if (Graph_) {
        CUDA_ASSERT(cudaGraphDestroy(Graph_));
    }
    if (GraphExec_) {
        CUDA_ASSERT(cudaGraphExecDestroy(GraphExec_));
    }

    // (lab4/02): Your code here: free allocated memory.
}

void TModel::Compile(
    const TCompilationOptions& options,
    const std::unordered_map<std::string, const char*>& constants)
{
    UseGpu_ = options.UseGpu;
    if (UseGpu_) {
        CUDA_CHECK_ERROR(cudaStreamCreate(&Stream_));
        Device_ = CreateCudaDevice(Stream_);
    } else {
        Device_ = CreateCpuDevice();
    }

    if (options.UseCudnn) {
#ifdef USE_CUDNN
        RootNode_ = RunCudnnOptimizer(RootNode_, Bootstrap_);
#else
        THROW("HamKaas was compiled without CUDNN support");
#endif
    }

    // NB: After this point, the model cannot be further modified.
    BuildEvaluationOrder();
    AllocateMemory();
    FillConstants(constants);
    InitializeNodes();

    // If GPU is used, we precompile a CUDA graph to evaluate it later.
    if (UseGpu_) {
        CUDA_CHECK_ERROR(cudaStreamBeginCapture(Stream_, cudaStreamCaptureModeGlobal));

        for (auto* node : EvaluationOrder_) {
            node->EvaluateGpu(TEvaluationContext{
                .Bootstrap = Bootstrap_,
                .Device = Device_.get(),
                .Stream = Stream_,
            });
        }

        CUDA_CHECK_ERROR(cudaGraphCreate(&Graph_, 0));
        CUDA_CHECK_ERROR(cudaStreamEndCapture(Stream_, &Graph_));
        CUDA_CHECK_ERROR(cudaGraphInstantiate(&GraphExec_, Graph_, nullptr, nullptr, 0));
    }
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
        Device_->CopyToDevice(buffer, it->second, inputNode->GetOutputSize(), /*sync*/ false);
    }

    if (UseGpu_) {
        CUDA_CHECK_ERROR(cudaGraphLaunch(GraphExec_, Stream_));
    } else {
        for (auto* node : EvaluationOrder_) {
            node->EvaluateCpu();
        }
    }

    // And copy the output tensor back.
    Device_->CopyToHost(output, RootNode_->GetOutput(), RootNode_->GetCapacity());
}

void TModel::BuildEvaluationOrder()
{
    std::unordered_set<const TNodeBase*> visited;
    std::function<void(TNodeBase*)> dfs = [&] (TNodeBase* node) -> void {
        if (visited.count(node)) {
            return;
        }
        visited.insert(node);

        for (const auto& input : node->GetInputs()) {
            dfs(input.get());
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
    // (lab4/02): Your code here: allocate output and buffer memory for all nodes.
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

void TModel::InitializeNodes()
{
    for (auto* node : EvaluationOrder_) {
        node->Initialize(Device_.get());
    }
}

} // namespace NHamKaas
