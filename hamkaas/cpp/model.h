#pragma once

#include "bootstrap.h"
#include "node.h"
#include "device.h"

#include "interface.h"

#include <unordered_map>

namespace NHamKaas {

class TModel
{
public:
    TModel(const TBootstrap* bootstrap, TNodeBasePtr rootNode);
    ~TModel();

    void Compile(
        const TCompilationOptions& options,
        const std::unordered_map<std::string, const char*>& constants);

    void Evaluate(
        const std::unordered_map<std::string, const char*>& inputs,
        char* output) const;

private:
    const TBootstrap* Bootstrap_;

    TNodeBasePtr RootNode_;

    bool UseGpu_ = false;

    std::unique_ptr<IDevice> Device_;

    char* MemoryPool_ = nullptr;

    cudaStream_t Stream_ = nullptr;
    cudaGraph_t Graph_ = nullptr;
    cudaGraphExec_t GraphExec_ = nullptr;

    std::vector<TInputNode*> InputNodes_;
    std::vector<TNodeBase*> EvaluationOrder_;

    void BuildEvaluationOrder();
    void AllocateMemory();
    void FillConstants(const std::unordered_map<std::string, const char*>& constants);
    void InitializeNodes();
};

} // namespace NHamKaas
