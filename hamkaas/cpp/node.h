#pragma once

#include "bootstrap.h"
#include "device.h"
#include "helpers.h"
#include "tensor.h"

#include <string>
#include <memory>
#include <unordered_map>

namespace NHamKaas {

struct TEvaluationContext
{
    const TBootstrap* Bootstrap;
    const IDevice* Device;

    cudaStream_t Stream = nullptr;
};

class TNodeBase
    : public std::enable_shared_from_this<TNodeBase>
{
public:
    explicit TNodeBase(TTensorMeta meta, std::vector<std::shared_ptr<TNodeBase>> inputs = {});

    virtual ~TNodeBase() = default;

    const TTensorMeta& GetMeta() const;

    EValueType GetValueType() const;
    int64_t GetDimensions() const;
    const std::vector<int64_t>& GetShape() const;
    int64_t GetElementCount() const;
    int64_t GetElementSize() const;
    int64_t GetCapacity() const;

    // Returns the list of nodes that are used as inputs for this node.
    // The order of nodes is important, inputs will be passed in this order.
    // Their outputs should be alive during evaluation.
    const std::vector<std::shared_ptr<TNodeBase>>& GetInputs() const;
    void ReplaceInput(std::shared_ptr<TNodeBase> oldInput, std::shared_ptr<TNodeBase> newInput);

    // Memory management.

    // Number of constant memory required for the node.
    virtual int64_t GetConstantMemorySize() const;
    // Number of bytes required during evaluation.
    virtual int64_t GetBufferSize() const;
    // Number of bytes required for output.
    virtual int64_t GetOutputSize() const;
    // Returns the node that allocates memory for output.
    // Typically this is the node itself, but for example reshape node
    // does not allocate any memory and just returns the input node with new meta.
    virtual TNodeBase* GetOutputOwner() const;

    // Sets the pointer to the constant memory for the node.
    virtual void SetConstantMemory(char* constantMemory);

    // Sets the pointer to the buffer that can be used during node evaluation.
    virtual void SetBuffer(char* buffer);

    // Sets the pointer to the output buffer that is alive during node evaluation
    // and the result usage.
    void SetOutput(char* output);

    // Returns the pointer to the output buffer.
    char* GetOutput() const;

    // Called before evaluation to initialize the node.
    virtual void Initialize(IDevice* device);

    virtual void EvaluateCpu() = 0;
    virtual void EvaluateGpu(const TEvaluationContext& context) = 0;

protected:
    const TTensorMeta Meta_;

    char* Output_ = nullptr;

    std::vector<std::shared_ptr<TNodeBase>> Inputs_;
};

using TNodeBasePtr = std::shared_ptr<TNodeBase>;

class TInputNode
    : public TNodeBase
{
public:
    TInputNode(std::string name, TTensorMeta meta);

    const std::string& GetName() const;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const std::string Name_;
};

class TBufferNode
    : public TNodeBase
{
public:
    explicit TBufferNode(TTensorMeta meta);

    TNodeBase* GetOutputOwner() const override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;
};

class TConstantNode
    : public TNodeBase
{
public:
    TConstantNode(TTensorMeta meta, std::string name);

    const std::string& GetName() const;

    TNodeBase* GetOutputOwner() const override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const std::string Name_;
};

template <EPointwiseOperation Operation>
class TPointwiseNode
    : public TNodeBase
{
public:
    explicit TPointwiseNode(TNodeBasePtr lhs);
    TPointwiseNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetConstantMemorySize() const override;
    void SetConstantMemory(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    int64_t* LhsShape_;
    int64_t* RhsShape_;

    bool NeedBroadcasting_ = false;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    void DoEvaluateCpu(const float* lhsPtr, float* outputPtr) const;
    void DoEvaluateCpu(const float* lhsPtr, const float* rhsPtr, float* outputPtr) const;
};

class TSumNode
    : public TPointwiseNode<EPointwiseOperation::Add>
{
public:
    using TPointwiseNode::TPointwiseNode;
};

class THadamardProductNode
    : public TPointwiseNode<EPointwiseOperation::HadamardProduct>
{
public:
    using TPointwiseNode::TPointwiseNode;
};

class TReLUNode
    : public TPointwiseNode<EPointwiseOperation::ReLU>
{
public:
    using TPointwiseNode::TPointwiseNode;
};

class TSiLUNode
    : public TPointwiseNode<EPointwiseOperation::SiLU>
{
public:
    using TPointwiseNode::TPointwiseNode;
};

class TMatMulNode
    : public TNodeBase
{
public:
    TMatMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetConstantMemorySize() const override;
    void SetConstantMemory(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    struct TParameters
    {
        int64_t B;
        int64_t N;
        int64_t M;
        int64_t K;
    };
    TParameters GetParameters() const;
};

class TSliceNode
    : public TNodeBase
{
public:
    TSliceNode(TNodeBasePtr input, int64_t begin, int64_t end);

    int64_t GetBegin() const;
    int64_t GetEnd() const;

    int64_t GetOutputSize() const override;
    TNodeBase* GetOutputOwner() const override;

    void Initialize(IDevice* /*device*/) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const int64_t Begin_;
    const int64_t End_;

    size_t Stride_;

    static TTensorMeta CalculateMeta(const TTensorMeta& input, int64_t begin, int64_t end);
};

class TReshapeNode
    : public TNodeBase
{
public:
    TReshapeNode(TNodeBasePtr input, std::vector<int64_t> shape);

    int64_t GetOutputSize() const override;
    TNodeBase* GetOutputOwner() const override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const std::vector<int64_t> Shape_;

    static TTensorMeta CalculateMeta(const TTensorMeta& input, const std::vector<int64_t>& shape);
};

class TPermuteNode
    : public TNodeBase
{
public:
    TPermuteNode(TNodeBasePtr input, std::vector<int64_t> permutation);

    int64_t GetConstantMemorySize() const override;
    void SetConstantMemory(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const std::vector<int64_t> Permutation_;

    static TTensorMeta CalculateMeta(const TTensorMeta& input, const std::vector<int64_t>& permutation);
};

class TReplaceSliceNode
    : public TNodeBase
{
public:
    TReplaceSliceNode(
        TNodeBasePtr input,
        TNodeBasePtr replacement,
        TNodeBasePtr begin,
        TNodeBasePtr end);

    int64_t GetOutputSize() const override;
    TNodeBase* GetOutputOwner() const override;

    void Initialize(IDevice* /*device*/) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;
};

} // namespace NHamKaas
