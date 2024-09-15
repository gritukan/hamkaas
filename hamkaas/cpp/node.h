#pragma once

#include "bootstrap.h"
#include "device.h"
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
    void ReplaceInput(TNodeBase* oldInput, std::shared_ptr<TNodeBase> newInput);

    // Memory management.

    // Number of bytes required during evaluation.
    virtual int64_t GetBufferSize() const;
    // Number of bytes required for output.
    virtual int64_t GetOutputSize() const;
    // Returns the node that allocates memory for output.
    // Typically this is the node itself, but for example reshape node
    // does not allocate any memory and just returns the input node with new meta.
    virtual TNodeBase* GetOutputOwner() const;

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

class TSumNode
    : public TNodeBase
{
public:
    TSumNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetBufferSize() const override;
    void SetBuffer(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    int64_t* LhsShape_;
    int64_t* RhsShape_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

class TMulNode
    : public TNodeBase
{
public:
    TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetBufferSize() const override;
    void SetBuffer(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    void** LhsMatrices_ = nullptr;
    void** RhsMatrices_ = nullptr;
    void** OutputMatrices_ = nullptr;

    char* TransposedProductBuffer_ = nullptr;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);

    struct TParameters
    {
        int64_t B;
        int64_t N;
        int64_t M;
        int64_t K;
    };
    TParameters GetParameters() const;
};

class TReLUNode
    : public TNodeBase
{
public:
    explicit TReLUNode(TNodeBasePtr input);

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

class TSiLUNode
    : public TNodeBase
{
public:
    explicit TSiLUNode(TNodeBasePtr input);

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    template <class T>
    void DoEvaluateCpu();

    template <typename T>
    void DoEvaluateGpu(const TEvaluationContext& context);
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

class TRmsNormNode
    : public TNodeBase
{
public:
    explicit TRmsNormNode(TNodeBasePtr input, TNodeBasePtr weights);

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
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

class TComplexHadamardProductNode
    : public TNodeBase
{
public:
    TComplexHadamardProductNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetBufferSize() const override;
    void SetBuffer(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    int64_t* LhsShape_;
    int64_t* RhsShape_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

class THadamardProductNode
    : public TNodeBase
{
public:
    THadamardProductNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    int64_t GetBufferSize() const override;
    void SetBuffer(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    int64_t* LhsShape_;
    int64_t* RhsShape_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

class TPermuteNode
    : public TNodeBase
{
public:
    TPermuteNode(TNodeBasePtr input, std::vector<int64_t> permutation);

    int64_t GetBufferSize() const override;
    void SetBuffer(char* buffer) override;

    void Initialize(IDevice* device) override;

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    const std::vector<int64_t> Permutation_;

    int64_t* InputShape_;
    int64_t* OutputShape_;
    int64_t* PermutationPtr_;

    bool Initialized_ = false;

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);

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

private:
    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

class TSlicedSoftmaxNode
    : public TNodeBase
{
public:
    TSlicedSoftmaxNode(TNodeBasePtr input, TNodeBasePtr prefixSize);

    void EvaluateCpu() override;
    void EvaluateGpu(const TEvaluationContext& context) override;

private:
    template <class T>
    void DoEvaluateCpu();

    template <class T>
    void DoEvaluateGpu(const TEvaluationContext& context);
};

} // namespace NHamKaas
