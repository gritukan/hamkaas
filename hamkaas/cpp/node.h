#pragma once

#include "tensor.h"

#include <string>
#include <memory>
#include <unordered_map>

namespace NHamKaas {

class TNodeBase
{
public:
    TNodeBase(TTensorMeta meta);

    virtual ~TNodeBase() = default;

    const TTensorMeta& GetMeta() const;

    EValueType GetValueType() const;
    int GetDimensions() const;
    const std::vector<int>& GetShape() const;
    int GetElementCount() const;
    int GetElementSize() const;
    int GetCapacity() const;

    // Returns the list of nodes that are used as inputs for this node.
    // The order of nodes is important, inputs will be passed in this order.
    // Their outputs should be alive during evaluation.
    virtual std::vector<TNodeBase*> GetInputs() const = 0;

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
    virtual void SetBuffer(void* buffer);

    // Sets the pointer to the output buffer that is alive during node evaluation
    // and the result usage.
    void SetOutput(void* output);

    // Returns the pointer to the output buffer.
    void* GetOutput() const;

    virtual void EvaluateCpu() const = 0;

private:
    const TTensorMeta Meta_;

    void* Output_ = nullptr;
};

using TNodeBasePtr = std::shared_ptr<TNodeBase>;

class TInputNode
    : public TNodeBase
{
public:
    TInputNode(std::string name, TTensorMeta meta);

    const std::string& GetName() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const std::string Name_;
};

class TConstantNode
    : public TNodeBase
{
public:
    TConstantNode(TTensorMeta meta, std::string name);

    const std::string& GetName() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const std::string Name_;
};

class TSumNode
    : public TNodeBase
{
public:
    TSumNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    const TNodeBasePtr& GetLhs() const;
    const TNodeBasePtr& GetRhs() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu() const;
};

class TMulNode
    : public TNodeBase
{
public:
    TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    const TNodeBasePtr& GetLhs() const;
    const TNodeBasePtr& GetRhs() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    void DoEvaluateCpu() const;
};

class TReLUNode
    : public TNodeBase
{
public:
    explicit TReLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const TNodeBasePtr Input_;

    template <class T>
    void DoEvaluateCpu() const;
};

class TSiLUNode
    : public TNodeBase
{
public:
    explicit TSiLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

    std::vector<TNodeBase*> GetInputs() const override;

    void EvaluateCpu() const override;

private:
    const TNodeBasePtr Input_;

    template <class T>
    void DoEvaluateCpu() const;
};

} // namespace NHamKaas
