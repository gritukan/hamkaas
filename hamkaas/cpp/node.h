#pragma once

#include "tensor.h"

#include <string>
#include <memory>
#include <unordered_map>

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

    virtual std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const = 0;

private:
    const TTensorMeta Meta_;
};

using TNodeBasePtr = std::shared_ptr<TNodeBase>;

class TInputNode
    : public TNodeBase
{
public:
    TInputNode(std::string name, TTensorMeta meta);

    const std::string& GetName() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const std::string Name_;
};

class TConstantNode
    : public TNodeBase
{
public:
    TConstantNode(TTensorMeta meta, const void* data);

    const void* GetData() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const void* Data_;
};

class TSumNode
    : public TNodeBase
{
public:
    TSumNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    const TNodeBasePtr& GetLhs() const;
    const TNodeBasePtr& GetRhs() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    std::vector<char> DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const;
};

class TMulNode
    : public TNodeBase
{
public:
    TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    const TNodeBasePtr& GetLhs() const;
    const TNodeBasePtr& GetRhs() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);

    template <class T>
    std::vector<char> DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const;
};

class TReLUNode
    : public TNodeBase
{
public:
    explicit TReLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const TNodeBasePtr Input_;

    template <class T>
    std::vector<char> DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const;
};

class TSiLUNode
    : public TNodeBase
{
public:
    explicit TSiLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

    std::vector<char> EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const override;

private:
    const TNodeBasePtr Input_;

    template <class T>
    std::vector<char> DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const;
};
