#include "tensor.h"

#include <string>
#include <memory>

class TNodeBase
{
public:
    TNodeBase(TTensorMeta meta);

    virtual ~TNodeBase() = default;

    const TTensorMeta& GetMeta() const;

    EValueType GetValueType() const;
    int GetDimensions() const;
    const std::vector<int>& GetShape() const;

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

private:
    const std::string Name_;
};

class TConstantNode
    : public TNodeBase
{
public:
    TConstantNode(TTensorMeta meta, void* data);

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

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);
};

class TMulNode
    : public TNodeBase
{
public:
    TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs);

    const TNodeBasePtr& GetLhs() const;
    const TNodeBasePtr& GetRhs() const;

private:
    const TNodeBasePtr Lhs_;
    const TNodeBasePtr Rhs_;

    static TTensorMeta CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs);
};

class TReLUNode
    : public TNodeBase
{
public:
    TReLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

private:
    const TNodeBasePtr Input_;
};

class TSiLUNode
    : public TNodeBase
{
public:
    TSiLUNode(TNodeBasePtr input);

    const TNodeBasePtr& GetInput() const;

private:
    const TNodeBasePtr Input_;
};
