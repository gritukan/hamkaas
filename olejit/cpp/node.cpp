#include "node.h"

#include <stdexcept>

TNodeBase::TNodeBase(TTensorMeta meta)
    : Meta_(std::move(meta))
{ }

const TTensorMeta& TNodeBase::GetMeta() const
{
    return Meta_;
}

EValueType TNodeBase::GetValueType() const
{
    return Meta_.ValueType;
}

int TNodeBase::GetDimensions() const
{
    return Meta_.GetDimensions();
}

const std::vector<int>& TNodeBase::GetShape() const
{
    return Meta_.Shape;
}

TInputNode::TInputNode(std::string name, TTensorMeta meta)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TInputNode::GetName() const
{
    return Name_;
}

TConstantNode::TConstantNode(TTensorMeta meta, void* data)
    : TNodeBase(std::move(meta))
    , Data_(data)
{ }

TSumNode::TSumNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()))
    , Lhs_(std::move(lhs))
    , Rhs_(std::move(rhs))
{ }

const TNodeBasePtr& TSumNode::GetLhs() const
{
    return Lhs_;
}

const TNodeBasePtr& TSumNode::GetRhs() const
{
    return Rhs_;
}

TTensorMeta TSumNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        throw std::runtime_error("Different value types");
    }

    if (lhs.Shape.size() != rhs.Shape.size()) {
        throw std::runtime_error("Different number of dimensions");
    }

    for (int index = 0; index < lhs.Shape.size(); ++index) {
        if (lhs.Shape[index] != rhs.Shape[index] && rhs.Shape[index] != 1) {
            throw std::runtime_error("Incompatible shapes for sum");
        }
    }

    return lhs;
}

TMulNode::TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()))
    , Lhs_(std::move(lhs))
    , Rhs_(std::move(rhs))
{ }

const TNodeBasePtr& TMulNode::GetLhs() const
{
    return Lhs_;
}

const TNodeBasePtr& TMulNode::GetRhs() const
{
    return Rhs_;
}

TTensorMeta TMulNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        throw std::runtime_error("Different value types");
    }

    if (lhs.GetDimensions() != 2 || rhs.GetDimensions() != 2) {
        throw std::runtime_error("Matrix multiplication is supported only for 2D tensors");
    }

    if (lhs.Shape[1] != rhs.Shape[0]) {
        throw std::runtime_error("Incompatible shapes for matrix multiplication");
    }

    return TTensorMeta{
        .ValueType = lhs.ValueType,
        .Shape = {lhs.Shape[0], rhs.Shape[1]},
    };
}

TReLUNode::TReLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TReLUNode::GetInput() const
{
    return Input_;
}

TSiLUNode::TSiLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TSiLUNode::GetInput() const
{
    return Input_;
}
