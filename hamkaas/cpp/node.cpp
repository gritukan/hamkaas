#include "node.h"

#include "error.h"

#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstring>

namespace NHamKaas {

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

int TNodeBase::GetElementCount() const
{
    return Meta_.GetElementCount();
}

int TNodeBase::GetElementSize() const
{
    return Meta_.GetElementSize();
}

int TNodeBase::GetCapacity() const
{
    return Meta_.GetCapacity();
}

int64_t TNodeBase::GetBufferSize() const
{
    return 0;
}

int64_t TNodeBase::GetOutputSize() const
{
    return GetCapacity();
}

TNodeBase* TNodeBase::GetOutputOwner() const
{
    return const_cast<TNodeBase*>(this);
}

void TNodeBase::SetBuffer(void* /*buffer*/)
{
    // Do nothing.
}

void TNodeBase::SetOutput(void* output)
{
    Output_ = output;
}

void* TNodeBase::GetOutput() const
{
    return Output_;
}

TInputNode::TInputNode(std::string name, TTensorMeta meta)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TInputNode::GetName() const
{
    return Name_;
}

std::vector<TNodeBase*> TInputNode::GetInputs() const
{
    return {};
}

void TInputNode::EvaluateCpu() const
{
    // Do nothing; buffer is already set by the model evaluator.
}

TConstantNode::TConstantNode(TTensorMeta meta, std::string name)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TConstantNode::GetName() const
{
    return Name_;
}

std::vector<TNodeBase*> TConstantNode::GetInputs() const
{
    return {};
}

void TConstantNode::EvaluateCpu() const
{
    // Do nothing; buffer is already set by the model evaluator.
}

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

std::vector<TNodeBase*> TSumNode::GetInputs() const
{
    return {Lhs_.get(), Rhs_.get()};
}

void TSumNode::EvaluateCpu() const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    case EValueType::Int16:
        DoEvaluateCpu<int16_t>();
        return;
    case EValueType::Int32:
        DoEvaluateCpu<int32_t>();
        return;
    case EValueType::Int64:
        DoEvaluateCpu<int64_t>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TSumNode::DoEvaluateCpu() const
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int index = 0; index < Lhs_->GetElementCount(); ++index) {
        std::vector<int> rhsIndices(Rhs_->GetDimensions());

        int indexCopy = index;
        for (int index = Rhs_->GetDimensions() - 1; index >= 0; --index) {
            rhsIndices[index] = indexCopy % Lhs_->GetShape()[index];
            indexCopy /= Lhs_->GetShape()[index];
            if (rhsIndices[index] >= Rhs_->GetShape()[index]) {
                assert(Rhs_->GetShape()[index] == 1);
                rhsIndices[index] = 0;
            }
        }

        int rhsIndex = 0;
        for (int index = 0; index < Rhs_->GetDimensions(); ++index) {
            rhsIndex = rhsIndex * Rhs_->GetShape()[index] + rhsIndices[index];
        }

        output[index] = lhs[index] + rhs[rhsIndex];
    }
}

TTensorMeta TSumNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.Shape.size() != rhs.Shape.size()) {
        THROW("Different number of dimensions", VAR(lhs.Shape.size()), VAR(rhs.Shape.size()));
    }

    for (int index = 0; index < lhs.Shape.size(); ++index) {
        if (lhs.Shape[index] != rhs.Shape[index] && rhs.Shape[index] != 1) {
            THROW("Incompatible shapes for sum", VAR(index), VAR(lhs.Shape[index]), VAR(rhs.Shape[index]));
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

std::vector<TNodeBase*> TMulNode::GetInputs() const
{
    return {Lhs_.get(), Rhs_.get()};
}

TTensorMeta TMulNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.GetDimensions() != 2 || rhs.GetDimensions() != 2) {
        THROW("Matrix multiplication is supported only for 2D tensors", VAR(lhs.GetDimensions()), VAR(rhs.GetDimensions()));
    }

    if (lhs.Shape[1] != rhs.Shape[0]) {
        THROW("Incompatible shapes for matrix multiplication", VAR(lhs.Shape[1]), VAR(rhs.Shape[0]));
    }

    return TTensorMeta{
        .ValueType = lhs.ValueType,
        .Shape = {lhs.Shape[0], rhs.Shape[1]},
    };
}

void TMulNode::EvaluateCpu() const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    case EValueType::Int16:
        DoEvaluateCpu<int16_t>();
        return;
    case EValueType::Int32:
        DoEvaluateCpu<int32_t>();
        return;
    case EValueType::Int64:
        DoEvaluateCpu<int64_t>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TMulNode::DoEvaluateCpu() const
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    int n = Lhs_->GetShape()[0];
    int k = Lhs_->GetShape()[1];
    int m = Rhs_->GetShape()[1];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0;
            for (int index = 0; index < k; ++index) {
                sum += lhs[i * k + index] * rhs[index * m + j];
            }
            output[i * m + j] = sum;
        }
    }
}

TReLUNode::TReLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TReLUNode::GetInput() const
{
    return Input_;
}

std::vector<TNodeBase*> TReLUNode::GetInputs() const
{
    return {Input_.get()};
}

void TReLUNode::EvaluateCpu() const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    case EValueType::Int16:
        DoEvaluateCpu<int16_t>();
        return;
    case EValueType::Int32:
        DoEvaluateCpu<int32_t>();
        return;
    case EValueType::Int64:
        DoEvaluateCpu<int64_t>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TReLUNode::DoEvaluateCpu() const
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int index = 0; index < Input_->GetElementCount(); ++index) {
        output[index] = std::max<T>(0.0, input[index]);
    }
}

TSiLUNode::TSiLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TSiLUNode::GetInput() const
{
    return Input_;
}

std::vector<TNodeBase*> TSiLUNode::GetInputs() const
{
    return {Input_.get()};
}

void TSiLUNode::EvaluateCpu() const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    case EValueType::Int16:
        DoEvaluateCpu<int16_t>();
        return;
    case EValueType::Int32:
        DoEvaluateCpu<int32_t>();
        return;
    case EValueType::Int64:
        DoEvaluateCpu<int64_t>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TSiLUNode::DoEvaluateCpu() const
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int index = 0; index < Input_->GetElementCount(); ++index) {
        output[index] = input[index] / (1 + exp(-input[index]));
    }
}

} // namespace NHamKaas
