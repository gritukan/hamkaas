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

TInputNode::TInputNode(std::string name, TTensorMeta meta)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TInputNode::GetName() const
{
    return Name_;
}

std::vector<char> TInputNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    auto inputIt = inputs.find(Name_);
    if (inputIt == inputs.end()) {
        THROW("Input not found", VAR(Name_));
    }

    const auto* input = inputIt->second;
    std::vector<char> result(GetCapacity());

    memcpy(result.data(), input, GetCapacity());

    return result;
}

TConstantNode::TConstantNode(TTensorMeta meta, const void* data)
    : TNodeBase(std::move(meta))
    , Data_(data)
{ }

const void* TConstantNode::GetData() const
{
    return Data_;
}

std::vector<char> TConstantNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    std::vector<char> result(GetCapacity());
    memcpy(result.data(), Data_, GetCapacity());
    return result;
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

std::vector<char> TSumNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        return DoEvaluateCpu<float>(inputs);
    case EValueType::Float64:
        return DoEvaluateCpu<double>(inputs);
    case EValueType::Int16:
        return DoEvaluateCpu<int16_t>(inputs);
    case EValueType::Int32:
        return DoEvaluateCpu<int32_t>(inputs);
    case EValueType::Int64:
        return DoEvaluateCpu<int64_t>(inputs);
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
std::vector<char> TSumNode::DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    auto lhsResult = Lhs_->EvaluateCpu(inputs);
    auto rhsResult = Rhs_->EvaluateCpu(inputs);

    const auto* lhs = reinterpret_cast<const T*>(lhsResult.data());
    const auto* rhs = reinterpret_cast<const T*>(rhsResult.data());

    std::vector<char> resultData(lhsResult.size());
    auto* result = reinterpret_cast<T*>(resultData.data());

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

        result[index] = lhs[index] + rhs[rhsIndex];
    }

    return resultData;
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

std::vector<char> TMulNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        return DoEvaluateCpu<float>(inputs);
    case EValueType::Float64:
        return DoEvaluateCpu<double>(inputs);
    case EValueType::Int16:
        return DoEvaluateCpu<int16_t>(inputs);
    case EValueType::Int32:
        return DoEvaluateCpu<int32_t>(inputs);
    case EValueType::Int64:
        return DoEvaluateCpu<int64_t>(inputs);
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
std::vector<char> TMulNode::DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    auto lhsResult = Lhs_->EvaluateCpu(inputs);
    auto rhsResult = Rhs_->EvaluateCpu(inputs);

    const auto* lhs = reinterpret_cast<const T*>(lhsResult.data());
    const auto* rhs = reinterpret_cast<const T*>(rhsResult.data());

    int n = Lhs_->GetShape()[0];
    int k = Lhs_->GetShape()[1];
    int m = Rhs_->GetShape()[1];

    std::vector<char> resultData(n * m * sizeof(T));
    auto* result = reinterpret_cast<T*>(resultData.data());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0;
            for (int index = 0; index < k; ++index) {
                sum += lhs[i * k + index] * rhs[index * m + j];
            }
            result[i * m + j] = sum;
        }
    }

    return resultData;
}

TReLUNode::TReLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TReLUNode::GetInput() const
{
    return Input_;
}

std::vector<char> TReLUNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        return DoEvaluateCpu<float>(inputs);
    case EValueType::Float64:
        return DoEvaluateCpu<double>(inputs);
    case EValueType::Int16:
        return DoEvaluateCpu<int16_t>(inputs);
    case EValueType::Int32:
        return DoEvaluateCpu<int32_t>(inputs);
    case EValueType::Int64:
        return DoEvaluateCpu<int64_t>(inputs);
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
std::vector<char> TReLUNode::DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    auto inputResult = Input_->EvaluateCpu(inputs);
    const auto* input = reinterpret_cast<const T*>(inputResult.data());

    std::vector<char> resultData(inputResult.size());
    auto* result = reinterpret_cast<T*>(resultData.data());

    for (int index = 0; index < Input_->GetElementCount(); ++index) {
        result[index] = std::max<T>(0.0, input[index]);
    }

    return resultData;
}

TSiLUNode::TSiLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
{ }

const TNodeBasePtr& TSiLUNode::GetInput() const
{
    return Input_;
}

std::vector<char> TSiLUNode::EvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    switch (GetValueType()) {
    case EValueType::Float32:
        return DoEvaluateCpu<float>(inputs);
    case EValueType::Float64:
        return DoEvaluateCpu<double>(inputs);
    case EValueType::Int16:
        return DoEvaluateCpu<int16_t>(inputs);
    case EValueType::Int32:
        return DoEvaluateCpu<int32_t>(inputs);
    case EValueType::Int64:
        return DoEvaluateCpu<int64_t>(inputs);
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
std::vector<char> TSiLUNode::DoEvaluateCpu(const std::unordered_map<std::string, const void*>& inputs) const
{
    auto inputResult = Input_->EvaluateCpu(inputs);
    const auto* input = reinterpret_cast<const T*>(inputResult.data());

    std::vector<char> resultData(inputResult.size());
    auto* result = reinterpret_cast<T*>(resultData.data());

    for (int index = 0; index < Input_->GetElementCount(); ++index) {
        result[index] = input[index] / (1 + exp(-input[index]));
    }

    return resultData;
}

} // namespace NHamKaas
