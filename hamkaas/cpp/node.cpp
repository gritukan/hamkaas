#include "node.h"

#include <cuda_runtime.h>

#include "error.h"
#include "helpers.h"
#include "kernels.h"

#include <stdexcept>
#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>
#include <unordered_set>

namespace NHamKaas {

TNodeBase::TNodeBase(TTensorMeta meta, std::vector<TNodeBasePtr> inputs)
    : Meta_(std::move(meta))
    , Inputs_(std::move(inputs))
{ }

const TTensorMeta& TNodeBase::GetMeta() const
{
    return Meta_;
}

EValueType TNodeBase::GetValueType() const
{
    return Meta_.ValueType;
}

int64_t TNodeBase::GetDimensions() const
{
    return Meta_.GetDimensions();
}

const std::vector<int64_t>& TNodeBase::GetShape() const
{
    return Meta_.Shape;
}

int64_t TNodeBase::GetElementCount() const
{
    return Meta_.GetElementCount();
}

int64_t TNodeBase::GetElementSize() const
{
    return Meta_.GetElementSize();
}

int64_t TNodeBase::GetCapacity() const
{
    return Meta_.GetCapacity();
}

const std::vector<std::shared_ptr<TNodeBase>>& TNodeBase::GetInputs() const
{
    return Inputs_;
}

void TNodeBase::ReplaceInput(TNodeBasePtr oldInput, TNodeBasePtr newInput)
{
    for (auto& input : Inputs_) {
        if (input == oldInput) {
            input = newInput;
        }
    }
}

int64_t TNodeBase::GetConstantMemorySize() const
{
    return 0;
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

void TNodeBase::SetConstantMemory(char* /*constantMemory*/)
{
    // Do nothing.
}

void TNodeBase::SetBuffer(char* /*buffer*/)
{
    // Do nothing.
}

void TNodeBase::SetOutput(char* output)
{
    Output_ = output;
}

char* TNodeBase::GetOutput() const
{
    return Output_;
}

void TNodeBase::Initialize(IDevice* /*device*/)
{
    // Do nothing.
}

TInputNode::TInputNode(std::string name, TTensorMeta meta)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TInputNode::GetName() const
{
    return Name_;
}

void TInputNode::EvaluateCpu()
{
    // Do nothing; buffer is already set by the model evaluator.
}

void TInputNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // Do nothing; buffer is already set by the model evaluator.
}

TBufferNode::TBufferNode(TTensorMeta meta)
    : TNodeBase(std::move(meta))
{ }

TNodeBase* TBufferNode::GetOutputOwner() const
{
    // We should never clear the output of a buffer node
    // in order to keep the data between evaluations.
    return nullptr;
}

void TBufferNode::EvaluateCpu()
{
    // Do nothing.
}

void TBufferNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // Do nothing.
}

TConstantNode::TConstantNode(TTensorMeta meta, std::string name)
    : TNodeBase(std::move(meta))
    , Name_(std::move(name))
{ }

const std::string& TConstantNode::GetName() const
{
    return Name_;
}

TNodeBase* TConstantNode::GetOutputOwner() const
{
    // We should never clear the output of a constant node
    // in order to keep the data between evaluations.
    return nullptr;
}

void TConstantNode::EvaluateCpu()
{
    // Do nothing; buffer is already set by the model evaluator.
}

void TConstantNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // Do nothing; buffer is already set by the model evaluator.
}

template <EPointwiseOperation Operation>
TPointwiseNode<Operation>::TPointwiseNode(TNodeBasePtr lhs)
    : TNodeBase(lhs->GetMeta(), {lhs})
{
    assert(!IsBinary(Operation));
}

template <EPointwiseOperation Operation>
TPointwiseNode<Operation>::TPointwiseNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{
    assert(IsBinary(Operation));

    NeedBroadcasting_ = lhs->GetShape() != rhs->GetShape();
}

template <EPointwiseOperation Operation>
int64_t TPointwiseNode<Operation>::GetConstantMemorySize() const
{
    return 2 * GetDimensions() * sizeof(int64_t);
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::SetConstantMemory(char* buffer)
{
    LhsShape_ = reinterpret_cast<int64_t*>(buffer);
    RhsShape_ = LhsShape_ + GetDimensions();
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::Initialize(IDevice* device)
{
    device->CopyToDevice(LhsShape_, Inputs_[0]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    if constexpr (IsBinary(Operation)) {
        device->CopyToDevice(RhsShape_, Inputs_[1]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    }
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::EvaluateCpu()
{
    auto* lhsPtr = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    const float* rhsPtr = nullptr;
    if constexpr (IsBinary(Operation)) {
        rhsPtr = reinterpret_cast<const float*>(Inputs_[1]->GetOutput());
    }

    auto* outputPtr = reinterpret_cast<float*>(GetOutput());

    std::vector<int64_t> rhsIndices(GetDimensions());

    constexpr int64_t lhsStep = (Operation == EPointwiseOperation::ComplexHadamardProduct) ? 2 : 1;

    for (int64_t lhsIndex = 0; lhsIndex < GetElementCount(); lhsIndex += lhsStep) {
        int64_t rhsIndex = lhsIndex;
        if (NeedBroadcasting_) {
            int64_t indexCopy = lhsIndex;
            for (int64_t index = GetDimensions() - 1; index >= 0; --index) {
                rhsIndices[index] = indexCopy % LhsShape_[index];
                indexCopy /= LhsShape_[index];
                if (rhsIndices[index] >= RhsShape_[index]) {
                    assert(RhsShape_[index] == 1);
                    rhsIndices[index] = 0;
                }
            }

            rhsIndex = 0;
            for (int64_t index = 0; index < GetDimensions(); ++index) {
                rhsIndex = rhsIndex * RhsShape_[index] + rhsIndices[index];
            }
        }

        if constexpr (IsBinary(Operation)) {
            DoEvaluateCpu(lhsPtr + lhsIndex, rhsPtr + rhsIndex, outputPtr + lhsIndex);
        } else {
            DoEvaluateCpu(lhsPtr + lhsIndex, outputPtr + lhsIndex);
        }
    }
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::EvaluateGpu(const TEvaluationContext& context)
{
    float* lhs = reinterpret_cast<float*>(Inputs_[0]->GetOutput());
    float* rhs = nullptr;
    if constexpr (IsBinary(Operation)) {
        rhs = reinterpret_cast<float*>(Inputs_[1]->GetOutput());
    }
    float* output = reinterpret_cast<float*>(GetOutput());

    Pointwise<Operation>(
        context.Stream,
        lhs,
        rhs,
        output,
        LhsShape_,
        RhsShape_,
        GetDimensions(),
        NeedBroadcasting_,
        GetElementCount());
}

template <EPointwiseOperation Operation>
TTensorMeta TPointwiseNode<Operation>::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    assert(IsBinary(Operation));

    const auto& lhsShape = lhs.Shape;
    const auto& rhsShape = rhs.Shape;

    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.ValueType != EValueType::Float32) {
        THROW("Unsupported value type", VAR(lhs.ValueType));
    }

    if (lhsShape.size() != rhsShape.size()) {
        THROW("Different number of dimensions", VAR(lhsShape.size()), VAR(rhsShape.size()));
    }

    for (int64_t index = 0; index < lhs.GetDimensions(); ++index) {
        if (lhsShape[index] != rhsShape[index] && !(IsBroadcastingSupported(Operation) && rhsShape[index] == 1)) {
            THROW("Incompatible shapes for pointwise operation", VAR(index), VAR(lhsShape[index]), VAR(rhsShape[index]));
        }
    }

    return lhs;
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::DoEvaluateCpu(const float* lhsPtr, float* outputPtr) const
{
    if constexpr (Operation == EPointwiseOperation::ReLU) {
        *outputPtr = std::max<float>(0.0, *lhsPtr);
    } else if constexpr (Operation == EPointwiseOperation::SiLU) {
        *outputPtr = *lhsPtr / (1 + exp(-*lhsPtr));
    } else {
        assert(false);
    }
}

template <EPointwiseOperation Operation>
void TPointwiseNode<Operation>::DoEvaluateCpu(const float* lhsPtr, const float* rhsPtr, float* outputPtr) const
{
    if constexpr (Operation == EPointwiseOperation::Add) {
        *outputPtr = *lhsPtr + *rhsPtr;
    } else if constexpr (Operation == EPointwiseOperation::HadamardProduct) {
        *outputPtr = *lhsPtr * *rhsPtr;
    } else if constexpr (Operation == EPointwiseOperation::ComplexHadamardProduct) {
        *outputPtr = *lhsPtr * *rhsPtr - *(lhsPtr + 1) * *(rhsPtr + 1);
        *(outputPtr + 1) = *lhsPtr * *(rhsPtr + 1) + *(lhsPtr + 1) * *rhsPtr;
    } else {
        assert(false);
    }
}

template class TPointwiseNode<EPointwiseOperation::Add>;
template class TPointwiseNode<EPointwiseOperation::HadamardProduct>;
template class TPointwiseNode<EPointwiseOperation::ComplexHadamardProduct>;
template class TPointwiseNode<EPointwiseOperation::ReLU>;
template class TPointwiseNode<EPointwiseOperation::SiLU>;

TMatMulNode::TMatMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{ }

int64_t TMatMulNode::GetConstantMemorySize() const
{
    // (lab3/04): you will probably need to change it if cuBLAS is used.
    return 0;
}

void TMatMulNode::SetConstantMemory(char* constantMemory)
{
    // (lab3/04): you will probably need to change it if cuBLAS is used.
}

TTensorMeta TMatMulNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.ValueType != EValueType::Float32) {
        THROW("Unsupported value type", VAR(lhs.ValueType));
    }

    if (lhs.GetDimensions() == 1) {
        if (rhs.GetDimensions() != 2) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.GetDimensions()), VAR(rhs.GetDimensions()));
        }
        if (lhs.Shape[0] != rhs.Shape[0]) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.Shape[0]), VAR(rhs.Shape[0]));
        }

        return TTensorMeta{
            .ValueType = lhs.ValueType,
            .Shape = {rhs.Shape[1]},
        };
    } else if (lhs.GetDimensions() == 2) {
        if (rhs.GetDimensions() != 2) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.GetDimensions()), VAR(rhs.GetDimensions()));
        }
        if (lhs.Shape[1] != rhs.Shape[0]) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.Shape[1]), VAR(rhs.Shape[0]));
        }

        return TTensorMeta{
            .ValueType = lhs.ValueType,
            .Shape = {lhs.Shape[0], rhs.Shape[1]},
        };
    } else if (lhs.GetDimensions() == 3) {
        if (rhs.GetDimensions() != 3) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.GetDimensions()), VAR(rhs.GetDimensions()));
        }
        if (lhs.Shape[2] != rhs.Shape[1]) {
            THROW("Incompatible shapes for matrix multiplication", VAR(lhs.Shape[2]), VAR(rhs.Shape[1]));
        }

        return TTensorMeta{
            .ValueType = lhs.ValueType,
            .Shape = {lhs.Shape[0], lhs.Shape[1], rhs.Shape[2]},
        };
    } else {
        THROW("Matrix multiplication is supported only for 1D, 2D, and 3D tensors",
            VAR(lhs.GetDimensions()),
            VAR(rhs.GetDimensions()));
    }
}

void TMatMulNode::Initialize(IDevice* device)
{
    // (lab4/03): Your code here: implement the matrix multiplication.
}

void TMatMulNode::EvaluateCpu()
{
    auto* lhs = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    auto* rhs = reinterpret_cast<const float*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<float*>(GetOutput());

    auto [b, n, m, k] = GetParameters();

    for (int64_t matrixIndex = 0; matrixIndex < b; ++matrixIndex) {
        for (int64_t x = 0; x < n; x++) {
            for (int64_t y = 0; y < m; y++) {
                float sum = 0.0;
                for (int64_t index = 0; index < k; ++index) {
                    sum += lhs[(matrixIndex * n + x) * k + index] * rhs[(matrixIndex * k + index) * m + y];
                }
                output[(matrixIndex * n + x) * m + y] = sum;
            }
        }
    }
}

void TMatMulNode::EvaluateGpu(const TEvaluationContext& context)
{
    // (lab4/03): Your code here: implement the matrix multiplication.
}

TMatMulNode::TParameters TMatMulNode::GetParameters() const
{
    if (Inputs_[0]->GetDimensions() == 1) {
        return TParameters{
            .B = 1,
            .N = 1,
            .M = Inputs_[1]->GetShape()[1],
            .K = Inputs_[0]->GetShape()[0],
        };
    } else if (Inputs_[0]->GetDimensions() == 2) {
        return TParameters{
            .B = 1,
            .N = Inputs_[0]->GetShape()[0],
            .M = Inputs_[1]->GetShape()[1],
            .K = Inputs_[0]->GetShape()[1],
        };
    } else {
        return TParameters{
            .B = Inputs_[0]->GetShape()[0],
            .N = Inputs_[0]->GetShape()[1],
            .M = Inputs_[1]->GetShape()[2],
            .K = Inputs_[0]->GetShape()[2],
        };
    }
}

TSliceNode::TSliceNode(TNodeBasePtr input, int64_t begin, int64_t end)
    : TNodeBase(CalculateMeta(input->GetMeta(), begin, end), {input})
    , Begin_(begin)
    , End_(end)
{
    Stride_ = 1;
    for (int64_t index = 1; index < GetDimensions(); ++index) {
        Stride_ *= GetShape()[index];
    }
}

int64_t TSliceNode::GetBegin() const
{
    return Begin_;
}

int64_t TSliceNode::GetEnd() const
{
    return End_;
}

int64_t TSliceNode::GetOutputSize() const
{
    // We do not need to allocate memory for the slice
    // as it is just a view of the input tensor.
    return 0;
}

TNodeBase* TSliceNode::GetOutputOwner() const
{
    return Inputs_[0]->GetOutputOwner();
}

void TSliceNode::Initialize(IDevice* /*device*/)
{
    Output_ = Inputs_[0]->GetOutput() + Begin_ * Stride_ * GetElementSize();
}

void TSliceNode::EvaluateCpu()
{
    // Do nothing.
}

void TSliceNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // Do nothing.
}

TTensorMeta TSliceNode::CalculateMeta(const TTensorMeta& input, int64_t begin, int64_t end)
{
    if (begin < 0 || begin >= input.Shape[0]) {
        THROW("Invalid begin index", VAR(begin), VAR(input.Shape[0]));
    }
    if (end < 0 || end > input.Shape[0]) {
        THROW("Invalid end index", VAR(end), VAR(input.Shape[0]));
    }
    if (begin >= end) {
        THROW("Invalid slice range", VAR(begin), VAR(end));
    }

    std::vector<int64_t> shape(input.Shape);
    shape[0] = end - begin;

    return TTensorMeta{
        .ValueType = input.ValueType,
        .Shape = std::move(shape),
    };
}

TRmsNormNode::TRmsNormNode(TNodeBasePtr input, TNodeBasePtr weights)
    : TNodeBase(input->GetMeta(), {input, weights})
{
    if (input->GetValueType() != weights->GetValueType()) {
        THROW("Different value types", VAR(input->GetValueType()), VAR(weights->GetValueType()));
    }
    if (input->GetValueType() != EValueType::Float32) {
        THROW("Unsupported value type", VAR(input->GetValueType()));
    }
    if (input->GetDimensions() != 1) {
        THROW("RMS normalization is supported for vectors only", VAR(input->GetDimensions()));
    }
    if (input->GetShape() != weights->GetShape()) {
        THROW("Different shapes", VAR(input->GetShape()), VAR(weights->GetShape()));
    }
}

void TRmsNormNode::EvaluateCpu()
{
    auto* input = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    auto* weights = reinterpret_cast<const float*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<float*>(GetOutput());

    float sum = 0.0;
    for (int64_t index = 0; index < Inputs_[0]->GetElementCount(); ++index) {
        sum += input[index] * input[index];
    }
    sum /= Inputs_[0]->GetElementCount();
    sum += 1e-5;
    sum = 1.0 / sqrt(sum);

    for (int64_t index = 0; index < Inputs_[0]->GetElementCount(); ++index) {
        output[index] = weights[index] * (input[index] * sum);
    }
}

void TRmsNormNode::EvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    auto* weights = reinterpret_cast<const float*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<float*>(GetOutput());

    RMSNorm(
        context.Stream,
        input,
        weights,
        output,
        Inputs_[0]->GetElementCount(),
        /*epsilon*/ 1e-5);
}

TReshapeNode::TReshapeNode(TNodeBasePtr input, std::vector<int64_t> shape)
    : TNodeBase(CalculateMeta(input->GetMeta(), shape), {input})
    , Shape_(std::move(shape))
{ }

int64_t TReshapeNode::GetOutputSize() const
{
    // Reshape node returns a view of the input tensor.
    return 0;
}

TNodeBase* TReshapeNode::GetOutputOwner() const
{
    return Inputs_[0]->GetOutputOwner();
}

void TReshapeNode::Initialize(IDevice* /*device*/)
{
    Output_ = Inputs_[0]->GetOutput();
}

void TReshapeNode::EvaluateCpu()
{
    // Do nothing.
}

void TReshapeNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // Do nothing.
}

TTensorMeta TReshapeNode::CalculateMeta(const TTensorMeta& input, const std::vector<int64_t>& shape)
{
    int64_t elementCount = 1;
    for (int64_t index = 0; index < shape.size(); ++index) {
        elementCount *= shape[index];
    }

    if (elementCount != input.GetElementCount()) {
        THROW("Incompatible number of elements", VAR(elementCount), VAR(input.GetElementCount()));
    }

    return TTensorMeta{
        .ValueType = input.ValueType,
        .Shape = shape,
    };
}

TPermuteNode::TPermuteNode(TNodeBasePtr input, std::vector<int64_t> permutation)
    : TNodeBase(CalculateMeta(input->GetMeta(), permutation), {input})
    , Permutation_(std::move(permutation))
{ }

void TPermuteNode::EvaluateCpu()
{
    auto dimensions = GetDimensions();
    auto elementCount = GetElementCount();
    std::vector<int64_t> inputIndices(GetDimensions());

    for (int64_t inputIndex = 0; inputIndex < elementCount; ++inputIndex) {
        int64_t indexCopy = inputIndex;
        const auto& inputShape = Inputs_[0]->GetShape();
        for (int64_t index = dimensions - 1; index >= 0; --index) {
            inputIndices[index] = indexCopy % inputShape[index];
            indexCopy /= inputShape[index];
        }

        int64_t outputIndex = 0;
        for (int64_t index = 0; index < dimensions; ++index) {
            outputIndex = outputIndex * GetShape()[index] + inputIndices[Permutation_[index]];
        }

        memcpy(
            GetOutput() + outputIndex * GetElementSize(),
            Inputs_[0]->GetOutput() + inputIndex * GetElementSize(),
            GetElementSize());
    }
}

int64_t TPermuteNode::GetConstantMemorySize() const
{
    // (lab3/04): you will probably need to change it.
    return 0;
}

void TPermuteNode::SetConstantMemory(char* buffer)
{
    // (lab3/04): you will probably need to change it.
}

void TPermuteNode::Initialize(IDevice* device)
{
    // (lab3/04): you will probably need to change it.
}

void TPermuteNode::EvaluateGpu(const TEvaluationContext& context)
{
    // (lab3/04): your code here: implement the tensor permutation.
}

TTensorMeta TPermuteNode::CalculateMeta(const TTensorMeta& input, const std::vector<int64_t>& permutation)
{
    if (input.GetDimensions() != permutation.size()) {
        THROW("Invalid permutation size", VAR(input.GetDimensions()), VAR(permutation.size()));
    }

    if (input.ValueType != EValueType::Float32) {
        THROW("Unsupported value type", VAR(input.ValueType));
    }

    int64_t outputSize = 1;
    std::unordered_set<int64_t> uniqueIndices;
    for (int64_t index = 0; index < input.GetDimensions(); ++index) {
        if (!uniqueIndices.insert(permutation[index]).second) {
            THROW("Duplicate permutation index", VAR(permutation[index]));
        }
        if (permutation[index] < 0 || permutation[index] >= input.GetDimensions()) {
            THROW("Invalid permutation index", VAR(permutation[index]), VAR(input.GetDimensions()));
        }
        outputSize *= input.Shape[permutation[index]];
    }

    if (outputSize != input.GetElementCount()) {
        THROW("Invalid permutation", VAR(outputSize), VAR(input.GetElementCount()));
    }

    std::vector<int64_t> shape(input.GetDimensions());
    for (int64_t index = 0; index < input.GetDimensions(); ++index) {
        shape[index] = input.Shape[permutation[index]];
    }

    return TTensorMeta{
        .ValueType = input.ValueType,
        .Shape = std::move(shape),
    };
}

TReplaceSliceNode::TReplaceSliceNode(TNodeBasePtr input, TNodeBasePtr replacement, TNodeBasePtr begin, TNodeBasePtr end)
    : TNodeBase(input->GetMeta(), {input, replacement, begin, end})
{
    if (input->GetValueType() != replacement->GetValueType()) {
        THROW("Different value types", VAR(input->GetValueType()), VAR(replacement->GetValueType()));
    }
    if (input->GetValueType() != EValueType::Float32) {
        THROW("Unsupported value type", VAR(input->GetValueType()));
    }
    if (input->GetDimensions() != replacement->GetDimensions()) {
        THROW("Different number of dimensions", VAR(input->GetDimensions()), VAR(replacement->GetDimensions()));
    }
    if (begin->GetDimensions() != 1 || end->GetDimensions() != 1) {
        THROW("Begin and end should be 1D tensors", VAR(begin->GetDimensions()), VAR(end->GetDimensions()));
    }
    if (begin->GetShape()[0] != 1 || end->GetShape()[0] != 1) {
        THROW("Begin and end should be 1D tensors", VAR(begin->GetShape()[0]), VAR(end->GetShape()[0]));
    }
    if (begin->GetValueType() != EValueType::Int64 || end->GetValueType() != EValueType::Int64) {
        THROW("Begin and end should be int64 tensors", VAR(begin->GetValueType()), VAR(end->GetValueType()));
    }
}

int64_t TReplaceSliceNode::GetOutputSize() const
{
    // We do not need to allocate memory for the slice
    // as it is passes modified input further.
    return 0;
}

TNodeBase* TReplaceSliceNode::GetOutputOwner() const
{
    return Inputs_[0]->GetOutputOwner();
}

void TReplaceSliceNode::Initialize(IDevice* /*device*/)
{
    Output_ = Inputs_[0]->GetOutput();
}

void TReplaceSliceNode::EvaluateCpu()
{
    const auto& input = Inputs_[0];
    const auto& replacement = Inputs_[1];
    const auto& begin = Inputs_[2];
    const auto& end = Inputs_[3];

    auto* output = input->GetOutput();
    auto* replacementPtr = replacement->GetOutput();
    auto beginPtr = *reinterpret_cast<const int64_t*>(begin->GetOutput());
    auto endPtr = *reinterpret_cast<const int64_t*>(end->GetOutput());

    assert(endPtr - beginPtr == replacement->GetElementCount());
    assert(beginPtr >= 0);
    assert(endPtr <= input->GetElementCount());

    memcpy(
        output + beginPtr * GetElementSize(),
        replacementPtr,
        (endPtr - beginPtr) * GetElementSize());
}

void TReplaceSliceNode::EvaluateGpu(const TEvaluationContext& context)
{
    // (lab3/04): your code here: implement the slice replacement.
}

TSlicedSoftmaxNode::TSlicedSoftmaxNode(TNodeBasePtr input, TNodeBasePtr prefixSize)
    : TNodeBase(input->GetMeta(), {input, prefixSize})
{
    if (prefixSize->GetDimensions() != 1) {
        THROW("Prefix size should be a 1D tensor", VAR(prefixSize->GetDimensions()));
    }
    if (prefixSize->GetShape()[0] != 1) {
        THROW("Prefix size should be a 1D tensor", VAR(prefixSize->GetShape()[0]));
    }
    if (prefixSize->GetValueType() != EValueType::Int64) {
        THROW("Prefix size should be an int64 tensor", VAR(prefixSize->GetValueType()));
    }
}

void TSlicedSoftmaxNode::EvaluateCpu()
{
    auto* input = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    auto prefixSize = *reinterpret_cast<const int64_t*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<float*>(GetOutput());

    if (prefixSize == 0) {
        memcpy(output, input, GetElementCount() * GetElementSize());
        return;
    }

    if (prefixSize > GetShape().back()) {
        THROW("Invalid prefix size", VAR(prefixSize), VAR(GetShape().back()));
    }

    int64_t vectorSize = GetShape().back();
    for (int64_t startIndex = 0; startIndex < GetElementCount(); startIndex += vectorSize) {
        float max = input[startIndex];
        for (int64_t index = 1; index < prefixSize; ++index) {
            max = std::max(max, input[startIndex + index]);
        }

        float expSum = 0.0;
        for (int64_t index = 0; index < prefixSize; ++index) {
            expSum += exp(input[startIndex + index] - max);
        }

        for (int64_t index = 0; index < prefixSize; ++index) {
            output[startIndex + index] = exp(input[startIndex + index] - max) / expSum;
        }

        for (int64_t index = prefixSize; index < vectorSize; ++index) {
            output[startIndex + index] = input[startIndex + index];
        }
    }
}

void TSlicedSoftmaxNode::EvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const float*>(Inputs_[0]->GetOutput());
    auto* prefixSizePtr = reinterpret_cast<int64_t*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<float*>(GetOutput());

    SlicedSoftmax(
        context.Stream,
        input,
        output,
        prefixSizePtr,
        GetElementCount(),
        GetShape().back());
}

} // namespace NHamKaas
