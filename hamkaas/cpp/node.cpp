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

void TNodeBase::ReplaceInput(TNodeBase* oldInput, std::shared_ptr<TNodeBase> newInput)
{
    for (auto& input : Inputs_) {
        if (input.get() == oldInput) {
            input = newInput;
        }
    }
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

TSumNode::TSumNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{ }

int64_t TSumNode::GetBufferSize() const
{
    return 2 * GetDimensions() * sizeof(int64_t);
}

void TSumNode::SetBuffer(char* buffer)
{
    LhsShape_ = reinterpret_cast<int64_t*>(buffer);
    RhsShape_ = LhsShape_ + GetDimensions();
}

void TSumNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TSumNode::Initialize(IDevice* device)
{
    device->CopyToDevice(LhsShape_, Inputs_[0]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    device->CopyToDevice(RhsShape_, Inputs_[1]->GetShape().data(), GetDimensions() * sizeof(int64_t));
}

void TSumNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TSumNode::DoEvaluateCpu()
{
    const auto& lhs = Inputs_[0];
    const auto& rhs = Inputs_[1];

    auto* lhsPtr = reinterpret_cast<const T*>(lhs->GetOutput());
    auto* rhsPtr = reinterpret_cast<const T*>(rhs->GetOutput());
    auto* outputPtr = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < lhs->GetElementCount(); ++index) {
        std::vector<int64_t> rhsIndices(rhs->GetDimensions());

        int64_t indexCopy = index;
        for (int64_t index = rhs->GetDimensions() - 1; index >= 0; --index) {
            rhsIndices[index] = indexCopy % lhs->GetShape()[index];
            indexCopy /= lhs->GetShape()[index];
            if (rhsIndices[index] >= rhs->GetShape()[index]) {
                assert(rhs->GetShape()[index] == 1);
                rhsIndices[index] = 0;
            }
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < rhs->GetDimensions(); ++index) {
            rhsIndex = rhsIndex * rhs->GetShape()[index] + rhsIndices[index];
        }

        outputPtr[index] = lhsPtr[index] + rhsPtr[rhsIndex];
    }
}

template <class T>
void TSumNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    SumTensorsBroadcast(
        context.Stream,
        reinterpret_cast<const T*>(Inputs_[0]->GetOutput()),
        reinterpret_cast<const T*>(Inputs_[1]->GetOutput()),
        reinterpret_cast<T*>(GetOutput()),
        LhsShape_,
        RhsShape_,
        GetDimensions(),
        GetElementCount());
}

TTensorMeta TSumNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.Shape.size() != rhs.Shape.size()) {
        THROW("Different number of dimensions", VAR(lhs.Shape.size()), VAR(rhs.Shape.size()));
    }

    for (int64_t index = 0; index < lhs.GetDimensions(); ++index) {
        if (lhs.Shape[index] != rhs.Shape[index] && rhs.Shape[index] != 1) {
            THROW("Incompatible shapes for sum", VAR(index), VAR(lhs.Shape[index]), VAR(rhs.Shape[index]));
        }
    }

    return lhs;
}

TMulNode::TMulNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{ }

int64_t TMulNode::GetBufferSize() const
{
    auto b = GetParameters().B;

    return 3 * Align(b) * sizeof(void*) + Align(GetOutputSize());
}

void TMulNode::SetBuffer(char* buffer)
{
    auto b = GetParameters().B;

    LhsMatrices_ = reinterpret_cast<void**>(buffer);
    RhsMatrices_ = reinterpret_cast<void**>(buffer + Align(b) * sizeof(void*));
    OutputMatrices_ = reinterpret_cast<void**>(buffer + 2 * Align(b) * sizeof(void*));

    TransposedProductBuffer_ = buffer + 3 * Align(b) * sizeof(void*);
}

TTensorMeta TMulNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
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

void TMulNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TMulNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TMulNode::Initialize(IDevice* device)
{
    auto [b, n, m, k] = GetParameters();

    std::vector<void*> lhsMatrices(b);
    std::vector<void*> rhsMatrices(b);
    std::vector<void*> outputMatrices(b);

    for (int index = 0; index < b; ++index) {
        lhsMatrices[index] = Inputs_[0]->GetOutput() + index * n * k * GetElementSize();
        rhsMatrices[index] = Inputs_[1]->GetOutput() + index * k * m * GetElementSize();
        outputMatrices[index] = TransposedProductBuffer_ + index * n * m * GetElementSize();
    }

    device->CopyToDevice(LhsMatrices_, lhsMatrices.data(), b * sizeof(char*));
    device->CopyToDevice(RhsMatrices_, rhsMatrices.data(), b * sizeof(char*));
    device->CopyToDevice(OutputMatrices_, outputMatrices.data(), b * sizeof(char*));
}

template <class T>
void TMulNode::DoEvaluateCpu()
{
    auto* lhs = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    auto [b, n, m, k] = GetParameters();

    for (int64_t matrixIndex = 0; matrixIndex < b; ++matrixIndex) {
        for (int64_t x = 0; x < n; x++) {
            for (int64_t y = 0; y < m; y++) {
                T sum = 0.0;
                for (int64_t index = 0; index < k; ++index) {
                    sum += lhs[(matrixIndex * n + x) * k + index] * rhs[(matrixIndex * k + index) * m + y];
                }
                output[(matrixIndex * n + x) * m + y] = sum;
            }
        }
    }
}

template <class T>
void TMulNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto [b, n, m, k] = GetParameters();

    T One = 1;
    T Zero = 0;

    cudaDataType_t type;
    switch (GetValueType()) {
    case EValueType::Float32:
        type = CUDA_R_32F;
        break;
    case EValueType::Float64:
        type = CUDA_R_64F;
        break;
    }

    CUBLAS_CHECK_ERROR(cublasSetStream(context.Bootstrap->GetCublasHandle(), context.Stream));

    CUBLAS_CHECK_ERROR(cublasGemmBatchedEx(
        context.Bootstrap->GetCublasHandle(),
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        n,
        m,
        k,
        &One,
        LhsMatrices_,
        type,
        k,
        RhsMatrices_,
        type,
        m,
        &Zero,
        OutputMatrices_,
        type,
        n,
        b,
        type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (type == CUDA_R_32F) {
        for (int index = 0; index < b; ++index) {
            auto* inputAddress = TransposedProductBuffer_ + index * n * m * sizeof(T);
            CUBLAS_CHECK_ERROR(cublasSgeam(
                context.Bootstrap->GetCublasHandle(),
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                m,
                n,
                reinterpret_cast<float*>(&One),
                reinterpret_cast<float*>(inputAddress),
                n,
                reinterpret_cast<float*>(&Zero),
                reinterpret_cast<float*>(inputAddress),
                n,
                reinterpret_cast<float*>(GetOutput() + index * n * m * sizeof(T)),
                m));
        }
    } else if (type == CUDA_R_64F) {
        for (int index = 0; index < b; ++index) {
            auto* inputAddress = TransposedProductBuffer_ + index * n * m * sizeof(T);
            CUBLAS_CHECK_ERROR(cublasDgeam(
                context.Bootstrap->GetCublasHandle(),
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                m,
                n,
                reinterpret_cast<double*>(&One),
                reinterpret_cast<double*>(inputAddress),
                n,
                reinterpret_cast<double*>(&Zero),
                reinterpret_cast<double*>(inputAddress),
                n,
                reinterpret_cast<double*>(GetOutput() + index * n * m * sizeof(T)),
                m));
        }
    } else {
        THROW("Unsupported value type", VAR(GetValueType()));
    }
}

TMulNode::TParameters TMulNode::GetParameters() const
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

TReLUNode::TReLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta(), {input})
{ }

void TReLUNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TReLUNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    }
}

template <class T>
void TReLUNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Inputs_[0]->GetElementCount(); ++index) {
        output[index] = std::max<T>(0.0, input[index]);
    }
}

template <class T>
void TReLUNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    ReLU(context.Stream, input, output, Inputs_[0]->GetElementCount());
}

TSiLUNode::TSiLUNode(TNodeBasePtr input)
    : TNodeBase(input->GetMeta(), {input})
{ }

void TSiLUNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TSiLUNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    }
}

template <class T>
void TSiLUNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Inputs_[0]->GetElementCount(); ++index) {
        output[index] = input[index] / (1 + exp(-input[index]));
    }
}

template <class T>
void TSiLUNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    SiLU(context.Stream, input, output, Inputs_[0]->GetElementCount());
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
    if (input->GetDimensions() != 1) {
        THROW("RMS normalization is supported for vectors only", VAR(input->GetDimensions()));
    }
    if (input->GetShape() != weights->GetShape()) {
        THROW("Different shapes", VAR(input->GetShape()), VAR(weights->GetShape()));
    }
}

void TRmsNormNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TRmsNormNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TRmsNormNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* weights = reinterpret_cast<const T*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    T sum = 0.0;
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

template <class T>
void TRmsNormNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* weights = reinterpret_cast<const T*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    RMSNorm(
        context.Stream,
        input,
        weights,
        output,
        Inputs_[0]->GetElementCount(),
        /*epsilon*/ T(1e-5));
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

TComplexHadamardProductNode::TComplexHadamardProductNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{ }

int64_t TComplexHadamardProductNode::GetBufferSize() const
{
    return 2 * GetDimensions() * sizeof(int64_t);
}

void TComplexHadamardProductNode::SetBuffer(char* buffer)
{
    LhsShape_ = reinterpret_cast<int64_t*>(buffer);
    RhsShape_ = LhsShape_ + GetDimensions();
}

void TComplexHadamardProductNode::Initialize(IDevice* device)
{
    device->CopyToDevice(LhsShape_, Inputs_[0]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    device->CopyToDevice(RhsShape_, Inputs_[1]->GetShape().data(), GetDimensions() * sizeof(int64_t));
}

void TComplexHadamardProductNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TComplexHadamardProductNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    }
}

template <typename T>
void TComplexHadamardProductNode::DoEvaluateCpu()
{
    const auto& lhs = Inputs_[0];
    const auto& rhs = Inputs_[1];

    auto* lhsPtr = reinterpret_cast<const T*>(lhs->GetOutput());
    auto* rhsPtr = reinterpret_cast<const T*>(rhs->GetOutput());
    auto* outputPtr = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < lhs->GetElementCount() / 2; ++index) {
        std::vector<int64_t> rhsIndices(rhs->GetDimensions() - 1);

        int64_t indexCopy = index;
        for (int64_t index = rhs->GetDimensions() - 2; index >= 0; --index) {
            rhsIndices[index] = indexCopy % lhs->GetShape()[index];
            indexCopy /= lhs->GetShape()[index];
            if (rhsIndices[index] >= rhs->GetShape()[index]) {
                rhsIndices[index] = 0;
            }
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < rhs->GetDimensions() - 1; ++index) {
            rhsIndex = rhsIndex * rhs->GetShape()[index] + rhsIndices[index];
        }

        outputPtr[2 * index] = lhsPtr[2 * index] * rhsPtr[2 * rhsIndex] - lhsPtr[2 * index + 1] * rhsPtr[2 * rhsIndex + 1];
        outputPtr[2 * index + 1] = lhsPtr[2 * index] * rhsPtr[2 * rhsIndex + 1] + lhsPtr[2 * index + 1] * rhsPtr[2 * rhsIndex];
    }
}

template <typename T>
void TComplexHadamardProductNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    ComplexHadamardProductBroadcast(
        context.Stream,
        reinterpret_cast<const T*>(Inputs_[0]->GetOutput()),
        reinterpret_cast<const T*>(Inputs_[1]->GetOutput()),
        reinterpret_cast<T*>(GetOutput()),
        LhsShape_,
        RhsShape_,
        GetDimensions(),
        GetElementCount());
}

TTensorMeta TComplexHadamardProductNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.Shape.size() != rhs.Shape.size()) {
        THROW("Different number of dimensions", VAR(lhs.Shape.size()), VAR(rhs.Shape.size()));
    }

    if (lhs.Shape.back() != 2 || rhs.Shape.back() != 2) {
        THROW("Complex Hadamard product is supported for complex tensors only", VAR(lhs.Shape.back()), VAR(rhs.Shape.back()));
    }

    for (int64_t index = 0; index + 1 < lhs.GetDimensions(); ++index) {
        if (lhs.Shape[index] != rhs.Shape[index] && rhs.Shape[index] != 1) {
            THROW("Incompatible shapes for Hadamard product", VAR(index), VAR(lhs.Shape[index]), VAR(rhs.Shape[index]));
        }
    }

    return lhs;
}

THadamardProductNode::THadamardProductNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(CalculateMeta(lhs->GetMeta(), rhs->GetMeta()), {lhs, rhs})
{ }

int64_t THadamardProductNode::GetBufferSize() const
{
    return 2 * GetDimensions() * sizeof(int64_t);
}

void THadamardProductNode::SetBuffer(char* buffer)
{
    LhsShape_ = reinterpret_cast<int64_t*>(buffer);
    RhsShape_ = LhsShape_ + GetDimensions();
}

void THadamardProductNode::Initialize(IDevice* device)
{
    device->CopyToDevice(LhsShape_, Inputs_[0]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    device->CopyToDevice(RhsShape_, Inputs_[1]->GetShape().data(), GetDimensions() * sizeof(int64_t));
}

void THadamardProductNode::EvaluateCpu()
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void THadamardProductNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <typename T>
void THadamardProductNode::DoEvaluateCpu()
{
    const auto& lhs = Inputs_[0];
    const auto& rhs = Inputs_[1];

    auto* lhsPtr = reinterpret_cast<const T*>(lhs->GetOutput());
    auto* rhsPtr = reinterpret_cast<const T*>(rhs->GetOutput());
    auto* outputPtr = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < lhs->GetElementCount(); ++index) {
        std::vector<int64_t> rhsIndices(rhs->GetDimensions());

        int64_t indexCopy = index;
        for (int64_t index = rhs->GetDimensions() - 1; index >= 0; --index) {
            rhsIndices[index] = indexCopy % lhs->GetShape()[index];
            indexCopy /= lhs->GetShape()[index];
            if (rhsIndices[index] >= rhs->GetShape()[index]) {
                assert(rhs->GetShape()[index] == 1);
                rhsIndices[index] = 0;
            }
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < rhs->GetDimensions(); ++index) {
            rhsIndex = rhsIndex * rhs->GetShape()[index] + rhsIndices[index];
        }

        outputPtr[index] = lhsPtr[index] * rhsPtr[rhsIndex];
    }
}

template <typename T>
void THadamardProductNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    HadamardProductBroadcast(
        context.Stream,
        reinterpret_cast<const T*>(Inputs_[0]->GetOutput()),
        reinterpret_cast<const T*>(Inputs_[1]->GetOutput()),
        reinterpret_cast<T*>(GetOutput()),
        LhsShape_,
        RhsShape_,
        GetDimensions(),
        GetElementCount());
}

TTensorMeta THadamardProductNode::CalculateMeta(const TTensorMeta& lhs, const TTensorMeta& rhs)
{
    if (lhs.ValueType != rhs.ValueType) {
        THROW("Different value types", VAR(lhs.ValueType), VAR(rhs.ValueType));
    }

    if (lhs.Shape.size() != rhs.Shape.size()) {
        THROW("Different number of dimensions", VAR(lhs.Shape.size()), VAR(rhs.Shape.size()));
    }

    for (int64_t index = 0; index < lhs.GetDimensions(); ++index) {
        if (lhs.Shape[index] != rhs.Shape[index] && rhs.Shape[index] != 1) {
            THROW("Incompatible shapes for Hadamard product", VAR(index), VAR(lhs.Shape[index]), VAR(rhs.Shape[index]));
        }
    }

    return lhs;
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

int64_t TPermuteNode::GetBufferSize() const
{
    return 3 * GetDimensions() * sizeof(int64_t);
}

void TPermuteNode::SetBuffer(char* buffer)
{
    InputShape_ = reinterpret_cast<int64_t*>(buffer);
    OutputShape_ = InputShape_ + GetDimensions();
    PermutationPtr_ = OutputShape_ + GetDimensions();
}

void TPermuteNode::Initialize(IDevice* device)
{
    device->CopyToDevice(InputShape_, Inputs_[0]->GetShape().data(), GetDimensions() * sizeof(int64_t));
    device->CopyToDevice(OutputShape_, GetShape().data(), GetDimensions() * sizeof(int64_t));
    device->CopyToDevice(PermutationPtr_, Permutation_.data(), GetDimensions() * sizeof(int64_t));
}

void TPermuteNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    }
}

template <class T>
void TPermuteNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    Permute(
        context.Stream,
        input,
        output,
        InputShape_,
        OutputShape_,
        PermutationPtr_,
        GetDimensions(),
        GetElementCount());
}

TTensorMeta TPermuteNode::CalculateMeta(const TTensorMeta& input, const std::vector<int64_t>& permutation)
{
    if (input.GetDimensions() != permutation.size()) {
        THROW("Invalid permutation size", VAR(input.GetDimensions()), VAR(permutation.size()));
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
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TReplaceSliceNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<T*>(Inputs_[0]->GetOutput());
    auto* replacement = reinterpret_cast<const T*>(Inputs_[1]->GetOutput());
    auto* begin = reinterpret_cast<const int64_t*>(Inputs_[2]->GetOutput());
    auto* end = reinterpret_cast<const int64_t*>(Inputs_[3]->GetOutput());

    ReplaceSlice(
        context.Stream,
        input,
        Inputs_[0]->GetElementCount(),
        replacement,
        Inputs_[1]->GetElementCount(),
        begin,
        end);
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
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateCpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateCpu<double>();
        return;
    default:
        THROW("CPU inference does not support this value type", VAR(GetValueType()));
    }
}

void TSlicedSoftmaxNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>(context);
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>(context);
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TSlicedSoftmaxNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto prefixSize = *reinterpret_cast<const int64_t*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    if (prefixSize == 0) {
        memcpy(output, input, GetElementCount() * GetElementSize());
        return;
    }

    if (prefixSize > GetShape().back()) {
        THROW("Invalid prefix size", VAR(prefixSize), VAR(GetShape().back()));
    }

    int64_t vectorSize = GetShape().back();
    for (int64_t startIndex = 0; startIndex < GetElementCount(); startIndex += vectorSize) {
        T max = input[startIndex];
        for (int64_t index = 1; index < prefixSize; ++index) {
            max = std::max(max, input[startIndex + index]);
        }

        T expSum = 0.0;
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

template <class T>
void TSlicedSoftmaxNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    auto* input = reinterpret_cast<const T*>(Inputs_[0]->GetOutput());
    auto* prefixSizePtr = reinterpret_cast<int64_t*>(Inputs_[1]->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    SlicedSoftmax(
        context.Stream,
        input,
        output,
        prefixSizePtr,
        GetElementCount(),
        GetShape().back());
}

} // namespace NHamKaas
