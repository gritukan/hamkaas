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

std::vector<TNodeBase*> TBufferNode::GetInputs() const
{
    return {};
}

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

std::vector<TNodeBase*> TConstantNode::GetInputs() const
{
    return {};
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

int64_t TSumNode::GetBufferSize() const
{
    return 2 * GetDimensions() * sizeof(int64_t);
}

void TSumNode::SetBuffer(char* buffer)
{
    LhsShape_ = reinterpret_cast<int64_t*>(buffer);
    RhsShapeMultiplier_ = LhsShape_ + GetDimensions();
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

void TSumNode::EvaluateGpu(const TEvaluationContext& context)
{
    //if (!Initialized_) {
        auto lhsShape = Lhs_->GetShape();

        std::vector<int64_t> rhsShapeMultiplier(Rhs_->GetDimensions());
        for (int index = 0; index < GetDimensions(); ++index) {
            auto rhsSize = Rhs_->GetShape()[index];
            if (lhsShape[index] == rhsSize) {
                rhsShapeMultiplier[index] = 1;
            } else {
                assert(rhsSize == 1);
                rhsShapeMultiplier[index] = 0;
            }
        }

        const auto* device = context.Device;
        device->CopyToDevice(LhsShape_, lhsShape.data(), GetDimensions() * sizeof(int64_t));
        device->CopyToDevice(RhsShapeMultiplier_, rhsShapeMultiplier.data(), GetDimensions() * sizeof(int64_t));

        //Initialized_ = true;
    //}

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
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Lhs_->GetElementCount(); ++index) {
        std::vector<int64_t> rhsIndices(Rhs_->GetDimensions());

        int64_t indexCopy = index;
        for (int64_t index = Rhs_->GetDimensions() - 1; index >= 0; --index) {
            rhsIndices[index] = indexCopy % Lhs_->GetShape()[index];
            indexCopy /= Lhs_->GetShape()[index];
            if (rhsIndices[index] >= Rhs_->GetShape()[index]) {
                assert(Rhs_->GetShape()[index] == 1);
                rhsIndices[index] = 0;
            }
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < Rhs_->GetDimensions(); ++index) {
            rhsIndex = rhsIndex * Rhs_->GetShape()[index] + rhsIndices[index];
        }

        output[index] = lhs[index] + rhs[rhsIndex];
    }
}

template <class T>
void TSumNode::DoEvaluateGpu(const TEvaluationContext& context)
{
    SumTensorsBroadcast(
        reinterpret_cast<const T*>(Lhs_->GetOutput()),
        reinterpret_cast<const T*>(Rhs_->GetOutput()),
        reinterpret_cast<T*>(GetOutput()),
        LhsShape_,
        RhsShapeMultiplier_,
        GetDimensions(),
        GetElementCount());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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

int64_t TMulNode::GetBufferSize() const
{
    int b = 1;
    if (Lhs_->GetDimensions() == 3) {
        b = Lhs_->GetShape()[0];
    }

    return 3 * Align(b) * sizeof(void*) + Align(GetOutputSize());
}

void TMulNode::SetBuffer(char* buffer)
{
    int b = 1;
    if (Lhs_->GetDimensions() == 3) {
        b = Lhs_->GetShape()[0];
    }

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

template <class T>
void TMulNode::DoEvaluateCpu()
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    int64_t b, n, m, k;
    if (Lhs_->GetDimensions() == 1) {
        b = 1;
        n = 1;
        m = Rhs_->GetShape()[1];
        k = Lhs_->GetShape()[0];
    } else if (Lhs_->GetDimensions() == 2) {
        b = 1;
        n = Lhs_->GetShape()[0];
        m = Rhs_->GetShape()[1];
        k = Lhs_->GetShape()[1];
    } else {
        b = Lhs_->GetShape()[0];
        n = Lhs_->GetShape()[1];
        m = Rhs_->GetShape()[2];
        k = Lhs_->GetShape()[2];
    }

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
    int64_t b, n, m, k;
    if (Lhs_->GetDimensions() == 1) {
        b = 1;
        n = 1;
        m = Rhs_->GetShape()[1];
        k = Lhs_->GetShape()[0];
    } else if (Lhs_->GetDimensions() == 2) {
        b = 1;
        n = Lhs_->GetShape()[0];
        m = Rhs_->GetShape()[1];
        k = Lhs_->GetShape()[1];
    } else {
        b = Lhs_->GetShape()[0];
        n = Lhs_->GetShape()[1];
        m = Rhs_->GetShape()[2];
        k = Lhs_->GetShape()[2];
    }

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

    //if (!Initialized_) {
        std::vector<void*> lhsMatrices(b);
        std::vector<void*> rhsMatrices(b);
        std::vector<void*> outputMatrices(b);

        for (int index = 0; index < b; ++index) {
            lhsMatrices[index] = Lhs_->GetOutput() + index * n * k * sizeof(T);
            rhsMatrices[index] = Rhs_->GetOutput() + index * k * m * sizeof(T);
            outputMatrices[index] = TransposedProductBuffer_ + index * n * m * sizeof(T);
        }

        const auto* device = context.Device;
        device->CopyToDevice(LhsMatrices_, lhsMatrices.data(), b * sizeof(char*));
        device->CopyToDevice(RhsMatrices_, rhsMatrices.data(), b * sizeof(char*));
        device->CopyToDevice(OutputMatrices_, outputMatrices.data(), b * sizeof(char*));

    //    Initialized_ = true;
    //}

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
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

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

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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

void TReLUNode::EvaluateCpu()
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

void TReLUNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    }
}

template <class T>
void TReLUNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Input_->GetElementCount(); ++index) {
        output[index] = std::max<T>(0.0, input[index]);
    }
}

template <class T>
void TReLUNode::DoEvaluateGpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    ReLU(input, output, Input_->GetElementCount());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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

void TSiLUNode::EvaluateCpu()
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

void TSiLUNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    }
}

template <class T>
void TSiLUNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Input_->GetElementCount(); ++index) {
        output[index] = input[index] / (1 + exp(-input[index]));
    }
}

template <class T>
void TSiLUNode::DoEvaluateGpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    SiLU(input, output, Input_->GetElementCount());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

TSliceNode::TSliceNode(TNodeBasePtr input, int64_t begin, int64_t end)
    : TNodeBase(CalculateMeta(input->GetMeta(), begin, end))
    , Input_(std::move(input))
    , Begin_(begin)
    , End_(end)
{ }

const TNodeBasePtr& TSliceNode::GetInput() const
{
    return Input_;
}

int64_t TSliceNode::GetBegin() const
{
    return Begin_;
}

int64_t TSliceNode::GetEnd() const
{
    return End_;
}

std::vector<TNodeBase*> TSliceNode::GetInputs() const
{
    return {Input_.get()};
}

TNodeBase* TSliceNode::GetOutputOwner() const
{
    return Input_->GetOutputOwner();
}

void TSliceNode::EvaluateCpu()
{
    int64_t stride = 1;
    for (int64_t index = 1; index < GetDimensions(); ++index) {
        stride *= GetShape()[index];
    }

    Output_ = Input_->GetOutput() + Begin_ * stride * GetElementSize();
}

void TSliceNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    // TODO: Refactor to avoid copypaste.
    int64_t stride = 1;
    for (int64_t index = 1; index < GetDimensions(); ++index) {
        stride *= GetShape()[index];
    }

    Output_ = Input_->GetOutput() + Begin_ * stride * GetElementSize();
}

int64_t TSliceNode::GetOutputSize() const
{
    // We do not need to allocate memory for the slice
    // as it is just a view of the input tensor.
    return 0;
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

TRMSNormNode::TRMSNormNode(TNodeBasePtr input, TNodeBasePtr weights)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
    , Weights_(std::move(weights))
{
    if (Input_->GetValueType() != Weights_->GetValueType()) {
        THROW("Different value types", VAR(Input_->GetValueType()), VAR(Weights_->GetValueType()));
    }
    if (Input_->GetDimensions() != 1) {
        THROW("RMS normalization is supported for vectors only", VAR(Input_->GetDimensions()));
    }
    if (Input_->GetShape() != Weights_->GetShape()) {
        THROW("Different shapes", VAR(Input_->GetShape()), VAR(Weights_->GetShape()));
    }
}

const TNodeBasePtr& TRMSNormNode::GetInput() const
{
    return Input_;
}

const TNodeBasePtr& TRMSNormNode::GetWeights() const
{
    return Weights_;
}

std::vector<TNodeBase*> TRMSNormNode::GetInputs() const
{
    return {Input_.get(), Weights_.get()};
}

void TRMSNormNode::EvaluateCpu()
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

void TRMSNormNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TRMSNormNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* weights = reinterpret_cast<const T*>(Weights_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    T sum = 0.0;
    for (int64_t index = 0; index < Input_->GetElementCount(); ++index) {
        sum += input[index] * input[index];
    }
    sum /= Input_->GetElementCount();
    sum += 1e-5;
    sum = 1.0 / sqrt(sum);

    for (int64_t index = 0; index < Input_->GetElementCount(); ++index) {
        output[index] = weights[index] * (input[index] * sum);
    }
}

template <class T>
void TRMSNormNode::DoEvaluateGpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* weights = reinterpret_cast<const T*>(Weights_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    RMSNorm(input, weights, output, Input_->GetElementCount(), /*epsilon*/ T(1e-5));

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

TReshapeNode::TReshapeNode(TNodeBasePtr input, std::vector<int64_t> shape)
    : TNodeBase(CalculateMeta(input->GetMeta(), shape))
    , Input_(std::move(input))
    , Shape_(std::move(shape))
{ }

const TNodeBasePtr& TReshapeNode::GetInput() const
{
    return Input_;
}

const std::vector<int64_t>& TReshapeNode::GetShape() const
{
    return Shape_;
}

std::vector<TNodeBase*> TReshapeNode::GetInputs() const
{
    return {Input_.get()};
}

int64_t TReshapeNode::GetOutputSize() const
{
    // Reshape node returns a view of the input tensor.
    return 0;
}

TNodeBase* TReshapeNode::GetOutputOwner() const
{
    return Input_->GetOutputOwner();
}

void TReshapeNode::EvaluateCpu()
{
    Output_ = Input_->GetOutput();
}

void TReshapeNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    Output_ = Input_->GetOutput();
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
    : TNodeBase(lhs->GetMeta())
    , Lhs_(std::move(lhs))
    , Rhs_(std::move(rhs))
{
    if (Lhs_->GetValueType() != Rhs_->GetValueType()) {
        THROW("Different value types", VAR(Lhs_->GetValueType()), VAR(Rhs_->GetValueType()));
    }
    if (Lhs_->GetDimensions() != 2 || Rhs_->GetDimensions() != 2) {
        THROW("Hadamard product is supported for complex vectors only", VAR(Lhs_->GetDimensions()), VAR(Rhs_->GetDimensions()));
    }
    if (Lhs_->GetShape()[1] != 2 || Rhs_->GetShape()[1] != 2) {
        THROW("Hadamard product is supported for complex vectors only", VAR(Lhs_->GetShape()[1]), VAR(Rhs_->GetShape()[1]));
    }
    if (Lhs_->GetShape()[0] != Rhs_->GetShape()[0]) {
        THROW("Different shapes", VAR(Lhs_->GetShape()[0]), VAR(Rhs_->GetShape()[0]));
    }
}

const TNodeBasePtr& TComplexHadamardProductNode::GetLhs() const
{
    return Lhs_;
}

const TNodeBasePtr& TComplexHadamardProductNode::GetRhs() const
{
    return Rhs_;
}

std::vector<TNodeBase*> TComplexHadamardProductNode::GetInputs() const
{
    return {Lhs_.get(), Rhs_.get()};
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

void TComplexHadamardProductNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    }
}

template <typename T>
void TComplexHadamardProductNode::DoEvaluateCpu()
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t index = 0; index < Lhs_->GetShape()[0]; ++index) {
        int64_t lhsIndex = index * 2;
        int64_t rhsIndex = index * 2;

        auto real = lhs[lhsIndex] * rhs[rhsIndex] - lhs[lhsIndex + 1] * rhs[rhsIndex + 1];
        auto imag = lhs[lhsIndex] * rhs[rhsIndex + 1] + lhs[lhsIndex + 1] * rhs[rhsIndex];

        output[lhsIndex] = real;
        output[lhsIndex + 1] = imag;
    }
}

template <typename T>
void TComplexHadamardProductNode::DoEvaluateGpu()
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    ComplexHadamardProduct(lhs, rhs, output, Lhs_->GetShape()[0]);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

THadamardProductNode::THadamardProductNode(TNodeBasePtr lhs, TNodeBasePtr rhs)
    : TNodeBase(lhs->GetMeta())
    , Lhs_(std::move(lhs))
    , Rhs_(std::move(rhs))
{
    if (Lhs_->GetValueType() != Rhs_->GetValueType()) {
        THROW("Different value types", VAR(Lhs_->GetValueType()), VAR(Rhs_->GetValueType()));
    }
    if (Lhs_->GetDimensions() != Rhs_->GetDimensions()) {
        THROW("Different number of dimensions", VAR(Lhs_->GetDimensions()), VAR(Rhs_->GetDimensions()));
    }
    for (int64_t index = 0; index < Lhs_->GetDimensions(); ++index) {
        if (Lhs_->GetShape()[index] != Rhs_->GetShape()[index] && Rhs_->GetShape()[index] != 1) {
            THROW("Incompatible shapes", VAR(Lhs_->GetShape()[index]), VAR(Rhs_->GetShape()[index]));
        }
    }
}

const TNodeBasePtr& THadamardProductNode::GetLhs() const
{
    return Lhs_;
}

const TNodeBasePtr& THadamardProductNode::GetRhs() const
{
    return Rhs_;
}

std::vector<TNodeBase*> THadamardProductNode::GetInputs() const
{
    return {Lhs_.get(), Rhs_.get()};
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

void THadamardProductNode::EvaluateGpu(const TEvaluationContext& context)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <typename T>
void THadamardProductNode::DoEvaluateCpu()
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    for (int64_t lhsIndex = 0; lhsIndex < Lhs_->GetElementCount(); ++lhsIndex) {
        std::vector<int64_t> rhsIndices(Rhs_->GetDimensions());
        int64_t lhsIndexCopy = lhsIndex;
        for (int64_t index = Rhs_->GetDimensions() - 1; index >= 0; --index) {
            rhsIndices[index] = lhsIndexCopy % Lhs_->GetShape()[index];
            lhsIndexCopy /= Lhs_->GetShape()[index];
            if (rhsIndices[index] >= Rhs_->GetShape()[index]) {
                assert(Rhs_->GetShape()[index] == 1);
                rhsIndices[index] = 0;
            }
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < Rhs_->GetDimensions(); ++index) {
            rhsIndex = rhsIndex * Rhs_->GetShape()[index] + rhsIndices[index];
        }

        output[lhsIndex] = lhs[lhsIndex] * rhs[rhsIndex];
    }
}

template <typename T>
void THadamardProductNode::DoEvaluateGpu()
{
    auto* lhs = reinterpret_cast<const T*>(Lhs_->GetOutput());
    auto* rhs = reinterpret_cast<const T*>(Rhs_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    HadamardProduct(lhs, rhs, output, GetElementCount());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

TPermuteNode::TPermuteNode(TNodeBasePtr input, std::vector<int64_t> permutation)
    : TNodeBase(CalculateMeta(input->GetMeta(), permutation))
    , Input_(std::move(input))
    , Permutation_(std::move(permutation))
{ }

const TNodeBasePtr& TPermuteNode::GetInput() const
{
    return Input_;
}

const std::vector<int64_t>& TPermuteNode::GetPermutation() const
{
    return Permutation_;
}

std::vector<TNodeBase*> TPermuteNode::GetInputs() const
{
    return {Input_.get()};
}

void TPermuteNode::EvaluateCpu()
{
    auto dimensions = GetDimensions();
    auto elementCount = GetElementCount();
    std::vector<int64_t> inputIndices(GetDimensions());

    for (int64_t inputIndex = 0; inputIndex < elementCount; ++inputIndex) {
        int64_t indexCopy = inputIndex;
        const auto& inputShape = Input_->GetShape();
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
            Input_->GetOutput() + inputIndex * GetElementSize(),
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

void TPermuteNode::EvaluateGpu(const TEvaluationContext& context)
{
    //if (!Initialized_) {
        const auto* device = context.Device;
        device->CopyToDevice(InputShape_, Input_->GetShape().data(), GetDimensions() * sizeof(int64_t));
        device->CopyToDevice(OutputShape_, GetShape().data(), GetDimensions() * sizeof(int64_t));
        device->CopyToDevice(PermutationPtr_, Permutation_.data(), GetDimensions() * sizeof(int64_t));

        //Initialized_ = true;
    //}

    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    }
}

template <class T>
void TPermuteNode::DoEvaluateGpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    Permute(
        input,
        output,
        InputShape_,
        OutputShape_,
        PermutationPtr_,
        GetDimensions(),
        GetElementCount());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
    , Replacement_(std::move(replacement))
    , Begin_(std::move(begin))
    , End_(std::move(end))
{
    if (Input_->GetValueType() != Replacement_->GetValueType()) {
        THROW("Different value types", VAR(Input_->GetValueType()), VAR(Replacement_->GetValueType()));
    }
    if (Input_->GetDimensions() != Replacement_->GetDimensions()) {
        THROW("Different number of dimensions", VAR(Input_->GetDimensions()), VAR(Replacement_->GetDimensions()));
    }
    if (Begin_->GetDimensions() != 1 || End_->GetDimensions() != 1) {
        THROW("Begin and end should be 1D tensors", VAR(Begin_->GetDimensions()), VAR(End_->GetDimensions()));
    }
    if (Begin_->GetShape()[0] != 1 || End_->GetShape()[0] != 1) {
        THROW("Begin and end should be 1D tensors", VAR(Begin_->GetShape()[0]), VAR(End_->GetShape()[0]));
    }
    if (Begin_->GetValueType() != EValueType::Int64 || End_->GetValueType() != EValueType::Int64) {
        THROW("Begin and end should be int64 tensors", VAR(Begin_->GetValueType()), VAR(End_->GetValueType()));
    }
}

const TNodeBasePtr& TReplaceSliceNode::GetInput() const
{
    return Input_;
}

const TNodeBasePtr& TReplaceSliceNode::GetReplacement() const
{
    return Replacement_;
}

const TNodeBasePtr& TReplaceSliceNode::GetBegin() const
{
    return Begin_;
}

const TNodeBasePtr& TReplaceSliceNode::GetEnd() const
{
    return End_;
}

std::vector<TNodeBase*> TReplaceSliceNode::GetInputs() const
{
    return {Input_.get(), Replacement_.get(), Begin_.get(), End_.get()};
}

int64_t TReplaceSliceNode::GetOutputSize() const
{
    // We do not need to allocate memory for the slice
    // as it is passes modified input further.
    return 0;
}

TNodeBase* TReplaceSliceNode::GetOutputOwner() const
{
    return Input_->GetOutputOwner();
}

void TReplaceSliceNode::EvaluateCpu()
{
    auto* replacement = Replacement_->GetOutput();
    auto begin = *reinterpret_cast<const int64_t*>(Begin_->GetOutput());
    auto end = *reinterpret_cast<const int64_t*>(End_->GetOutput());
    auto* output = Input_->GetOutput();

    int64_t stride = 1;
    for (int64_t index = 1; index < GetDimensions(); ++index) {
        stride *= GetShape()[index];
    }

    assert(end - begin == Replacement_->GetElementCount());
    assert(begin >= 0);
    assert(end <= Input_->GetElementCount());

    memcpy(
        output + begin * stride * GetElementSize(),
        replacement,
        (end - begin) * stride * GetElementSize());

    Output_ = Input_->GetOutput();
}

void TReplaceSliceNode::EvaluateGpu(const TEvaluationContext& context)
{
    auto* replacement = Replacement_->GetOutput();

    int64_t begin;
    int64_t end;
    CUDA_CHECK_ERROR(cudaMemcpy(&begin, Begin_->GetOutput(), sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(&end, End_->GetOutput(), sizeof(int64_t), cudaMemcpyDeviceToHost));

    auto* output = Input_->GetOutput();

    int64_t stride = 1;
    for (int64_t index = 1; index < GetDimensions(); ++index) {
        stride *= GetShape()[index];
    }

    assert(end - begin == Replacement_->GetElementCount());
    assert(begin >= 0);
    assert(end <= Input_->GetElementCount());

    context.Device->DeviceCopy(
        output + begin * stride * GetElementSize(),
        replacement,
        (end - begin) * stride * GetElementSize());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    Output_ = Input_->GetOutput();
}

TSlicedSoftmaxNode::TSlicedSoftmaxNode(TNodeBasePtr input, TNodeBasePtr prefixSize)
    : TNodeBase(input->GetMeta())
    , Input_(std::move(input))
    , PrefixSize_(std::move(prefixSize))
{
    if (PrefixSize_->GetDimensions() != 1) {
        THROW("Prefix size should be a 1D tensor", VAR(PrefixSize_->GetDimensions()));
    }
    if (PrefixSize_->GetShape()[0] != 1) {
        THROW("Prefix size should be a 1D tensor", VAR(PrefixSize_->GetShape()[0]));
    }
    if (PrefixSize_->GetValueType() != EValueType::Int64) {
        THROW("Prefix size should be an int64 tensor", VAR(PrefixSize_->GetValueType()));
    }
}

const TNodeBasePtr& TSlicedSoftmaxNode::GetInput() const
{
    return Input_;
}

const TNodeBasePtr& TSlicedSoftmaxNode::GetPrefixSize() const
{
    return PrefixSize_;
}

std::vector<TNodeBase*> TSlicedSoftmaxNode::GetInputs() const
{
    return {Input_.get(), PrefixSize_.get()};
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

void TSlicedSoftmaxNode::EvaluateGpu(const TEvaluationContext& /*context*/)
{
    switch (GetValueType()) {
    case EValueType::Float32:
        DoEvaluateGpu<float>();
        return;
    case EValueType::Float64:
        DoEvaluateGpu<double>();
        return;
    default:
        THROW("GPU inference does not support this value type", VAR(GetValueType()));
    }
}

template <class T>
void TSlicedSoftmaxNode::DoEvaluateCpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto prefixSize = *reinterpret_cast<const int64_t*>(PrefixSize_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    assert(prefixSize >= 1);
    assert(prefixSize <= GetElementCount());

    T max = input[0];
    for (int64_t index = 1; index < prefixSize; ++index) {
        max = std::max(max, input[index]);
    }

    T expSum = 0.0;
    for (int64_t index = 0; index < prefixSize; ++index) {
        expSum += exp(input[index] - max);
    }

    for (int64_t index = 0; index < prefixSize; ++index) {
        output[index] = exp(input[index] - max) / expSum;
    }

    for (int64_t index = prefixSize; index < GetElementCount(); ++index) {
        output[index] = input[index];
    }
}

template <class T>
void TSlicedSoftmaxNode::DoEvaluateGpu()
{
    auto* input = reinterpret_cast<const T*>(Input_->GetOutput());
    auto* prefixSizePtr = reinterpret_cast<int64_t*>(PrefixSize_->GetOutput());
    auto* output = reinterpret_cast<T*>(GetOutput());

    SlicedSoftmax(input, output, prefixSizePtr, GetElementCount());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

} // namespace NHamKaas
