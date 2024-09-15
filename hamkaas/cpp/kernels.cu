#include <cstdint>

#include <cassert>
#include <cstdio>

#include "helpers.h"
#include "tensor.h"

namespace NHamKaas {

constexpr int64_t MaxThreadsPerBlock = 256;
constexpr int64_t MaxBlockCount = 65535;

template <EPointwiseOperation Operation>
__device__ void DoPointwise(const float* in, float* out)
{
    if constexpr (Operation == EPointwiseOperation::ReLU) {
        *out = *in > 0 ? *in : 0;
    } else if constexpr (Operation == EPointwiseOperation::SiLU) {
        *out = *in / (1 + exp(-*in));
        out[0] = in[0] / (1 + exp(-in[0]));
    } else {
        assert(false);
    }
}

template <EPointwiseOperation Operation>
__device__ void DoPointwise(
    const float* lhs,
    const float* rhs,
    float* out)
{
    if constexpr (Operation == EPointwiseOperation::Add) {
        *out = *lhs + *rhs;
    } else if constexpr (Operation == EPointwiseOperation::HadamardProduct) {
        *out = *lhs * *rhs;
    } else if constexpr (Operation == EPointwiseOperation::ComplexHadamardProduct) {
        out[0] = lhs[0] * rhs[0] - lhs[1] * rhs[1];
        out[1] = lhs[0] * rhs[1] + lhs[1] * rhs[0];
    } else {
        assert(false);
    }
}

template <EPointwiseOperation Operation>
__global__ void PointwiseKernel(
    const float* lhs,
    const float* rhs,
    float* out,
    int64_t size,
    int64_t step)
{
    for (
        int64_t index = step * (blockIdx.x * blockDim.x + threadIdx.x);
        index < size;
        index += step * gridDim.x * blockDim.x)
    {
        if constexpr (IsBinary(Operation)) {
            DoPointwise<Operation>(lhs + index, rhs + index, out + index);
        } else {
            DoPointwise<Operation>(lhs + index, out + index);
        }
    }
}

template <EPointwiseOperation Operation>
__global__ void PointwiseKernelBroadcast(
    const float* lhs,
    const float* rhs,
    float* out,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize,
    int64_t step)
{
    int64_t rhsIndices[MaxDimensions];

    for (
        int64_t lhsIndex = step * (blockIdx.x * blockDim.x + threadIdx.x);
        lhsIndex < outputSize;
        lhsIndex += step * gridDim.x * blockDim.x)
    {
        int64_t lhsIndexCopy = lhsIndex;
        for (int64_t index = dimensions - 1; index >= 0; --index) {
            rhsIndices[index] = lhsIndexCopy % lhsShape[index];
            if (rhsIndices[index] >= rhsShape[index]) {
                rhsIndices[index] = 0;
            }

            lhsIndexCopy /= lhsShape[index];
        }

        int64_t rhsIndex = 0;
        for (int64_t index = 0; index < dimensions; ++index) {
            rhsIndex = rhsIndex * rhsShape[index] + rhsIndices[index];
        }

        if constexpr (IsBinary(Operation)) {
            DoPointwise<Operation>(lhs + lhsIndex, rhs + rhsIndex, out + lhsIndex);
        } else {
            DoPointwise<Operation>(lhs + lhsIndex, out + lhsIndex);
        }
    }
}

template <EPointwiseOperation Operation>
void Pointwise(
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* out,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    bool needBroadcasting,
    int64_t outputSize)
{
    constexpr int64_t ThreadsPerBlock = 256;

    int64_t threadCount = outputSize;
    int64_t step = 1;
    if constexpr (Operation == EPointwiseOperation::ComplexHadamardProduct) {
        threadCount /= 2;
        step = 2;
    }

    int64_t blocks = (threadCount + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    if (needBroadcasting) {
        PointwiseKernelBroadcast<Operation><<<blocks, ThreadsPerBlock, 0, stream>>>(
            lhs,
            rhs,
            out,
            lhsShape,
            rhsShape,
            dimensions,
            outputSize,
            step);
    } else {
        PointwiseKernel<Operation><<<blocks, ThreadsPerBlock, 0, stream>>>(
            lhs,
            rhs,
            out,
            outputSize,
            step);
    }
}

#define INSTANTIATE_POINTWISE(Operation) \
    template void Pointwise<Operation>( \
        cudaStream_t stream, \
        const float* lhs, \
        const float* rhs, \
        float* out, \
        int64_t* lhsShape, \
        int64_t* rhsShape, \
        int64_t dimensions, \
        bool needBroadcasting, \
        int64_t outputSize);
INSTANTIATE_POINTWISE(EPointwiseOperation::Add)
INSTANTIATE_POINTWISE(EPointwiseOperation::HadamardProduct)
INSTANTIATE_POINTWISE(EPointwiseOperation::ComplexHadamardProduct)
INSTANTIATE_POINTWISE(EPointwiseOperation::ReLU)
INSTANTIATE_POINTWISE(EPointwiseOperation::SiLU)
#undef INSTANTIATE_POINTWISE

__global__ void RMSNormKernel(
    const float* input,
    const float* weights,
    float* output,
    int64_t size,
    float epsilon)
{
    assert(blockIdx.x == 0);

    __shared__ float blockSum[MaxThreadsPerBlock];
    __shared__ float sharedNorm;

    float localSum = 0;
    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        localSum += input[i] * input[i];
    }
    blockSum[threadIdx.x] = localSum;

    __syncthreads();

    if (threadIdx.x == 0) {
        float norm = 0;
        for (int64_t i = 0; i < blockDim.x; ++i) {
            norm += blockSum[i];
        }
        norm /= size;
        norm += epsilon;
        norm = 1.0 / sqrt(norm);
        sharedNorm = norm;
    }

    __syncthreads();

    float norm = sharedNorm;

    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = weights[i] * (input[i] * norm);
    }
}

void RMSNorm(
    cudaStream_t stream,
    const float* input,
    const float* weights,
    float* output,
    int64_t size,
    float epsilon)
{
    constexpr int64_t ThreadsPerBlock = 256;
    RMSNormKernel<<<1, ThreadsPerBlock, 0, stream>>>(input, weights, output, size, epsilon);
}

__global__ void SoftmaxKernel(
    const float* input,
    float* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize)
{
    __shared__ float buffer[MaxThreadsPerBlock];

    int64_t prefixSize = *prefixSizePtr;

    for (int64_t vectorIndex = blockIdx.x; vectorIndex < size / vectorSize; vectorIndex += gridDim.x) {
        const float* in = input + vectorIndex * vectorSize;
        float* out = output + vectorIndex * vectorSize;

        if (threadIdx.x < prefixSize) {
            float max = in[threadIdx.x];
            for (int64_t index = threadIdx.x; index < prefixSize; index += blockDim.x) {
                max = max > in[index] ? max : in[index];
            }

            buffer[threadIdx.x] = max;
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            float max = buffer[0];
            for (int64_t i = 1; i < prefixSize && i < blockDim.x; ++i) {
                max = max > buffer[i] ? max : buffer[i];
            }

            buffer[threadIdx.x] = max;
        }

        __syncthreads();

        float max = buffer[0];
        float sum = 0;
        for (int64_t index = threadIdx.x; index < prefixSize; index += blockDim.x) {
            sum += exp(in[index] - max);
        }

        buffer[threadIdx.x] = sum;

        __syncthreads();

        if (threadIdx.x == 0) {
            float sum = 0;
            for (int64_t i = 0; i < prefixSize && i < blockDim.x; ++i) {
                sum += buffer[i];
            }

            buffer[threadIdx.x] = sum;
        }

        __syncthreads();

        sum = buffer[0];

        for (int64_t index = threadIdx.x; index < vectorSize; index += blockDim.x) {
            if (index < prefixSize) {
                out[index] = exp(in[index] - max) / sum;
            } else {
                out[index] = in[index];
            }
        }

        __syncthreads();
    }
}

void SlicedSoftmax(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize)
{
    constexpr int64_t ThreadsPerBlock = 256;

    int64_t blocks = std::min(MaxBlockCount, size / vectorSize);
    SoftmaxKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        output,
        prefixSizePtr,
        size,
        vectorSize);
}

__global__ void ReplaceKernel(
    float* input,
    int64_t inputSize,
    const float* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end)
{
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t index = threadIndex; index < replacementSize; index += gridDim.x * blockDim.x) {
        input[index + *begin] = replacement[index];
    }
}

void ReplaceSlice(
    cudaStream_t stream,
    float* input,
    int64_t inputSize,
    const float* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (replacementSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    ReplaceKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        inputSize,
        replacement,
        replacementSize,
        begin,
        end);
}

__global__ void PermuteKernel(
    const float* input,
    float* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size)
{
    int64_t indices[MaxDimensions];

    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t inputIndex = threadIndex; inputIndex < size; inputIndex += gridDim.x * blockDim.x) {
        int64_t inputIndexCopy = inputIndex;

        for (int64_t i = dimensions - 1; i >= 0; --i) {
            indices[i] = inputIndexCopy % inputShape[i];
            inputIndexCopy /= inputShape[i];
        }

        int64_t outputIndex = 0;
        for (int64_t i = 0; i < dimensions; ++i) {
            outputIndex = outputIndex * outputShape[i] + indices[permutation[i]];
        }

        output[outputIndex] = input[inputIndex];
    }
}

void Permute(
    cudaStream_t stream,
    const float* input,
    float* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    PermuteKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        output,
        inputShape,
        outputShape,
        permutation,
        dimensions,
        size);
}

} // namespace NHamKaas
