#include <cstdint>

#include <cassert>
#include <cstdio>

#include "helpers.h"
#include "tensor.h"

namespace NHamKaas {

constexpr int64_t MaxThreadsPerBlock = 256;
constexpr int64_t MaxBlockCount = 65535;

#define FOR_ALL_FLOAT_TYPES(XX) \
    XX(float) \
    XX(double) \

template <class T>
__global__ void SumTensorsBroadcastKernel(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    int64_t indices[MaxDimensions];

    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t lhsIndex = threadIndex; lhsIndex < outputSize; lhsIndex += gridDim.x * blockDim.x) {
        int64_t lhsIndexCopy = lhsIndex;

        for (int64_t i = dimensions - 1; i >= 0; --i) {
            indices[i] = lhsIndexCopy % lhsShape[i];
            lhsIndexCopy /= lhsShape[i];
        }

        int64_t rhsIndex = 0;
        for (int64_t i = 0; i < dimensions; ++i) {
            int64_t index = rhsShape[i] == 1 ? 0 : indices[i];
            rhsIndex = rhsIndex * rhsShape[i] + index;
        }

        output[lhsIndex] = lhs[lhsIndex] + rhs[rhsIndex];
    }
}

template <class T>
void SumTensorsBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (outputSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    SumTensorsBroadcastKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        lhs,
        rhs,
        output,
        lhsShape,
        rhsShape,
        dimensions,
        outputSize);
}

#define INSTANTIATE(T) \
    template void SumTensorsBroadcast( \
        cudaStream_t stream, \
        const T* lhs, \
        const T* rhs, \
        T* output, \
        int64_t* lhsShape, \
        int64_t* rhsShape, \
        int64_t dimensions, \
        int64_t outputSize);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void ReLUKernel(
    const T* input,
    T* output,
    int64_t size)
{
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t index = threadIndex; index < size; index += gridDim.x * blockDim.x) {
        output[index] = input[index] > 0 ? input[index] : 0;
    }
}

template <class T>
void ReLU(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t size)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    ReLUKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(input, output, size);
}

#define INSTANTIATE(T) \
    template void ReLU( \
        cudaStream_t stream, \
        const T* input, \
        T* output, \
        int64_t size);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void SiLUKernel(
    const T* input,
    T* output,
    int64_t size)
{
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t index = threadIndex; index < size; index += gridDim.x * blockDim.x) {
        output[index] = input[index] / (1 + exp(-input[index]));
    }
}

template <class T>
void SiLU(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t size)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    SiLUKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(input, output, size);
}

#define INSTANTIATE(T) \
    template void SiLU( \
        cudaStream_t stream, \
        const T* input, \
        T* output, \
        int64_t size);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void RMSNormKernel(
    const T* input,
    const T* weights,
    T* output,
    int64_t size,
    T epsilon)
{
    assert(blockIdx.x == 0);

    __shared__ T blockSum[MaxThreadsPerBlock];
    __shared__ T sharedNorm;

    T localSum = 0;
    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        localSum += input[i] * input[i];
    }
    blockSum[threadIdx.x] = localSum;

    __syncthreads();

    if (threadIdx.x == 0) {
        T norm = 0;
        for (int64_t i = 0; i < blockDim.x; ++i) {
            norm += blockSum[i];
        }
        norm /= size;
        norm += epsilon;
        norm = 1.0 / sqrt(norm);
        sharedNorm = norm;
    }

    __syncthreads();

    T norm = sharedNorm;

    for (int64_t i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = weights[i] * (input[i] * norm);
    }
}

template <class T>
void RMSNorm(
    cudaStream_t stream,
    const T* input,
    const T* weights,
    T* output,
    int64_t size,
    T epsilon)
{
    constexpr int64_t ThreadsPerBlock = 256;
    RMSNormKernel<T><<<1, ThreadsPerBlock, 0, stream>>>(input, weights, output, size, epsilon);
}

#define INSTANTIATE(T) \
    template void RMSNorm( \
        cudaStream_t stream, \
        const T* input, \
        const T* weights, \
        T* output, \
        int64_t size, \
        T epsilon);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void ComplexHadamardProductBroadcastKernel(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    int64_t indices[MaxDimensions];

    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t lhsIndex = threadIndex; lhsIndex < outputSize / 2; lhsIndex += gridDim.x * blockDim.x) {
        int64_t lhsIndexCopy = lhsIndex;

        for (int64_t i = dimensions - 2; i >= 0; --i) {
            indices[i] = lhsIndexCopy % lhsShape[i];
            lhsIndexCopy /= lhsShape[i];
        }

        int64_t rhsIndex = 0;
        for (int64_t i = 0; i + 1 < dimensions; ++i) {
            int64_t index = rhsShape[i] == 1 ? 0 : indices[i];
            rhsIndex = rhsIndex * rhsShape[i] + index;
        }

        output[2 * lhsIndex] = lhs[2 * lhsIndex] * rhs[2 * rhsIndex] - lhs[2 * lhsIndex + 1] * rhs[2 * rhsIndex + 1];
        output[2 * lhsIndex + 1] = lhs[2 * lhsIndex] * rhs[2 * rhsIndex + 1] + lhs[2 * lhsIndex + 1] * rhs[2 * rhsIndex];
    }
}

template <class T>
void ComplexHadamardProductBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (outputSize / 2 + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    ComplexHadamardProductBroadcastKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        lhs,
        rhs,
        output,
        lhsShape,
        rhsShape,
        dimensions,
        outputSize);
}

#define INSTANTIATE(T) \
    template void ComplexHadamardProductBroadcast( \
        cudaStream_t stream, \
        const T* lhs, \
        const T* rhs, \
        T* output, \
        int64_t* lhsShape, \
        int64_t* rhsShape, \
        int64_t dimensions, \
        int64_t outputSize);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void HadamardProductBroadcastKernel(
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    int64_t indices[MaxDimensions];

    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t lhsIndex = threadIndex; lhsIndex < outputSize; lhsIndex += gridDim.x * blockDim.x) {
        int64_t lhsIndexCopy = lhsIndex;

        for (int64_t i = dimensions - 1; i >= 0; --i) {
            indices[i] = lhsIndexCopy % lhsShape[i];
            lhsIndexCopy /= lhsShape[i];
        }

        int64_t rhsIndex = 0;
        for (int64_t i = 0; i < dimensions; ++i) {
            int64_t index = rhsShape[i] == 1 ? 0 : indices[i];
            rhsIndex = rhsIndex * rhsShape[i] + index;
        }

        output[lhsIndex] = lhs[lhsIndex] * rhs[rhsIndex];
    }
}

template <class T>
void HadamardProductBroadcast(
    cudaStream_t stream,
    const T* lhs,
    const T* rhs,
    T* output,
    int64_t* lhsShape,
    int64_t* rhsShape,
    int64_t dimensions,
    int64_t outputSize)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (outputSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    HadamardProductBroadcastKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        lhs,
        rhs,
        output,
        lhsShape,
        rhsShape,
        dimensions,
        outputSize);
}

#define INSTANTIATE(T) \
    template void HadamardProductBroadcast( \
        cudaStream_t stream, \
        const T* lhs, \
        const T* rhs, \
        T* output, \
        int64_t* lhsShape, \
        int64_t* rhsShape, \
        int64_t dimensions, \
        int64_t outputSize);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void SoftmaxKernel(
    const T* input,
    T* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize)
{
    __shared__ T buffer[MaxThreadsPerBlock];

    int64_t prefixSize = *prefixSizePtr;

    for (int64_t vectorIndex = blockIdx.x; vectorIndex < size / vectorSize; vectorIndex += gridDim.x) {
        const T* in = input + vectorIndex * vectorSize;
        T* out = output + vectorIndex * vectorSize;

        if (threadIdx.x < prefixSize) {
            T max = in[threadIdx.x];
            for (int64_t index = threadIdx.x; index < prefixSize; index += blockDim.x) {
                max = max > in[index] ? max : in[index];
            }

            buffer[threadIdx.x] = max;
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            T max = buffer[0];
            for (int64_t i = 1; i < prefixSize && i < blockDim.x; ++i) {
                max = max > buffer[i] ? max : buffer[i];
            }

            buffer[threadIdx.x] = max;
        }

        __syncthreads();

        T max = buffer[0];
        T sum = 0;
        for (int64_t index = threadIdx.x; index < prefixSize; index += blockDim.x) {
            sum += exp(in[index] - max);
        }

        buffer[threadIdx.x] = sum;

        __syncthreads();

        if (threadIdx.x == 0) {
            T sum = 0;
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

template <class T>
void SlicedSoftmax(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t* prefixSizePtr,
    int64_t size,
    int64_t vectorSize)
{
    constexpr int64_t ThreadsPerBlock = 256;

    int64_t blocks = std::min(MaxBlockCount, size / vectorSize);
    SoftmaxKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        output,
        prefixSizePtr,
        size,
        vectorSize);
}

#define INSTANTIATE(T) \
    template void SlicedSoftmax( \
        cudaStream_t stream, \
        const T* input, \
        T* output, \
        int64_t* prefixSizePtr, \
        int64_t size, \
        int64_t vectorSize);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void ReplaceKernel(
    T* input,
    int64_t inputSize,
    const T* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end)
{
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t index = threadIndex; index < replacementSize; index += gridDim.x * blockDim.x) {
        input[index + *begin] = replacement[index];
    }
}

template <class T>
void ReplaceSlice(
    cudaStream_t stream,
    T* input,
    int64_t inputSize,
    const T* replacement,
    int64_t replacementSize,
    const int64_t* begin,
    const int64_t* end)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (replacementSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    ReplaceKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        inputSize,
        replacement,
        replacementSize,
        begin,
        end);
}

#define INSTANTIATE(T) \
    template void ReplaceSlice( \
        cudaStream_t stream, \
        T* input, \
        int64_t inputSize, \
        const T* replacement, \
        int64_t replacementSize, \
        const int64_t* begin, \
        const int64_t* end);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

template <class T>
__global__ void PermuteKernel(
    const T* input,
    T* output,
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

template <class T>
void Permute(
    cudaStream_t stream,
    const T* input,
    T* output,
    int64_t* inputShape,
    int64_t* outputShape,
    int64_t* permutation,
    int64_t dimensions,
    int64_t size)
{
    constexpr int64_t ThreadsPerBlock = 256;
    int64_t blocks = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;
    blocks = std::min(blocks, MaxBlockCount);

    PermuteKernel<T><<<blocks, ThreadsPerBlock, 0, stream>>>(
        input,
        output,
        inputShape,
        outputShape,
        permutation,
        dimensions,
        size);
}

#define INSTANTIATE(T) \
    template void Permute( \
        cudaStream_t stream, \
        const T* input, \
        T* output, \
        int64_t* inputShape, \
        int64_t* outputShape, \
        int64_t* permutation, \
        int64_t dimensions, \
        int64_t size);
FOR_ALL_FLOAT_TYPES(INSTANTIATE)
#undef INSTANTIATE

} // namespace NHamKaas
