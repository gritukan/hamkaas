#include <cstdint>

#include <cassert>
#include <cstdio>

#include "helpers.h"
#include "tensor.h"

namespace NHamKaas {

constexpr int64_t MaxThreadsPerBlock = 256;
constexpr int64_t MaxBlockCount = 65535;

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
    // (lab4/03): Implement the pointwise operations.
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

} // namespace NHamKaas
