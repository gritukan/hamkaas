#include "common.h"

#include <cassert>
#include <iostream>
#include <vector>

__global__ void AddVectorsKernel(double* inA, double* inB, double* out, int n, int k)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= k) {
        return;
    }

    for (int index = (n / k) * threadIndex; index < min(n, (n / k) * (threadIndex + 1)); ++index) {
        out[index] = inA[index] + inB[index];
    }
}

__global__ void AddVectorsKernelOpt(double* inA, double* inB, double* out, int n, int k)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= k) {
        return;
    }

    for (int index = threadIndex; index < n; index += k) {
        out[index] = inA[index] + inB[index];
    }
}

int main()
{
    constexpr int N = 1e9;
    constexpr int K = 1e6;

    std::vector<double> dataA(N, 0);
    std::vector<double> dataB(N, 0);
    for (int i = 0; i < N; ++i) {
        dataA[i] = i;
        dataB[i] = 2 * i + 17;
    }

    double* gpuInA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuInA, N * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuInA, dataA.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    double* gpuInB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuInB, N * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuInB, dataB.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    double* gpuOut;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuOut, N * sizeof(double)));

    constexpr int ThreadsPerBlock = 256;

    {
        AddVectorsKernel<<<(K + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuInA, gpuInB, gpuOut, N, K);
        CUDA_CHECK_KERNEL();

        std::vector<double> out(N, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(out.data(), gpuOut, N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(out[i] == dataA[i] + dataB[i]);
        }
    }
    {
        AddVectorsKernelOpt<<<(K + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuInA, gpuInB, gpuOut, N, K);
        CUDA_CHECK_KERNEL();

        std::vector<double> out(N, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(out.data(), gpuOut, N * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(out[i] == dataA[i] + dataB[i]);
        }
    }
}
