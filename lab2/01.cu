#include "common.h"

#include <cassert>
#include <iostream>
#include <vector>

__global__ void AddVectorsKernel(int64_t* inA, int64_t* inB, int64_t* out, int64_t n)
{
    int64_t threadCount = gridDim.x * blockDim.x;
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t index = (n / threadCount) * threadIndex; index < min(n, (n / threadCount) * (threadIndex + 1)); ++index) {
        out[index] = inA[index] + inB[index];
    }
}

__global__ void AddVectorsKernelOpt(int64_t* inA, int64_t* inB, int64_t* out, int64_t n)
{
    int64_t threadCount = gridDim.x * blockDim.x;
    int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t index = (n / threadCount) * threadIndex; index < min(n, (n / threadCount) * (threadIndex + 1)); ++index) {
        out[index] = inA[index] + inB[index];
    }
}

int main()
{
    constexpr int N = 1 << 30;
    constexpr int K = 1 << 20;

    std::vector<int64_t> dataA(N, 0);
    std::vector<int64_t> dataB(N, 0);
    for (int64_t i = 0; i < N; ++i) {
        dataA[i] = i;
        dataB[i] = 2 * i + 17;
    }

    int64_t* gpuInA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuInA, N * sizeof(int64_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuInA, dataA.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t* gpuInB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuInB, N * sizeof(int64_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuInB, dataB.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t* gpuOut;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuOut, N * sizeof(int64_t)));

    constexpr int ThreadsPerBlock = 256;

    {
        for (int it = 0; it < 3; ++it) {
            TCudaEventTimer timer;
            timer.Start();

            AddVectorsKernel<<<(K + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuInA, gpuInB, gpuOut, N);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            std::cout << "Baseline: time = " << timer.Stop() << "ms" << std::endl;
        }

        std::vector<int64_t> out(N, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(out.data(), gpuOut, N * sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(out[i] == dataA[i] + dataB[i]);
        }
    }
    {
        for (int it = 0; it < 3; ++it) {
            TCudaEventTimer timer;
            timer.Start();

            AddVectorsKernelOpt<<<(K + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuInA, gpuInB, gpuOut, N);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            std::cout << "Optimized: time = " << timer.Stop() << "ms" << std::endl;
        }

        std::vector<int64_t> out(N, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(out.data(), gpuOut, N * sizeof(int64_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(out[i] == dataA[i] + dataB[i]);
        }
    }

    CUDA_CHECK_ERROR(cudaFree(gpuInA));
    CUDA_CHECK_ERROR(cudaFree(gpuInB));
    CUDA_CHECK_ERROR(cudaFree(gpuOut));
}
