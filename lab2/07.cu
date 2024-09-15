#include "common.h"

#include <cassert>
#include <vector>
#include <iostream>

__global__ void Do(int* a, int* b, int* c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    c[index] = 3 * a[index] + b[index];
}

__global__ void DoPtx(int* a, int* b, int* c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    asm("Your Code Here"
        : "=r"(c[index])
        : "r"(a[index])
        , "r"(b[index]));
}

int main()
{
    constexpr int N = 1 << 20;
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);

    for (int i = 0; i < N; ++i) {
        a[i] = 3 * i - 7;
        b[i] = i - 19;
    }

    int* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, N * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int* gpuB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, N * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuB, b.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, N * sizeof(int)));

    {
        constexpr int ThreadsPerBlock = 256;
        Do<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, gpuB, gpuC, N);

        CUDA_CHECK_ERROR(cudaMemcpy(c.data(), gpuC, N * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(c[i] == 3 * a[i] + b[i]);
        }
    }

    // Clear output buffer.
    CUDA_CHECK_ERROR(cudaMemset(gpuC, 0, N * sizeof(int)));

    {
        constexpr int ThreadsPerBlock = 256;
        DoPtx<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, gpuB, gpuC, N);

        CUDA_CHECK_ERROR(cudaMemcpy(c.data(), gpuC, N * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            assert(c[i] == 3 * a[i] + b[i]);
        }
    }

    CUDA_CHECK_ERROR(cudaFree(gpuA));
    CUDA_CHECK_ERROR(cudaFree(gpuB));
    CUDA_CHECK_ERROR(cudaFree(gpuC));
}
