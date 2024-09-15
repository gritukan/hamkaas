#include "common.h"

#include <iostream>
#include <vector>

constexpr int64_t K = 128;

__global__ void Kernel(float* a, float* b, float* c, int64_t n)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    float result = a[index];

    for (int i = 0; i < K; ++i) {
        if (b[index + i] == 1) {
            result += c[i];
        } else {
            result -= c[i];
        }
    }

    a[index] = result;
}

__global__ void KernelOpt(float* a, float* b, float* c, int64_t n)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    float result = a[index];

    for (int i = 0; i < K; ++i) {
        result += (2 * b[index + i] - 1) * c[i];
    }

    a[index] = result;
}

int main()
{
    constexpr int64_t N = 1 << 27;

    std::vector<float> dataA(N, 0);
    std::vector<float> dataB(N + K, 0);
    std::vector<float> dataC(K, 0);

    for (int64_t i = 0; i < N; ++i) {
        dataA[i] = i;
    }

    for (int64_t i = 0; i < N + K; ++i) {
        dataB[i] = (i < 1000) ? rand() % 2 : dataB[i - 1000];
    }

    for (int64_t i = 0; i < K; ++i) {
        dataC[i] = 3 * i + 17;
    }

    float* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, dataA.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float* gpuB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, (N + K) * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuB, dataB.data(), (N + K) * sizeof(float), cudaMemcpyHostToDevice));

    float* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, K * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuC, dataC.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int ThreadsPerBlock = 256;

    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        Kernel<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, gpuB, gpuC, N);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto time = timer.Stop();
        std::cout << "Kernel time=" << time << "ms" << std::endl;
    }

    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        KernelOpt<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, gpuB, gpuC, N);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto time = timer.Stop();
        std::cout << "KernelOpt time=" << time << "ms" << std::endl;
    }
}