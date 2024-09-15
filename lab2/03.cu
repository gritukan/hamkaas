#include "common.h"

#include <iostream>
#include <vector>

__global__ void Kernel(float* a, int64_t n)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    int x = 0;
    for (int i = 0; i < 100; i++) {
        if (index >= n) {
            continue;
        }

        x += 1;
    }

    if (index < n) {
        a[index] = x;
    }
}

int main()
{
    constexpr int64_t N1 = 1024;
    constexpr int64_t N2 = 1000;

    float* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, N1 * sizeof(float)));

    constexpr int ThreadsPerBlock = 256;

    for (int it = 0; it < 5; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        Kernel<<<(N1 + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, N1);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto time = timer.Stop();
        std::cout << "N1 time=" << time << "ms" << std::endl;
    }

    for (int it = 0; it < 5; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        Kernel<<<(N2 + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(gpuA, N2);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto time = timer.Stop();
        std::cout << "N2 time=" << time << "ms" << std::endl;
    }

    CUDA_CHECK_ERROR(cudaFree(gpuA));
}
