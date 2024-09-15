#include "common.h"

#include <iostream>
#include <vector>

__global__ void DoSomethingKernel(double* ptr)
{
    double* x = ptr + blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 1e6; i++) {
        *x += 0.2;
        *x = cos(*x);
    }
}

int main()
{
    constexpr int N = 1000;

    for (int i = 0; i < 10; i++) {
        std::vector<double> data(N, 0);
        for (auto& x : data) {
            for (int j = 0; j < 1000; j++) {
                x += 0.3;
                x = cos(x);
            }
        }

        double* gpuData;
        CUDA_CHECK_ERROR(cudaMalloc(&gpuData, N * sizeof(double)));
        CUDA_CHECK_ERROR(cudaMemcpy(gpuData, data.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
        DoSomethingKernel<<<N, 1>>>(gpuData);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK_ERROR(cudaMemcpy(data.data(), gpuData, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaFree(gpuData));
    }
}
