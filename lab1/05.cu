#include "common.h"

#include <iostream>
#include <vector>

__global__ void SiluKernel(double* a, int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        double x = a[index];
        a[index] = x / (1 + exp(-x));
    }
}

std::vector<double> SiluGpu(std::vector<double> data)
{
    double* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, data.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int MaxThreadsPerBlock = 256;
    int size = data.size();
    int threadsPerBlock = std::min(size, MaxThreadsPerBlock);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    SiluKernel<<<blocksPerGrid, threadsPerBlock>>>(gpuA, size);
    CUDA_CHECK_KERNEL();

    std::vector<double> result(data.size());
    CUDA_CHECK_ERROR(cudaMemcpy(result.data(), gpuA, data.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(gpuA));

    return result;
}

std::vector<double> SiluCpu(std::vector<double> data)
{
    for (int i = 0; i < data.size(); i++) {
        double x = data[i];
        data[i] = x / (1 + exp(-x));
    }

    return data;
}

bool DoTest(std::vector<double> data)
{
    auto gpuResult = SiluGpu(data);
    auto cpuResult = SiluCpu(data);

    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (n = " << data.size() << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (n = " << data.size() << ")" << std::endl;
        return false;
    }
}

int main()
{
    std::vector<std::vector<double>> testValues;

    for (int size : {1, 2, 3, 100, 1000, 10000}) {
        std::vector<double> a(size);
        for (int i = 0; i < size; i++) {
            if (i % 2 == 0) {
                a[i] = i / 1000;
            } else {
                a[i] = -i / 1000;
            }
        }
        testValues.push_back(a);
    }

    int passedTests = 0;
    for (auto data : testValues) {
        if (DoTest(data)) {
            passedTests++;
        }
    }

    if (passedTests == testValues.size()) {
        std::cerr << "All tests passed" << std::endl;
        return 0;
    } else {
        std::cerr << "Some tests failed" << std::endl;
        return 1;
    }
}
