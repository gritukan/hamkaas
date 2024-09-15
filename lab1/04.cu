#include "common.h"

#include <algorithm>
#include <iostream>
#include <vector>

__global__ void SwapAdjacentKernel(double* data)
{
    __shared__ double buffer[2];

    // Your code here.
}

std::vector<double> SwapAdjacentGpu(std::vector<double> data)
{
    // CUDA does not like zero block count.
    if (data.size() == 1) {
        return data;
    }

    double* gpuData;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuData, data.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuData, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));

    SwapAdjacentKernel<<<data.size() / 2, 2>>>(gpuData);

    std::vector<double> result(data.size());
    CUDA_CHECK_ERROR(cudaMemcpy(result.data(), gpuData, result.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(gpuData));

    return result;
}

std::vector<double> SwapAdjacentCpu(std::vector<double> data)
{
    for (int i = 0; i + 1 < data.size(); i += 2) {
        std::swap(data[i], data[i + 1]);
    }

    return data;
}

bool DoTest(std::vector<double> data)
{
    auto gpuResult = SwapAdjacentGpu(data);
    auto cpuResult = SwapAdjacentCpu(data);
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
    auto addTest = [&] (int size) {
        std::vector<double> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        testValues.push_back(data);
    };
    for (int size = 1; size <= 10; size++) {
        addTest(size);
    }
    for (int size : {(1 << 15) - 17, (1 << 15), (1 << 15) + 1, (1 << 20) - 1, (1 << 20), (1 << 20) + 1}) {
        addTest(size);
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
