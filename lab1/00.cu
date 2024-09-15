#include "common.h"

#include <iostream>
#include <vector>

__global__ void AddOneKernel(double* ptr)
{
    *ptr += 1;
}

double AddOneGpu(double x)
{
    double* gpuData;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuData, sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuData, &x, sizeof(double), cudaMemcpyHostToDevice));

    AddOneKernel<<<1, 1>>>(gpuData);

    double result;
    CUDA_CHECK_ERROR(cudaMemcpy(&result, gpuData, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(gpuData));

    return result;
}

double AddOneCpu(double x)
{
    return x + 1;
}

bool DoTest(double x)
{
    auto gpuResult = AddOneGpu(x);
    auto cpuResult = AddOneCpu(x);
    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (x = " << x << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (x = " << x << ")" << std::endl;
        std::cerr << "GPU: " << gpuResult << std::endl;
        std::cerr << "CPU: " << cpuResult << std::endl;
        return false;
    }
}

int main()
{
    std::vector<double> testValues = {-42, 0, 3.14, 1e100};
    int passedTests = 0;
    for (auto x : testValues) {
        if (DoTest(x)) {
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
