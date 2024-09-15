#include "common.h"

#include <iostream>
#include <vector>

__global__ void AddVectorsKernel(double* a, double* b, double* c)
{
    // Write your code here.
// SOLUTION START
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
// SOLUTION END
}

std::vector<double> AddVectorsGpu(std::vector<double> a, std::vector<double> b)
{
    double* gpuA;
    double* gpuB;
    double* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, a.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, b.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, a.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuB, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));

    AddVectorsKernel<<<1, a.size()>>>(gpuA, gpuB, gpuC);
    CUDA_CHECK_KERNEL();

    std::vector<double> c(a.size());
    CUDA_CHECK_ERROR(cudaMemcpy(c.data(), gpuC, a.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(gpuA));
    CUDA_CHECK_ERROR(cudaFree(gpuB));
    CUDA_CHECK_ERROR(cudaFree(gpuC));

    return c;
}

std::vector<double> AddVectorsCpu(std::vector<double> a, std::vector<double> b)
{
    std::vector<double> c(a.size());
    for (int i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

bool DoTest(std::vector<double> a, std::vector<double> b)
{
    auto gpuResult = AddVectorsGpu(a, b);
    auto cpuResult = AddVectorsCpu(a, b);
    for (int i = 0; i < a.size(); i++) {
        if (gpuResult[i] != cpuResult[i]) {
            std::cerr << "Test failed (n = " << a.size() << ")" << std::endl;
            std::cerr << "GPU: " << gpuResult[i] << std::endl;
            std::cerr << "CPU: " << cpuResult[i] << std::endl;
            std::cerr << "Index: " << i << std::endl;

            return false;
        }
    }

    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (n = " << a.size() << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (n = " << a.size() << ")" << std::endl;
        return false;
    }
}

int main()
{
    std::vector<std::pair<std::vector<double>, std::vector<double>>> testValues = {
        {{1.0}, {2.0}},
        {{1.0, -1.0}, {-1.0, 1.0}},
        {{3.14}, {1.59}},
        {{1e100, 1e-100}, {1e-100, 1e100}},
    };

    {
        std::vector<double> bigA(1 << 9);
        std::vector<double> bigB(1 << 9);
        for (int i = 0; i < bigA.size(); i++) {
            bigA[i] = i;
            bigB[i] = 2 * i + 13;
        }
        testValues.push_back({bigA, bigB});
    }

    int passedTests = 0;
    for (const auto& [a, b] : testValues) {
        if (DoTest(a, b)) {
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
