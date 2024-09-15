#include "common.h"

#include <algorithm>
#include <iostream>
#include <vector>

__global__ void AddVectors(int n, double* a, double* b, double* c)
{
    // Write your code here.
}

std::vector<double> AddVectorsGpu(std::vector<double> a, std::vector<double> b)
{
    // CUDA does not like zero thread count.
    if (a.empty()) {
        return {};
    }

    double* gpuA;
    double* gpuB;
    double* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, a.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, b.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, a.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuB, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int ThreadsPerBlock = 8;
    int blocksPerGrid = (a.size() + ThreadsPerBlock - 1) / ThreadsPerBlock;
    AddVectors<<<blocksPerGrid, ThreadsPerBlock>>>(a.size(), gpuA, gpuB, gpuC);
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

    auto createTest = [] (int size) {
        std::vector<double> a(size);
        std::vector<double> b(size);
        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = 2 * i + 13;
        }
        return std::make_pair(a, b);
    };
    for (int size = 0; size < 30; ++size) {
        testValues.push_back(createTest(size));
    }
    for (int size : {(1 << 15) - 1, (1 << 15), (1 << 15) + 1, (1 << 20) - 1234, (1 << 20), (1 << 20) + 1234}) {
        testValues.push_back(createTest(size));
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
