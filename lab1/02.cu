#include "common.h"

#include <iostream>
#include <vector>

__global__ void AddMatricesKernel(double *a, double *b, double *c, int m)
{
    int x = threadIdx.x;
    int y = threadIdx.y;

    c[y * m + x] = a[y * m + x] + b[y * m + x];
}

std::vector<std::vector<double>> AddMatricesGpu(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b)
{
    int n = a.size();
    int m = a[0].size();

    double* gpuA;
    double* gpuB;
    double* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, n * m * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, n * m * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, n * m * sizeof(double)));
    for (int i = 0; i < n; i++) {
        CUDA_CHECK_ERROR(cudaMemcpy(gpuA + (i * m), a[i].data(), m * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(gpuB + (i * m), b[i].data(), m * sizeof(double), cudaMemcpyHostToDevice));
    }

    dim3 threadsPerBlock(n, m);
    AddMatricesKernel<<<1, n * m>>>(gpuA, gpuB, gpuC, m);
    CUDA_CHECK_KERNEL();

    std::vector<std::vector<double>> c(n, std::vector<double>(m));
    for (int i = 0; i < n; i++) {
        CUDA_CHECK_ERROR(cudaMemcpy(c[i].data(), gpuC + (i * m), m * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_ERROR(cudaFree(gpuA));
    CUDA_CHECK_ERROR(cudaFree(gpuB));
    CUDA_CHECK_ERROR(cudaFree(gpuC));

    return c;
}

std::vector<std::vector<double>> AddMatricesCpu(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b)
{
    int n = a.size();
    int m = a[0].size();

    std::vector<std::vector<double>> c(n, std::vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    return c;
}

bool DoTest(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b)
{
    auto gpuResult = AddMatricesGpu(a, b);
    auto cpuResult = AddMatricesCpu(a, b);

    int n = a.size();
    int m = a[0].size();

    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (n = " << n << ", m = " << m << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (n = " << n << ", m = " << m << ")" << std::endl;
        return false;
    }
}

int main()
{
    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>> testValues = {
        {{{1.0}}, {{2.0}}},
        {{{1.0, -1.0}, {-1.0, 1.0}}, {{1.0, -1.0}, {1.0, -1.0}}},
    };

    for (auto [x, y] : std::vector<std::pair<int, int>>({{1, 10}, {10, 1}, {5, 5}, {7, 8}, {10, 10}})) {
        std::vector<std::vector<double>> a(x, std::vector<double>(y));
        std::vector<std::vector<double>> b(x, std::vector<double>(y));
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                a[i][j] = i + j;
                b[i][j] = 2 * i + 13;
            }
        }

        testValues.push_back({a, b});
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
