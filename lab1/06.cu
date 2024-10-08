#include "common.h"

#include <algorithm>
#include <iostream>
#include <vector>

template <int KernelSize, int BlockDimensionSize>
__global__ void MaxPoolingKernel(float* input, float* output, int n, int m)
{
    // Write your code here.
}

std::vector<std::vector<float>> MaxPoolingGpu(std::vector<std::vector<float>> input)
{
    int n = input.size();
    int m = input[0].size();

    float* gpuInput;
    float* gpuOutput;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuInput, n * m * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&gpuOutput, n * m * sizeof(float)));

    for (int i = 0; i < n; i++) {
        CUDA_CHECK_ERROR(cudaMemcpy(gpuInput + i * m, input[i].data(), m * sizeof(float), cudaMemcpyHostToDevice));
    }

    constexpr int KernelSize = 4;
    constexpr int BlockDimensionSize = 16;

    dim3 blocks((n + BlockDimensionSize - 1) / BlockDimensionSize, (m + BlockDimensionSize - 1) / BlockDimensionSize);
    dim3 threads(BlockDimensionSize, BlockDimensionSize);
    MaxPoolingKernel<KernelSize, BlockDimensionSize><<<blocks, threads>>>(gpuInput, gpuOutput, n, m);

    std::vector<std::vector<float>> output(n, std::vector<float>(m));
    for (int i = 0; i < n; i++) {
        CUDA_CHECK_ERROR(cudaMemcpy(output[i].data(), gpuOutput + i * m, m * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_ERROR(cudaFree(gpuInput));
    CUDA_CHECK_ERROR(cudaFree(gpuOutput));

    return output;
}

std::vector<std::vector<float>> MaxPoolingCpu(std::vector<std::vector<float>> data)
{
    int n = data.size();
    int m = data[0].size();

    std::vector<std::vector<float>> result(n, std::vector<float>(m));
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < m; ++y) {
            float& max = result[x][y];
            for (int dx = 0; dx < 4; dx++) {
                for (int dy = 0; dy < 4; dy++) {
                    if (x + dx < n && y + dy < m) {
                        max = std::max(max, data[x + dx][y + dy]);
                    }
                }
            }
        }
    }

    return result;
}

bool DoTest(std::vector<std::vector<float>> data)
{
    auto gpuResult = MaxPoolingGpu(data);
    auto cpuResult = MaxPoolingCpu(data);
    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (n = " << data.size() << ", m = " << data[0].size() << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (n = " << data.size() << ", m = " << data[0].size() << ")" << std::endl;
        return false;
    }
}

int main()
{
    std::vector<std::vector<std::vector<float>>> testValues;
    auto addTest = [&] (int n, int m) {
        std::vector<std::vector<float>> data(n, std::vector<float>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = 1.0 * rand() / RAND_MAX;
            }
        }

        testValues.push_back(data);
    };
    for (int x = 1; x <= 10; x++) {
        for (int y = 1; y <= 10; y++) {
            addTest(x, y);
        }
    }

    for (int x = 1; x <= 5; x++) {
        addTest(x, 1000);
        addTest(x, 999);
        addTest(x, 1001);
        addTest(1000, x);
        addTest(999, x);
        addTest(1001, x);
    }

    addTest(1000, 1000);
    addTest(999, 1000);
    addTest(1001, 1000);
    addTest(1000, 999);
    addTest(1000, 1000);
    addTest(1000, 1001);

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
