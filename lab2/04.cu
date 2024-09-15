#include "common.h"

#include <cassert>
#include <vector>
#include <iostream>

__global__ void Kernel(float* a, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    a[index] += 1.0;
}

int TOTAL_RUNS = 0;

void DoSlow(float* a, int n)
{
    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        for (int i = 0; i < 100; ++i) {
            constexpr int ThreadsPerBlock = 256;
            Kernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(a, n);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        }

        ++TOTAL_RUNS;

        std::cout << "Slow: time=" << timer.Stop() << "ms" << std::endl;

        std::vector<float> data(n, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(data.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            assert(fabs(data[i] - (1.0 * i / 1e6 + TOTAL_RUNS * 100.0)) < 1e-2);
        }
    }
}

void DoStream(float* a, int n)
{
    // Your code here: create a stream.

    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        // Your code here: launch kernels with the stream.

        ++TOTAL_RUNS;

        std::cout << "Stream: time=" << timer.Stop() << "ms" << std::endl;

        std::vector<float> data(n, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(data.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            assert(fabs(data[i] - (1.0 * i / 1e6 + TOTAL_RUNS * 100.0)) < 1e-2);
        }
    }

    // Your code here: cleanup.
}

void DoGraph(float* a, int n)
{
    // Your code here: capture the stream and compile the graph.

    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        // Your code here: launch the graph.

        ++TOTAL_RUNS;

        std::cout << "Graph: time=" << timer.Stop() << "ms" << std::endl;

        std::vector<float> data(n, 0);
        CUDA_CHECK_ERROR(cudaMemcpy(data.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            assert(fabs(data[i] - (1.0 * i / 1e6 + TOTAL_RUNS * 100.0)) < 1e-2);
        }
    }
}

int main()
{
    constexpr int N = 1 << 17;

    std::vector<float> data(N, 0);
    for (int i = 0; i < N; ++i) {
        data[i] = 1.0 * i / 1e6;
    }

    float* gpuData;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuData, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuData, data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    DoSlow(gpuData, N);
    // DoStream(gpuData, N);
    // DoGraph(gpuData, N);

    CUDA_CHECK_ERROR(cudaFree(gpuData));
}
