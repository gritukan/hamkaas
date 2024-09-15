#include "common.h"

#include <cassert>
#include <vector>
#include <iostream>

__global__ void AddKernel(int64_t* a, int64_t* b, int64_t* c, int64_t n)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    c[index] = a[index] + b[index];
}

void DoStream(int64_t* a, int64_t* b, int64_t* c, int64_t n)
{
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    int64_t* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, n * sizeof(int64_t)));

    int64_t* gpuB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, n * sizeof(int64_t)));

    int64_t* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, n * sizeof(int64_t)));

    for (int i = 0; i < 5; i++) {
        TCudaEventTimer timer;
        timer.Start();

        CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a, n * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(gpuB, b, n * sizeof(int64_t), cudaMemcpyHostToDevice));

        constexpr int ThreadsPerBlock = 256;
        AddKernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, stream>>>(gpuA, gpuB, gpuC, n);

        CUDA_CHECK_ERROR(cudaMemcpy(c, gpuC, n * sizeof(int64_t), cudaMemcpyDeviceToHost));

        std::cout << "Stream: time=" << timer.Stop() << "ms" << std::endl;
    }

    for (int64_t i = 0; i < n; ++i) {
        assert(c[i] == a[i] + b[i]);
    }

    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));

    CUDA_CHECK_ERROR(cudaFree(gpuA));
    CUDA_CHECK_ERROR(cudaFree(gpuB));
    CUDA_CHECK_ERROR(cudaFree(gpuC));
}

void DoGraph(int64_t* a, int64_t* b, int64_t* c, int64_t n)
{
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    int64_t* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, n * sizeof(int64_t)));

    int64_t* gpuB;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuB, n * sizeof(int64_t)));

    int64_t* gpuC;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuC, n * sizeof(int64_t)));

    CUDA_CHECK_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    constexpr int ThreadsPerBlock = 256;
    AddKernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, stream>>>(gpuA, gpuB, gpuC, n);

    cudaGraph_t graph;
    CUDA_CHECK_ERROR(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK_ERROR(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    for (int i = 0; i < 5; i++) {
        TCudaEventTimer timer;
        timer.Start();

        CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a, n * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(gpuB, b, n * sizeof(int64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK_ERROR(cudaGraphLaunch(graphExec, stream));

        CUDA_CHECK_ERROR(cudaMemcpy(c, gpuC, n * sizeof(int64_t), cudaMemcpyDeviceToHost));

        std::cout << "Graph: time=" << timer.Stop() << "ms" << std::endl;
    }

    CUDA_CHECK_ERROR(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK_ERROR(cudaGraphDestroy(graph));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));

    CUDA_CHECK_ERROR(cudaFree(gpuA));
    CUDA_CHECK_ERROR(cudaFree(gpuB));
    CUDA_CHECK_ERROR(cudaFree(gpuC));
}

int main()
{
    constexpr int N = 1 << 25;
    std::vector<int64_t> a(N);
    std::vector<int64_t> b(N);
    std::vector<int64_t> c(N);
    for (int i = 0; i < N; ++i) {
        a[i] = i + 1;
        b[i] = 3 * i - 17;
    }

    DoStream(a.data(), b.data(), c.data(), N);
    DoGraph(a.data(), b.data(), c.data(), N);

    return 0;
}
