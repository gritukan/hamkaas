#include "common.h"

#include <cassert>
#include <vector>
#include <iostream>

__global__ void Kernel(int64_t* a, int i)
{
    a[i] = a[2 * i] + a[2 * i + 1];
}

void DoSlow(int64_t* a, int n)
{
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    CUDA_CHECK_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int i = n - 1; i >= 1; --i) {
        Kernel<<<1, 1, 0, stream>>>(a, i);
    }

    cudaGraph_t graph;
    CUDA_CHECK_ERROR(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK_ERROR(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    for (int i = 0; i < 5; i++) {
        TCudaEventTimer timer;
        timer.Start();

        CUDA_CHECK_ERROR(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        std::cout << "Slow: time=" << timer.Stop() << "ms" << std::endl;

        int64_t result;
        CUDA_CHECK_ERROR(cudaMemcpy(&result, a + 1, sizeof(int64_t), cudaMemcpyDeviceToHost));
        assert(result == int64_t(n) * (n + 1) / 2);
    }

    CUDA_CHECK_ERROR(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK_ERROR(cudaGraphDestroy(graph));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}

void DoFast(int64_t* a, int n)
{
    cudaGraph_t graph;
    CUDA_CHECK_ERROR(cudaGraphCreate(&graph, 0));

    // Your code here: add nodes to the graph.

    cudaGraphExec_t graphExec;
    CUDA_CHECK_ERROR(cudaGraphInstantiate(&graphExec, graph, 0));

    for (int i = 0; i < 5; ++i) {
        TCudaEventTimer timer;
        timer.Start();

        CUDA_CHECK_ERROR(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        std::cout << "Fast: time=" << timer.Stop() << "ms" << std::endl;

        int64_t result;
        CUDA_CHECK_ERROR(cudaMemcpy(&result, a + 1, sizeof(int64_t), cudaMemcpyDeviceToHost));
        assert(result == int64_t(n) * (n + 1) / 2);
    }

    CUDA_CHECK_ERROR(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK_ERROR(cudaGraphDestroy(graph));
}

int main()
{
    constexpr int N = 1 << 17;
    std::vector<int64_t> a(N);
    for (int i = 0; i < N; ++i) {
        a[i] = i + 1;
    }

    int64_t* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, 2 * N * sizeof(int64_t)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA + N, a.data(), N * sizeof(int64_t), cudaMemcpyHostToDevice));

    DoSlow(gpuA, N);

    // Clear result.
    constexpr int64_t Zero = 0;
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA + 1, &Zero, sizeof(int64_t), cudaMemcpyHostToDevice));

    // DoFast(gpuA, N);

    return 0;
}
