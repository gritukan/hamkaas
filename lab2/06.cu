#include "common.h"

#include <cassert>
#include <vector>
#include <iostream>

__global__ void BiasKernel(float* a, float* w, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    a[index] += w[index];
}

__global__ void SiLUKernel(float* a, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    a[index] = a[index] / (1.0 + exp(-a[index]));
}

__global__ void FusedKernel(float* a, float* w, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    float x = a[index] + w[index];
    a[index] = x / (1.0 + exp(-x));
}

void DoGraph(float* a, float* w, int n)
{
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    CUDA_CHECK_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    constexpr int ThreadsPerBlock = 256;
    BiasKernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, stream>>>(a, w, n);
    SiLUKernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock, 0, stream>>>(a, n);

    cudaGraph_t graph;
    CUDA_CHECK_ERROR(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK_ERROR(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    for (int i = 0; i < 3; i++) {
        TCudaEventTimer timer;
        timer.Start();

        CUDA_CHECK_ERROR(cudaGraphLaunch(graphExec, 0));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        std::cout << "Graph: time=" << timer.Stop() << "ms" << std::endl;
    }
}

void DoFused(float* a, float* w, int n)
{
    for (int it = 0; it < 3; ++it) {
        TCudaEventTimer timer;
        timer.Start();

        constexpr int ThreadsPerBlock = 256;
        FusedKernel<<<(n + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(a, w, n);

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        std::cout << "Fused: time=" << timer.Stop() << "ms" << std::endl;
    }
}

int main()
{
    constexpr int N = 1 << 30;

    std::vector<float> a(N), w(N);
    for (int i = 0; i < N; i++) {
        a[i] = 1.0 * i / 1e6;
        w[i] = 1.0 * i / 1e6;
    }

    float* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, a.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float* gpuW;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuW, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuW, w.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    DoGraph(gpuA, gpuW, N);
    DoFused(gpuA, gpuW, N);
}
