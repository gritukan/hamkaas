#include "common.h"

#include <cassert>
#include <iostream>
#include <vector>

constexpr int SIZE = 32768;

// Each block processes a submatrix of size BLOCK_SIZE x BLOCK_SIZE.
constexpr int BLOCK_SIZE = 32;
// Each thread processes THREAD_SIZE elements of a input matrix row.
constexpr int THREAD_SIZE = 4;


__global__ void TransposeMatrixKernel1(double* in, double* out)
{
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int baseY = blockIdx.y * BLOCK_SIZE + threadIdx.y * THREAD_SIZE;

    for (int index = 0; index < THREAD_SIZE; ++index) {
        int y = baseY + index;
        out[y * SIZE + x] = 2 * in[x * SIZE + y];
    }
}

__global__ void TransposeMatrixKernel2(double* in, double* out)
{
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    //int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    //int baseIn = x * SIZE + y;
    //int baseOut = y * SIZE + x;
/*
    out[baseOut] = in[baseIn];
    out[baseOut + 131072] = in[baseIn + 4];
    out[baseOut + 262144] = in[baseIn + 8];
    out[baseOut + 393216] = in[baseIn + 12];
    out[baseOut + 524288] = in[baseIn + 16];
    out[baseOut + 655360] = in[baseIn + 20];
    out[baseOut + 786432] = in[baseIn + 24];
    out[baseOut + 917504] = in[baseIn + 28];
*/
    int baseY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    constexpr int OFFSET = BLOCK_SIZE / THREAD_SIZE;
    for (int index = 0; index < BLOCK_SIZE; index += OFFSET) {
        int y = baseY + index;
        out[y * SIZE + x] = 2 * in[x * SIZE + y];
    }
}

__global__ void TransposeMatrixKernel3(double* in, double* out)
{
    int baseX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int baseY = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    __shared__ double shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    constexpr int OFFSET = BLOCK_SIZE / THREAD_SIZE;
    for (int index = 0; index < BLOCK_SIZE; index += OFFSET) {
        int y = baseY + index;
        shared[threadIdx.x][threadIdx.y + index] = 2 * in[baseX * SIZE + y];
    }

    __syncthreads();

    baseX = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    baseY = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    for (int index = 0; index < BLOCK_SIZE; index += OFFSET) {
        int y = baseY + index;
        out[baseX * SIZE + y] = shared[threadIdx.y + index][threadIdx.x];
    }
}

int main()
{
    std::vector<std::vector<double>> in(SIZE, std::vector<double>(SIZE, 0));
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            in[i][j] = i * SIZE + j;
        }
    }

    double* gpuIn;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuIn, SIZE * SIZE * sizeof(double)));
    for (int i = 0; i < SIZE; ++i) {
        CUDA_CHECK_ERROR(cudaMemcpy(gpuIn + i * SIZE, in[i].data(), SIZE * sizeof(double), cudaMemcpyHostToDevice));
    }

    double* gpuOut;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuOut, SIZE * SIZE * sizeof(double)));

    for (int i = 0; i < 10; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK_ERROR(cudaEventCreate(&start));
        CUDA_CHECK_ERROR(cudaEventCreate(&stop));
        
        CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

        dim3 blocks(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE);
        TransposeMatrixKernel1<<<blocks, threads>>>(gpuIn, gpuOut);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

        float duration = 0;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&duration, start, stop));
        duration /= 1000.0;

        std::cerr << "Duration: " << duration << "s, " << sizeof(double) * SIZE * SIZE * 2 / duration / 1024 / 1024 / 1024 / 1024 << "TB/s" << std::endl;
    }

    for (int i = 0; i < 10; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK_ERROR(cudaEventCreate(&start));
        CUDA_CHECK_ERROR(cudaEventCreate(&stop));
        
        CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

        dim3 blocks(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE);
        TransposeMatrixKernel2<<<blocks, threads>>>(gpuIn, gpuOut);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

        float duration = 0;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&duration, start, stop));
        duration /= 1000.0;

        std::cerr << "Duration: " << duration << "s, " << sizeof(double) * SIZE * SIZE * 2 / duration / 1024 / 1024 / 1024 / 1024 << "TB/s" << std::endl;
    }

    for (int i = 0; i < 10; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK_ERROR(cudaEventCreate(&start));
        CUDA_CHECK_ERROR(cudaEventCreate(&stop));
        
        CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

        dim3 blocks(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE);
        TransposeMatrixKernel3<<<blocks, threads>>>(gpuIn, gpuOut);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

        float duration = 0;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&duration, start, stop));
        duration /= 1000.0;

        std::cerr << "Duration: " << duration << "s, " << sizeof(double) * SIZE * SIZE * 2 / duration / 1024 / 1024 / 1024 / 1024 << "TB/s" << std::endl;
    }

    std::vector<std::vector<double>> out(SIZE, std::vector<double>(SIZE, 0));
    for (int i = 0; i < SIZE; ++i) {
        CUDA_CHECK_ERROR(cudaMemcpy(out[i].data(), gpuOut + i * SIZE, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            assert(out[i][j] == 2 * in[j][i]);
        }
    }
}
