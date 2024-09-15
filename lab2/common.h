#pragma once

#include <cstdio>

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }

class TCudaEventTimer
{
public:
    void Start()
    {
        CUDA_CHECK_ERROR(cudaEventCreate(&Start_));
        CUDA_CHECK_ERROR(cudaEventCreate(&Stop_));
        CUDA_CHECK_ERROR(cudaEventRecord(Start_));
    }

    float Stop()
    {
        CUDA_CHECK_ERROR(cudaEventRecord(Stop_));
        CUDA_CHECK_ERROR(cudaEventSynchronize(Stop_));

        float duration = 0;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&duration, Start_, Stop_));
        return duration;
    }

private:
    cudaEvent_t Start_;
    cudaEvent_t Stop_;
};