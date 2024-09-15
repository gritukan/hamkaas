#include "device.h"

#include "error.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

class TCpuDevice
    : public IDevice
{
public:
    void CopyToDevice(void* dest, const void* src, int64_t size) const override
    {
        memcpy(dest, src, size);
    }

    void CopyToHost(void* dest, const void* src, int64_t size) const override
    {
        memcpy(dest, src, size);
    }

    char* DeviceMalloc(int64_t size) const override
    {
        return static_cast<char*>(malloc(size));
    }

    void DeviceFree(char* ptr) const override
    {
        free(ptr);
    }
};

std::unique_ptr<IDevice> CreateCpuDevice()
{
    return std::make_unique<TCpuDevice>();
}

class TCudaDevice
    : public IDevice
{
public:
    void CopyToDevice(void* dest, const void* src, int64_t size) const override
    {
        CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
    }

    void CopyToHost(void* dest, const void* src, int64_t size) const override
    {
        CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
    }

    char* DeviceMalloc(int64_t size) const override
    {
        char* ptr;
        auto error = cudaMalloc(&ptr, size);
        if (error != cudaSuccess) {
            return nullptr;
        }

        return ptr;
    }

    void DeviceFree(char* ptr) const override
    {
        CUDA_ASSERT(cudaFree(ptr));
    }
};

std::unique_ptr<IDevice> CreateCudaDevice()
{
    return std::make_unique<TCudaDevice>();
}
