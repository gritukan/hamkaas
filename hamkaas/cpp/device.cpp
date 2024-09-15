#include "device.h"

#include "error.h"

#include <cstdlib>
#include <cstring>

namespace NHamKaas {

class TCpuDevice
    : public IDevice
{
public:
    void CopyToDevice(void* dest, const void* src, int64_t size, bool /*sync*/) const override
    {
        memcpy(dest, src, size);
    }

    void CopyToHost(void* dest, const void* src, int64_t size, bool /*sync*/) const override
    {
        memcpy(dest, src, size);
    }

    void Synchronize() const override
    { }

    char* DeviceMalloc(int64_t size) const override
    {
        return static_cast<char*>(malloc(size));
    }

    void DeviceFree(char* ptr) const override
    {
        free(ptr);
    }

    char* HostMalloc(int64_t size) const override
    {
        return static_cast<char*>(malloc(size));
    }

    void HostFree(char* ptr) const override
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
    explicit TCudaDevice(cudaStream_t stream)
        : Stream_(stream)
    { }

    void CopyToDevice(void* dest, const void* src, int64_t size, bool sync) const override
    {
        CUDA_CHECK_ERROR(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice));
        if (sync) {
            Synchronize();
        }
    }

    void CopyToHost(void* dest, const void* src, int64_t size, bool sync) const override
    {
        CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
        if (sync) {
            Synchronize();
        }
    }

    void Synchronize() const override
    {
        CUDA_ASSERT(cudaStreamSynchronize(Stream_));
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

    char* HostMalloc(int64_t size) const override
    {
        char* ptr;
        auto error = cudaMallocHost(&ptr, size);
        if (error != cudaSuccess) {
            return nullptr;
        }

        return ptr;
    }

    void HostFree(char* ptr) const override
    {
        CUDA_ASSERT(cudaFreeHost(ptr));
    }

private:
    const cudaStream_t Stream_;
};

std::unique_ptr<IDevice> CreateCudaDevice(cudaStream_t stream)
{
    return std::make_unique<TCudaDevice>(stream);
}

} // namespace NHamKaas
