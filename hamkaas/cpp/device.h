#pragma once

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

namespace NHamKaas {

class IDevice
{
public:
    virtual ~IDevice() = default;

    virtual void CopyToDevice(void* dest, const void* src, int64_t size, bool sync = true) const = 0;
    virtual void CopyToHost(void* dest, const void* src, int64_t size, bool sync = true) const = 0;

    virtual void Synchronize() const = 0;

    virtual char* DeviceMalloc(int64_t size) const = 0;
    virtual void DeviceFree(char* ptr) const = 0;

    virtual char* HostMalloc(int64_t size) const = 0;
    virtual void HostFree(char* ptr) const = 0;
};

std::unique_ptr<IDevice> CreateCpuDevice();
std::unique_ptr<IDevice> CreateCudaDevice(cudaStream_t stream);

} // namespace NHamKaas
