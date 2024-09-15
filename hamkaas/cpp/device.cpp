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
    explicit TCudaDevice(cudaStream_t stream)
        : Stream_(stream)
    { }

    void CopyToDevice(void* dest, const void* src, int64_t size, bool sync) const override
    {
        // (lab4/02): Your code here: copy memory to device.
    }

    void CopyToHost(void* dest, const void* src, int64_t size, bool sync) const override
    {
        // (lab4/02): Your code here: copy memory to host.
    }

    char* DeviceMalloc(int64_t size) const override
    {
        // (lab4/02): Your code here: allocate device memory.
        return nullptr;
    }

    void DeviceFree(char* ptr) const override
    {
        // (lab4/02): Your code here: free device memory.
    }

private:
    const cudaStream_t Stream_;
};

std::unique_ptr<IDevice> CreateCudaDevice(cudaStream_t stream)
{
    return std::make_unique<TCudaDevice>(stream);
}

} // namespace NHamKaas
