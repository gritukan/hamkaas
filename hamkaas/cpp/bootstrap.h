#pragma once

#include <cublas_v2.h>

#ifdef USE_CUDNN
#include "cudnn-frontend/include/cudnn_frontend.h"
#endif

namespace NHamKaas {

class TBootstrap
{
public:
    TBootstrap();
    ~TBootstrap();

    cublasHandle_t GetCublasHandle() const;

#ifdef USE_CUDNN
    cudnnHandle_t GetCudnnHandle() const;
#endif

private:
    cublasHandle_t CublasHandle_;

#ifdef USE_CUDNN
    cudnnHandle_t CudnnHandle_;
#endif
};

} // namespace NHamKaas
