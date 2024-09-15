#include "bootstrap.h"

#include "error.h"

namespace NHamKaas {

TBootstrap::TBootstrap()
{
    CUBLAS_CHECK_ERROR(cublasCreate(&CublasHandle_));

#ifdef USE_CUDNN
    CUDNN_CHECK_ERROR(cudnnCreate(&CudnnHandle_));
#endif
}

TBootstrap::~TBootstrap()
{
    CUBLAS_ASSERT(cublasDestroy(CublasHandle_));

#ifdef USE_CUDNN
    CUDNN_ASSERT(cudnnDestroy(CudnnHandle_));
#endif
}

cublasHandle_t TBootstrap::GetCublasHandle() const
{
    return CublasHandle_;
}

#ifdef USE_CUDNN
cudnnHandle_t TBootstrap::GetCudnnHandle() const
{
    return CudnnHandle_;
}
#endif

} // namespace NHamKaas
