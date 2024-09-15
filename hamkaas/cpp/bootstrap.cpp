#include "bootstrap.h"

#include "error.h"

namespace NHamKaas {

TBootstrap::TBootstrap()
{
    CUBLAS_CHECK_ERROR(cublasCreate(&CublasHandle_));
}

TBootstrap::~TBootstrap()
{
    CUBLAS_ASSERT(cublasDestroy(CublasHandle_));
}

cublasHandle_t TBootstrap::GetCublasHandle() const
{
    return CublasHandle_;
}

} // namespace NHamKaas
