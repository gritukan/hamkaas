#pragma once

#include <cublas_v2.h>

namespace NHamKaas {

class TBootstrap
{
public:
    TBootstrap();
    ~TBootstrap();

    cublasHandle_t GetCublasHandle() const;

private:
    cublasHandle_t CublasHandle_;
};

} // namespace NHamKaas
