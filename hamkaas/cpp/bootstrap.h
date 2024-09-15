#pragma once

#include <cublas_v2.h>

class TBootstrap
{
public:
    TBootstrap();
    ~TBootstrap();

    cublasHandle_t GetCublasHandle() const;

private:
    cublasHandle_t CublasHandle_;
};
