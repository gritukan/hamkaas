#include "common.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Implements C := A * B^T + C with A being a matrix of size N x M, B being a matrix of size K x M and C being a matrix of size N x K.
// The intended usage is to evaluate a Ax + b linear transformation for a batch of x vectors.
// In this case A is a matrix of N input vectors of size M where M is an input dimension.
// B^T is a transposed matrix of the weigths with K being the output dimension and M be input dimension.
// C is initially filled with the bias vector in each row.
template <int N, int M, int K>
__global__ void LinearLayerKernel(const TFloatType* A, const TFloatType* B, TFloatType* C)
{
    // Your code here.
}

// Applies the ReLU activation function to the batch of B vectors of size N.
template <int N, int B>
__global__ void ReLUKernel(TFloatType* a)
{
    // Your code here.
}

// Computes the argmax of each of the B vectors of size N.
template <int N, int B>
__global__ void ArgMaxKernel(const TFloatType* a, int* result)
{
    // Your code here.
}

template <size_t InputDimension, size_t OutputDimension, size_t BatchSize>
class TGpuLinearLayer
{
public:
    explicit TGpuLinearLayer(const TLinearLayer<TFloatType, InputDimension, OutputDimension>& layer, cublasHandle_t cublasHandle)
        : CublasHandle_(cublasHandle)
    {
        Weights_.ToDevice(layer.Weights);

        TMatrix<TFloatType, BatchSize, OutputDimension> biases;
        for (size_t i = 0; i < BatchSize; i++) {
            biases[i] = layer.Biases;
        }
        Biases_.ToDevice(biases);
    }

    // NB: Does not wait for the kernel completion.
    void Apply(const TGpuMatrix<TFloatType, BatchSize, InputDimension>& input, TGpuMatrix<TFloatType, BatchSize, OutputDimension>& output)
    {
        // Initialize the output with the biases.
        CUDA_CHECK_ERROR(cudaMemcpy(output.Data(), Biases_.Data(), BatchSize * OutputDimension * sizeof(TFloatType), cudaMemcpyDeviceToDevice));

#ifdef USE_CUBLAS
        // Your code here.
#else
        // Your code here.
#endif
    }

private:
    const cublasHandle_t CublasHandle_;

    TGpuMatrix<TFloatType, OutputDimension, InputDimension> Weights_;
    TGpuMatrix<TFloatType, BatchSize, OutputDimension> Biases_;
};

template <size_t Size, size_t BatchSize>
class TGpuReLULayer
{
public:
    // NB: Does not wait for the kernel completion.
    void Apply(TGpuMatrix<TFloatType, BatchSize, Size>& input)
    {
        // Your code here.
    }
};

template <size_t Size, size_t BatchSize>
class TGpuArgMaxLayer
{
public:
    // NB: Does not wait for the kernel completion.
    void Apply(const TGpuMatrix<TFloatType, BatchSize, Size>& input, TGpuVector<int, BatchSize>& result)
    {
        // Your code here.
    }
};

template <size_t BatchSize>
struct TGpuMNISTNetwork
{
public:
    explicit TGpuMNISTNetwork(const TMNISTNetwork& network, cublasHandle_t cublasHandle)
        : L1_(network.L1, cublasHandle)
        , L2_(network.L2, cublasHandle)
    {
        L1Output_.Allocate();
        L2Output_.Allocate();
        Result_.Allocate();
    }

    std::array<int, BatchSize> Eval(const TImageBatch<BatchSize>& input)
    {
        TMatrix<TFloatType, BatchSize, ImageSize * ImageSize> inputMatrix;
        for (size_t i = 0; i < BatchSize; i++) {
            for (int x = 0; x < ImageSize; x++) {
                for (int y = 0; y < ImageSize; y++) {
                    inputMatrix[i][x * ImageSize + y] = input[i][x][y];
                }
            }
        }

        TGpuMatrix<TFloatType, BatchSize, ImageSize * ImageSize> gpuInputMatrix;
        gpuInputMatrix.ToDevice(inputMatrix);

        L1_.Apply(gpuInputMatrix, L1Output_);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        L1ReLU_.Apply(L1Output_);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        L2_.Apply(L1Output_, L2Output_);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        ArgMax_.Apply(L2Output_, Result_);
        return Result_.ToHost();
    }

private:
    TGpuLinearLayer<ImageSize * ImageSize, HiddenLayerSize, BatchSize> L1_;
    TGpuReLULayer<HiddenLayerSize, BatchSize> L1ReLU_;
    TGpuLinearLayer<HiddenLayerSize, OutputClassCount, BatchSize> L2_;
    TGpuArgMaxLayer<OutputClassCount, BatchSize> ArgMax_;

    // Do not allocate intermediate data for each evaluation.
    TGpuMatrix<TFloatType, BatchSize, HiddenLayerSize> L1Output_;
    TGpuMatrix<TFloatType, BatchSize, OutputClassCount> L2Output_;
    TGpuVector<int, BatchSize> Result_;
};

int main()
{
    auto network = ReadMNISTNetwork("data/model.bin");
    auto test = ReadTestSuite("data/test.bin");

    constexpr int BatchSize = 256;

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK_ERROR(cublasCreate(&cublasHandle));

    const TMNISTNetwork& networkRef = *network;
    TGpuMNISTNetwork<BatchSize> gpuNetwork(networkRef, cublasHandle);

    auto eval = [&gpuNetwork](const TImageBatch<BatchSize>& image) {
        return gpuNetwork.Eval(image);
    };

    TestMNISTNetwork<BatchSize>(test, eval);

    CUBLAS_CHECK_ERROR(cublasDestroy(cublasHandle));

    return 0;
}
