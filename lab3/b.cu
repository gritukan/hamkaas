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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= K) {
        return;
    }

    TFloatType sum = C[x * K + y];
    for (int i = 0; i < M; i++) {
        sum += A[x * M + i] * B[y * M + i];
    }
    C[x * K + y] = sum;
}

// Applies the ReLU activation function to the batch of B vectors of size N.
template <int N, int B>
__global__ void ReLUKernel(TFloatType* a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * N) {
        return;
    }

    a[i] = max(0.0, a[i]);
}

// Computes the argmax of each of the B vectors of size N.
template <int N, int B>
__global__ void ArgMaxKernel(const TFloatType* a, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * N) {
        return;
    }

    TFloatType max = a[i * N];
    int maxIndex = 0;
    for (int j = 1; j < N; j++) {
        if (a[i * N + j] > max) {
            max = a[i * N + j];
            maxIndex = j;
        }
    }

    result[i] = maxIndex;
}

template <size_t InputDimension, size_t OutputDimension, size_t BatchSize>
class TGPULinearLayer
{
public:
    explicit TGPULinearLayer(const TLinearLayer<TFloatType, InputDimension, OutputDimension>& layer, cublasHandle_t cublasHandle)
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
    void Apply(const TGPUMatrix<TFloatType, BatchSize, InputDimension>& input, TGPUMatrix<TFloatType, BatchSize, OutputDimension>& output)
    {
        // Initialize the output with the biases.
        CUDA_CHECK_ERROR(cudaMemcpy(output.Data(), Biases_.Data(), BatchSize * OutputDimension * sizeof(TFloatType), cudaMemcpyDeviceToDevice));

//#define CUBLAS
#ifdef CUBLAS
        TFloatType alpha = 1.0;
        TFloatType beta = 1.0;
        CUBLAS_CHECK_ERROR(cublasGemmEx(
            CublasHandle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            OutputDimension,
            BatchSize,
            InputDimension,
            &alpha,
            Weights_.Data(),
            CUDA_R_32F,
            InputDimension,
            input.Data(),
            CUDA_R_32F,
            InputDimension,
            &beta,
            output.Data(),
            CUDA_R_32F,
            OutputDimension,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
#else
        constexpr int ThreadsPerBlockDim = 16;
        dim3 threads(ThreadsPerBlockDim, ThreadsPerBlockDim);

        int blocksPerX = (BatchSize + ThreadsPerBlockDim - 1) / ThreadsPerBlockDim;
        int blocksPerY = (OutputDimension + ThreadsPerBlockDim - 1) / ThreadsPerBlockDim;
        dim3 blocks(blocksPerX, blocksPerY);

        LinearLayerKernel<BatchSize, InputDimension, OutputDimension><<<threads, blocks>>>(input.Data(), Weights_.Data(), output.Data());
        CUDA_CHECK_ERROR(cudaGetLastError());
#endif
    }

private:
    const cublasHandle_t CublasHandle_;

    TGPUMatrix<TFloatType, OutputDimension, InputDimension> Weights_;
    TGPUMatrix<TFloatType, BatchSize, OutputDimension> Biases_;
};

template <size_t Size, size_t BatchSize>
class TGPUReLULayer
{
public:
    // NB: Does not wait for the kernel completion.
    void Apply(TGPUMatrix<TFloatType, BatchSize, Size>& input)
    {
        constexpr int ThreadsPerBlock = 16;
        int blocks = (Size * BatchSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
        ReLUKernel<Size, BatchSize><<<blocks, ThreadsPerBlock>>>(input.Data());
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
};

template <size_t Size, size_t BatchSize>
class TGPUArgMaxLayer
{
public:
    // NB: Does not wait for the kernel completion.
    void Apply(const TGPUMatrix<TFloatType, BatchSize, Size>& input, TGPUVector<int, BatchSize>& result)
    {
        ArgMaxKernel<Size, BatchSize><<<1, BatchSize>>>(input.Data(), result.Data());
        CUDA_CHECK_ERROR(cudaGetLastError());
    }
};

template <size_t BatchSize>
struct TGPUMNISTNetwork
{
public:
    explicit TGPUMNISTNetwork(const TMNISTNetwork& network, cublasHandle_t cublasHandle)
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

        TGPUMatrix<TFloatType, BatchSize, ImageSize * ImageSize> gpuInputMatrix;
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
    TGPULinearLayer<ImageSize * ImageSize, HiddenLayerSize, BatchSize> L1_;
    TGPUReLULayer<HiddenLayerSize, BatchSize> L1ReLU_;
    TGPULinearLayer<HiddenLayerSize, OutputClassCount, BatchSize> L2_;
    TGPUArgMaxLayer<OutputClassCount, BatchSize> ArgMax_;

    // Do not allocate intermediate data for each evaluation.
    TGPUMatrix<TFloatType, BatchSize, HiddenLayerSize> L1Output_;
    TGPUMatrix<TFloatType, BatchSize, OutputClassCount> L2Output_;
    TGPUVector<int, BatchSize> Result_;
};

int main()
{
    auto network = ReadMNISTNetwork("data/model.bin");
    auto test = ReadTestSuite("data/test.bin");

    constexpr int BatchSize = 256;

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK_ERROR(cublasCreate(&cublasHandle));

    const TMNISTNetwork& networkRef = *network;
    TGPUMNISTNetwork<BatchSize> gpuNetwork(networkRef, cublasHandle);

    auto eval = [&gpuNetwork](const TImageBatch<BatchSize>& image) {
        return gpuNetwork.Eval(image);
    };

    TestMNISTNetwork<BatchSize>(test, eval);

    return 0;
}
