#include "common.h"

#include "cudnn-frontend/include/cudnn_frontend.h"

#define CUDNN_CHECK_ERROR(error) \
    if (error != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error("CUDNN error: " + std::to_string(error)); \
    }

namespace fe = cudnn_frontend;

void ThrowOnError(fe::error_t status)
{
    if (status.is_bad()) {
        throw std::runtime_error(status.get_message());
    }
}

constexpr int BatchSize = 128;

// Computes the argmax of each of the B vectors of size N.
template <int N, int B>
__global__ void ArgMaxKernel(const TFloatType* a, int* result)
{
    // Your code here.
}

class TCudnnGraphBuilder
{
public:
    explicit TCudnnGraphBuilder(cudnnHandle_t handle)
        : Handle_(handle)
        , Graph_(std::make_shared<fe::graph::Graph>())
    { }

    std::shared_ptr<fe::graph::Tensor_attributes> AddVector(
        const std::string& name,
        int n)
    {
        // Your code here.
    }

    std::shared_ptr<fe::graph::Tensor_attributes> AddRowMajorMatrix(
        const std::string& name,
        int n,
        int m)
    {
        // Your code here.
    }

    std::shared_ptr<fe::graph::Tensor_attributes> AddColumnMajorMatrix(
        const std::string& name,
        int n,
        int m)
    {
        // Your code here.
    }

    std::shared_ptr<fe::graph::Tensor_attributes> AddReLU(
        const std::string& name,
        std::shared_ptr<fe::graph::Tensor_attributes> input)
    {
        // Your code here.
    }

    template <size_t InputDimension, size_t OutputDimension>
    std::shared_ptr<fe::graph::Tensor_attributes> AddLinearLayer(
        const std::string& name,
        std::shared_ptr<fe::graph::Tensor_attributes> input,
        const TGpuMatrix<TFloatType, OutputDimension, InputDimension>& weightsData,
        const TGpuVector<TFloatType, OutputDimension>& biasData)
    {
        auto weights = AddColumnMajorMatrix(name + "_weights", InputDimension, OutputDimension);
        TensorMap_[weights] = const_cast<void*>(reinterpret_cast<const void*>(weightsData.Data()));

        auto bias = AddVector(name + "_bias", OutputDimension);
        TensorMap_[bias] = const_cast<void*>(reinterpret_cast<const void*>(biasData.Data()));

        // Your code here.
    }

    std::shared_ptr<fe::graph::Graph> Build() const
    {
        ThrowOnError(Graph_->validate());
        ThrowOnError(Graph_->build_operation_graph(Handle_));
        ThrowOnError(Graph_->create_execution_plans({fe::HeurMode_t::A}));
        ThrowOnError(Graph_->check_support(Handle_));
        ThrowOnError(Graph_->build_plans(Handle_, fe::BuildPlanPolicy_t::ALL));

        return Graph_;
    }

    const std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>& GetTensorMap() const
    {
        return TensorMap_;
    }

private:
    const cudnnHandle_t Handle_;

    std::shared_ptr<fe::graph::Graph> Graph_;

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> TensorMap_;
};

class TMnistCudnnNetwork
{
public:
    TMnistCudnnNetwork(const TMNISTNetwork& network, cudnnHandle_t handle)
        : Handle_(handle)
    {
        // Build L1 graph.
        {
            TCudnnGraphBuilder builder(handle);

            auto input = builder.AddRowMajorMatrix("input", BatchSize, ImageSize * ImageSize);

            L1Weights_.ToDevice(network.L1.Weights);
            L1Biases_.ToDevice(network.L1.Biases);
    
            auto l1Result = builder.AddLinearLayer(
                "l1",
                input,
                L1Weights_,
                L1Biases_);
            auto l1ReLU = builder.AddReLU("l1_relu", l1Result);

            L1Graph_ = builder.Build();

            L1TensorMap_ = builder.GetTensorMap();
            Input_.Allocate();
            L1TensorMap_[input] = Input_.Data();
            L1Result_.Allocate();
            L1TensorMap_[l1ReLU] = L1Result_.Data();

            CUDA_CHECK_ERROR(cudaMalloc(&L1GraphWorkspace_, L1Graph_->get_workspace_size()));
        }

        // Build L2 graph.
        {
            TCudnnGraphBuilder builder(handle);

            auto l1Result = builder.AddRowMajorMatrix("l1_result", BatchSize, HiddenLayerSize);

            L2Weights_.ToDevice(network.L2.Weights);
            L2Biases_.ToDevice(network.L2.Biases);

            auto l2Result = builder.AddLinearLayer(
                "l2",
                l1Result,
                L2Weights_,
                L2Biases_);
            l2Result->set_output(true).set_data_type(fe::DataType_t::FLOAT);            

            L2Graph_ = builder.Build();

            L2TensorMap_ = builder.GetTensorMap();
            L1Result_.Allocate();
            L2TensorMap_[l1Result] = L1Result_.Data();
            L2Result_.Allocate();
            L2TensorMap_[l2Result] = L2Result_.Data();

            CUDA_CHECK_ERROR(cudaMalloc(&L2GraphWorkspace_, L2Graph_->get_workspace_size()));
        }

        Result_.Allocate();
    }

    ~TMnistCudnnNetwork()
    {
        CUDA_CHECK_ERROR(cudaFree(L1GraphWorkspace_));
        CUDA_CHECK_ERROR(cudaFree(L2GraphWorkspace_));
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

        Input_.ToDevice(inputMatrix);

        ThrowOnError(L1Graph_->execute(Handle_, L1TensorMap_, L1GraphWorkspace_));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        ThrowOnError(L2Graph_->execute(Handle_, L2TensorMap_, L2GraphWorkspace_));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        ArgMaxKernel<OutputClassCount, BatchSize><<<1, BatchSize>>>(L2Result_.Data(), Result_.Data());
        CUDA_CHECK_ERROR(cudaGetLastError());
        return Result_.ToHost();
    }

private:
    const cudnnHandle_t Handle_;

    std::shared_ptr<fe::graph::Graph> L1Graph_;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> L1TensorMap_;
    uint8_t* L1GraphWorkspace_;

    std::shared_ptr<fe::graph::Graph> L2Graph_;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> L2TensorMap_;
    uint8_t* L2GraphWorkspace_;

    std::shared_ptr<fe::graph::Tensor_attributes> L1Input_;
    std::shared_ptr<fe::graph::Tensor_attributes> L1Output_;

    std::shared_ptr<fe::graph::Tensor_attributes> L2Input_;
    std::shared_ptr<fe::graph::Tensor_attributes> L2Output_;

    TGpuMatrix<TFloatType, HiddenLayerSize, ImageSize * ImageSize> L1Weights_;
    TGpuVector<TFloatType, HiddenLayerSize> L1Biases_;
    TGpuMatrix<TFloatType, OutputClassCount, HiddenLayerSize> L2Weights_;
    TGpuVector<TFloatType, OutputClassCount> L2Biases_;

    TGpuMatrix<TFloatType, BatchSize, ImageSize * ImageSize> Input_;
    TGpuMatrix<TFloatType, BatchSize, HiddenLayerSize> L1Result_;
    TGpuMatrix<TFloatType, BatchSize, OutputClassCount> L2Result_;
    TGpuVector<int, BatchSize> Result_;
};

int main()
{
    cudnnHandle_t handle;
    CUDNN_CHECK_ERROR(cudnnCreate(&handle));

    auto network = ReadMNISTNetwork("data/model.bin");
    auto test = ReadTestSuite("data/test.bin");

    const TMNISTNetwork& networkRef = *network;
    TMnistCudnnNetwork gpuNetwork(networkRef, handle);

    auto eval = [&gpuNetwork](const TImageBatch<BatchSize>& image) {
        return gpuNetwork.Eval(image);
    };

    TestMNISTNetwork<BatchSize>(test, eval);

    CUDNN_CHECK_ERROR(cudnnDestroy(handle));

    return 0;
}
