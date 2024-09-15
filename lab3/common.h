#pragma once

#include <array>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>

#include <cublas_v2.h>

constexpr int ImageSize = 28;

constexpr int HiddenLayerSize = 1000;
constexpr int OutputClassCount = 10;

template <class TFloatType, size_t N>
using TVector = std::array<TFloatType, N>;

template <class TFloatType, size_t N, size_t M>
using TMatrix = std::array<TVector<TFloatType, M>, N>;

template <class TFloatType, size_t InDimension, size_t OutDimension>
struct TLinearLayer
{
    TMatrix<TFloatType, OutDimension, InDimension> Weights;
    TVector<TFloatType, OutDimension> Biases;
};

// For this lab we always use floats, but let's keep some
// primitives generic.
using TFloatType = float;

struct TMNISTNetwork
{
    TLinearLayer<TFloatType, ImageSize * ImageSize, HiddenLayerSize> L1;
    TLinearLayer<TFloatType, HiddenLayerSize, OutputClassCount> L2;
};
using TMNISTNetworkPtr = std::unique_ptr<TMNISTNetwork>;

using TImage = TMatrix<TFloatType, ImageSize, ImageSize>;

template <size_t N>
using TImageBatch = std::array<TImage, N>;

struct TTestCase
{
    TImage Image;
    int Class;
};
using TTestSuite = std::vector<TTestCase>;

class TNonCopyable
{
public: 
    TNonCopyable(const TNonCopyable&) = delete;
    TNonCopyable& operator = (const TNonCopyable&) = delete;

protected:
    TNonCopyable() = default;
    ~TNonCopyable() = default;
};

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }

#define CUBLAS_CHECK_ERROR(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error in %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    }

#define CUDA_CHECK_KERNEL() \
    CUDA_CHECK_ERROR(cudaPeekAtLastError());

template <class TFloatType, size_t N>
class TGPUVector
    : public TNonCopyable
{
public:
    TGPUVector() = default;

    ~TGPUVector()
    {
        MaybeFree();
    }

    void Allocate()
    {
        if (!Data_) {
            MaybeFree();
            CUDA_CHECK_ERROR(cudaMalloc(&Data_, N * sizeof(TFloatType)));
        }
    }

    void ToDevice(const TVector<TFloatType, N>& hostData)
    {
        Allocate();
        CUDA_CHECK_ERROR(cudaMemcpy(Data_, hostData.data(), N * sizeof(TFloatType), cudaMemcpyHostToDevice));
    }

    TVector<TFloatType, N> ToHost() const
    {
        TVector<TFloatType, N> hostData;
        CUDA_CHECK_ERROR(cudaMemcpy(hostData.data(), Data_, N * sizeof(TFloatType), cudaMemcpyDeviceToHost));

        return hostData;
    }

    TFloatType* Data()
    {
        return Data_;
    }

    const TFloatType* Data() const
    {
        return Data_;
    }

private:
    TFloatType* Data_ = nullptr;

    void MaybeFree()
    {
        if (Data_) {
            CUDA_CHECK_ERROR(cudaFree(Data_));
            Data_ = nullptr;
        }
    }
};

// Stores elements in row-major order.
template <class TFloatType, size_t N, size_t M>
class TGPUMatrix
    : public TNonCopyable
{
public:
    TGPUMatrix() = default;

    ~TGPUMatrix()
    {
        MaybeFree();
    }

    void Allocate()
    {
        if (!Data_) {
            CUDA_CHECK_ERROR(cudaMalloc(&Data_, N * M * sizeof(TFloatType)));
        }
    }

    void ToDevice(const TMatrix<TFloatType, N, M>& hostData)
    {
        Allocate();
        cudaMemcpy(Data_, hostData.data(), N * M * sizeof(TFloatType), cudaMemcpyHostToDevice);
    }

    void ToDeviceTransposed(const TMatrix<TFloatType, M, N>& hostData)
    {
        TMatrix<TFloatType, N, M> transposed;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                transposed[i][j] = hostData[j][i];
            }
        }

        ToDevice(transposed);
    }

    TMatrix<TFloatType, N, M> ToHost() const
    {
        TMatrix<TFloatType, N, M> hostData;
        CUDA_CHECK_ERROR(cudaMemcpy(hostData.data(), Data_, N * M * sizeof(TFloatType), cudaMemcpyDeviceToHost));

        return hostData;
    }

    TMatrix<TFloatType, M, N> ToHostTransposed() const
    {
        auto hostData = ToHost();

        TMatrix<TFloatType, M, N> result;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                result[j][i] = hostData[i][j];
            }
        }

        return result;
    }

    TFloatType* Data()
    {
        return Data_;
    }

    const TFloatType* Data() const
    {
        return Data_;
    }

private:
    TFloatType* Data_ = nullptr;

    void MaybeFree()
    {
        if (Data_) {
            CUDA_CHECK_ERROR(cudaFree(Data_));
            Data_ = nullptr;
        }
    }
};

template <class TFloatType>
TFloatType ReadFloat(std::ifstream& in)
{
    std::string hex;
    hex.reserve(2 * sizeof(TFloatType));
    in >> hex;
    assert(hex.size() == 2 * sizeof(TFloatType));

    std::array<char, sizeof(TFloatType)> bytes;
    
    auto hexToInt = [] (char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }

        assert(c >= 'a' && c <= 'f');
        return c - 'a' + 10;
    };
    for (int i = 0; i < sizeof(TFloatType); ++i) {
        char c = hexToInt(hex[2 * i]) * 16 + hexToInt(hex[2 * i + 1]);
        bytes[i] = c;
    }

    TFloatType result;
    memcpy(&result, bytes.data(), sizeof(TFloatType));
    return result;
}

template <class TFloatType, size_t N>
TVector<TFloatType, N> ReadVector(std::ifstream& in)
{
    int shapeSize;
    in >> shapeSize;
    assert(shapeSize == 1);

    int n;
    in >> n;
    assert(n == N);

    TVector<TFloatType, N> result;
    for (auto& value : result) {
        value = ReadFloat<TFloatType>(in);
    }

    return result;
}

template <class TFloatType, size_t N, size_t M>
TMatrix<TFloatType, N, M> ReadMatrix(std::ifstream& in)
{
    int shapeSize;
    in >> shapeSize;
    assert(shapeSize == 2);

    int n;
    in >> n;
    assert(n == N);

    int m;
    in >> m;
    assert(m == M);

    TMatrix<TFloatType, N, M> result;
    for (auto& row : result) {
        for (auto& value : row) {
            value = ReadFloat<TFloatType>(in);
        }
    }

    return result;
}

TImage ReadImage(std::ifstream& in)
{
    return ReadMatrix<TFloatType, ImageSize, ImageSize>(in);
}

TTestSuite ReadTestSuite(const std::string& filename)
{
    std::ifstream in(filename);
    assert(in.is_open());

    int testCount;
    in >> testCount;
    TTestSuite result(testCount);
    for (int testIndex = 0; testIndex < testCount; ++testIndex) {
        std::string name;
        in >> name;
        assert(name == "image");

        auto image = ReadImage(in);

        int imageClass;
        in >> imageClass;

        result[testIndex] = TTestCase{
            .Image = image,
            .Class = imageClass,
        };
    }

    return result;
}

TMNISTNetworkPtr ReadMNISTNetwork(const std::string& filename)
{
    auto network = std::make_unique<TMNISTNetwork>();

    std::ifstream in(filename);
    assert(in.is_open());

    for (int i = 0; i < 4; i++) {
        std::string name;
        in >> name;
        if (name == "l1.weight") {
            network->L1.Weights = ReadMatrix<TFloatType, HiddenLayerSize, ImageSize * ImageSize>(in);
        } else if (name == "l1.bias") {
            network->L1.Biases = ReadVector<TFloatType, HiddenLayerSize>(in);
        } else if (name == "l2.weight") {
            network->L2.Weights = ReadMatrix<TFloatType, OutputClassCount, HiddenLayerSize>(in);
        } else if (name == "l2.bias") {
            network->L2.Biases = ReadVector<TFloatType, OutputClassCount>(in);
        } else {
            assert(false);
        }
    }

    return network;
}

template <size_t BatchSize, class FEval>
void TestMNISTNetwork(const TTestSuite& test, FEval eval)
{
    auto startTimestamp = std::chrono::steady_clock::now();

    int batchIndex = 0;
    int correctCount = 0;
    int totalCount = 0;

    int logAccuracyEveryBatchCount = 1000 / BatchSize;
    for (size_t batchStart = 0; batchStart < test.size(); batchStart += BatchSize) {
        batchIndex++;

        TImageBatch<BatchSize> batch;
        std::array<int, BatchSize> expectedClasses;
        for (size_t i = 0; i < BatchSize; ++i) {
            if (batchStart + i >= test.size()) {
                break;
            }

            batch[i] = test[batchStart + i].Image;
            expectedClasses[i] = test[batchStart + i].Class;
        }

        std::array<int, BatchSize> predictedClasses = eval(batch);
        for (size_t i = 0; i < BatchSize; ++i) {
            if (predictedClasses[i] == expectedClasses[i]) {
                ++correctCount;
            }
        }

        totalCount += BatchSize;

        if (batchIndex % logAccuracyEveryBatchCount == 0) {
            std::cout << "Accuracy: " << correctCount << " / " << totalCount << " = " << 1.0 * correctCount / totalCount << std::endl;
        }
    }

    std::cout << "Final Accuracy: " << correctCount << " / " << totalCount << " = " << 1.0 * correctCount / totalCount << std::endl;

    auto endTimestamp = std::chrono::steady_clock::now();
    auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTimestamp - startTimestamp).count();
    int batchCount = (test.size() + BatchSize - 1) / BatchSize;
    std::cout <<
        "Elapsed time: " << elapsedMilliseconds << " ms, " <<
        "Time per batch: " << 1.0 * elapsedMilliseconds / batchCount << " ms, " <<
        "Images per second: " << 1000.0 * test.size() / elapsedMilliseconds << " imgs/s" << std::endl;
}
