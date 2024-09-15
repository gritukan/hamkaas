#include "common.h"

#include <algorithm>

int EvalNetwork(const TMNISTNetwork& network, const TImage& image)
{
    std::array<double, HiddenLayerSize> hiddenLayer;
    for (int i = 0; i < HiddenLayerSize; ++i) {
        hiddenLayer[i] = network.L1.Biases[i];
        for (int j = 0; j < ImageSize * ImageSize; ++j) {
            hiddenLayer[i] += image[j / ImageSize][j % ImageSize] * network.L1.Weights[i][j];
        }
        hiddenLayer[i] = std::max(0.0, hiddenLayer[i]);
    }

    for (int i = 0; i < 10; i++) {
        std::cout << hiddenLayer[i] << " ";
    }
    std::cout << std::endl;
    assert(false);

    std::array<double, OutputClassCount> outputLayer;
    for (int i = 0; i < OutputClassCount; ++i) {
        outputLayer[i] = network.L2.Biases[i];
        for (int j = 0; j < HiddenLayerSize; ++j) {
            outputLayer[i] += hiddenLayer[j] * network.L2.Weights[i][j];
        }
    }

    return std::max_element(outputLayer.begin(), outputLayer.end()) - outputLayer.begin();
}

int main()
{
    auto network = ReadMNISTNetwork("data/model.bin");
    auto test = ReadTestSuite("data/test.bin");

    auto eval = [&network](const TImageBatch<1>& image) {
        std::array<int, 1> result;
        result[0] = EvalNetwork(*network, image[0]);
        return result;
    };

    TestMNISTNetwork</*BatchSize*/ 1>(test, eval);

    return 0;
}
