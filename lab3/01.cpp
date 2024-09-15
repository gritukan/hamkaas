#include "common.h"

#include <algorithm>

int EvalNetwork(const TMNISTNetwork& network, const TImage& image)
{
    std::array<double, HiddenLayerSize> hiddenLayer;
    // Your code here: multiply image by L1 weights, add biases and apply ReLU.

    std::array<double, OutputClassCount> outputLayer;
    // Your code here: multiply hiddenLayer by L2 weights and add biases.

    // Retun the index of the digit with maximum probability.
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
