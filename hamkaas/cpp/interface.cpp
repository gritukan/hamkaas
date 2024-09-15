#include <cassert>
#include <cstring>
#include <stdexcept>
#include <stdlib.h>

extern "C" void FreeErrorMessage(char* message)
{
    free(message);
}

void DoInverseElements(float* inputTensor, float* outputTensor, int size)
{
    for (int i = 0; i < size; ++i) {
        if (inputTensor[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        outputTensor[i] = 1.0f / inputTensor[i];
    }
}

extern "C" char* InverseElements(float* inputTensor, float* outputTensor, int size)
{
    try {
        DoInverseElements(inputTensor, outputTensor, size);
        return nullptr;
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return message;
    }
}
