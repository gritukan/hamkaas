#include "interface.h"
#include "model.h"
#include "parser.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <stdlib.h>
#include <unordered_map>

extern "C" void FreeErrorMessage(char* message)
{
    printf("%p\n", message);
    free(message);
}

extern "C" TCompilationResult CompileModel(
    const char* scriptString,
    const TNamedTensor* constantTensors,
    int constantTensorCount)
{
    std::unordered_map<std::string, const void*> constants;
    for (int index = 0; index < constantTensorCount; ++index) {
        constants[constantTensors[index].Name] = constantTensors[index].Data;
    }

    TScript script{
        .Script = std::string{scriptString},
        .Constants = std::move(constants),
    };

    try {
        auto rootNode = ParseScript(script);
        auto model = new TModel{std::move(rootNode)};
        return {model, nullptr};
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return {nullptr, message};
    }
}

extern "C" void FreeModel(const void* model)
{
    printf("%p\n", model);
    delete static_cast<const TModel*>(model);
}

extern "C" const char* EvaluateModel(
    const void* model,
    const TNamedTensor* inputTensors,
    int inputTensorCount,
    void* outputTensor)
{
    std::unordered_map<std::string, const void*> inputs;
    for (int index = 0; index < inputTensorCount; ++index) {
        inputs[inputTensors[index].Name] = inputTensors[index].Data;
    }

    try {
        auto modelPtr = static_cast<const TModel*>(model);
        modelPtr->Evaluate(inputs, outputTensor);
        return nullptr;
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return message;
    }
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
