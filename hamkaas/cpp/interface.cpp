#include "interface.h"

#include "error.h"
#include "model.h"
#include "parser.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <stdlib.h>
#include <unordered_map>

extern "C" void HamKaasFreeErrorMessage(char* message)
{
    free(message);
}

extern "C" TCompilationResult HamKaasCompileModel(
    const char* scriptString,
    const TNamedTensor* constantTensors,
    int constantTensorCount)
{
    std::unordered_map<std::string, const char*> constants;
    for (int index = 0; index < constantTensorCount; ++index) {
        constants[constantTensors[index].Name] = constantTensors[index].Data;
    }

    std::string script(scriptString);

    try {
        auto rootNode = NHamKaas::ParseScript(script);
        auto model = new NHamKaas::TModel{std::move(rootNode)};
        model->Compile(constants);
        return {model, nullptr};
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return {nullptr, message};
    }
}

extern "C" void HamKaasFreeModel(const void* model)
{
    delete static_cast<const NHamKaas::TModel*>(model);
}

extern "C" const char* HamKaasEvaluateModel(
    const void* model,
    const TNamedTensor* inputTensors,
    int inputTensorCount,
    char* outputTensor)
{
    std::unordered_map<std::string, const char*> inputs;
    for (int index = 0; index < inputTensorCount; ++index) {
        inputs[inputTensors[index].Name] = inputTensors[index].Data;
    }

    try {
        auto modelPtr = static_cast<const NHamKaas::TModel*>(model);
        modelPtr->Evaluate(inputs, outputTensor);
        return nullptr;
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return message;
    }
}

void HamKaasDoInverseElements(float* inputTensor, float* outputTensor, int size)
{
    for (int i = 0; i < size; ++i) {
        if (inputTensor[i] == 0.0) {
            THROW("Division by zero", VAR(index, i));
        }
        outputTensor[i] = 1.0f / inputTensor[i];
    }
}

extern "C" char* HamKaasInverseElements(float* inputTensor, float* outputTensor, int size)
{
    try {
        HamKaasDoInverseElements(inputTensor, outputTensor, size);
        return nullptr;
    } catch (const std::exception& e) {
        char* message = strdup(e.what());
        assert(message && "Out of memory for error");
        return message;
    }
}
