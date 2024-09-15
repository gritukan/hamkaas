#pragma once

extern "C" void HamKaasFreeErrorMessage(char* message);

struct TCompilationResult
{
    const void* Model;
    const char* ErrorMessage;
};

struct TNamedTensor
{
    const char* Name;
    const char* Data;
};

extern "C" TCompilationResult HamKaasCompileModel(
    const char* scriptString,
    const TNamedTensor* constantTensors,
    int constantTensorCount);

extern "C" void HamKaasFreeModel(const void* model);

extern "C" const char* HamKaasEvaluateModel(
    const void* model,
    const TNamedTensor* inputTensors,
    int inputTensorCount,
    char* outputTensor);

// For testing purposes only.

// Takes a tensor and returns the inverse of each element.
// Returns null if successful, otherwise returns an error message.
// If error message is returned, caller must free the memory by calling FreeErrorMessage.
extern "C" char* HamKaasInverseElements(float* inputTensor, float* outputTensor, int size);
