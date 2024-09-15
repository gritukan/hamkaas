#pragma once

extern "C" void FreeErrorMessage(char* message);

struct TCompilationResult
{
    const void* Model;
    const char* ErrorMessage;
};

struct TNamedTensor
{
    const char* Name;
    const void* Data;
};

extern "C" TCompilationResult CompileModel(
    const char* scriptString,
    const TNamedTensor* constantTensors,
    int constantTensorCount);

extern "C" void FreeModel(const void* model);

extern "C" const char* EvaluateModel(
    const void* model,
    const TNamedTensor* inputTensors,
    int inputTensorCount,
    void* outputTensor);

// For testing purposes only.

// Takes a tensor and returns the inverse of each element.
// Returns null if successful, otherwise returns an error message.
// If error message is returned, caller must free the memory by calling FreeErrorMessage.
extern "C" char* InverseElements(float* inputTensor, float* outputTensor, int size);
