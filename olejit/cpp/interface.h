extern "C" void FreeErrorMessage(char* message);

extern "C" void* CompileModel(const char* oleScript);
// For testing purposes only.

// Takes a tensor and returns the inverse of each element.
// Returns null if successful, otherwise returns an error message.
// If error message is returned, caller must free the memory by calling FreeErrorMessage.
extern "C" char* InverseElements(float* inputTensor, float* outputTensor, int size);
