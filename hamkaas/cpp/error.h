#include <cublas_v2.h>

#include <stdexcept>
#include <sstream>
#include <vector>

#ifdef USE_CUDNN
#include "cudnn-frontend/include/cudnn_frontend.h"
#endif

namespace NHamKaas {

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& data)
{
    stream << "[";
    for (size_t index = 0; index < data.size(); ++index) {
        stream << data[index];
        if (index != data.size() - 1) {
            stream << ", ";
        }
    }
    stream << "]";

    return stream;
}

template <class T>
class NamedValue
{
public:
    NamedValue(const char* name, T value)
        : Name_(name)
        , Value_(std::move(value))
    { }
    
    std::string ToString() const
    {
        std::ostringstream stream;
        stream << Name_ << " = " << Value_;
        return stream.str();
    }

private:
    const char* Name_;
    const T Value_;
};

class HamkaasException
    : public std::exception
{
public:
    template <typename... Args>
    HamkaasException(const std::string& file, int line, const std::string& message)
    {
        std::ostringstream stream;
        stream << file << ":" << line << ": " << message;
        Message_ = stream.str();
    }

    template <typename... Args>
    HamkaasException(const std::string& file, int line, const std::string& message, Args... args)
    {
        std::ostringstream stream;
        stream << file << ":" << line << ": " << message << " (";
        AddArgs(stream, /*first*/ true, args...);
        stream << ")";

        Message_ = stream.str();
    }

    const char* what() const noexcept override
    {
        return Message_.c_str();
    }

private:
    std::string Message_;

    void AddArgs(std::ostringstream& oss, bool /*first*/)
    { }

    template<typename T, typename... Args>
    void AddArgs(std::ostringstream& stream, bool first, const NamedValue<T>& value, Args... args)
    {
        if (!first) {
            stream << ", ";
        }
        stream << value.ToString();

        AddArgs(stream, /*first*/ false, args...);
    }
};

} // namespace NHamKaas

#define THROW(message, ...) \
    throw NHamKaas::HamkaasException(__FILE__, __LINE__, message, ##__VA_ARGS__)

#define NUM_ARGS(...)  NUM_ARGS_IMPL(__VA_ARGS__, 2, 1)
#define NUM_ARGS_IMPL(_1, _2, N, ...) N

#define MACRO_CHOOSER(name, ...) MACRO_CHOOSER_IMPL(name, NUM_ARGS(__VA_ARGS__))
#define MACRO_CHOOSER_IMPL(name, n) MACRO_CHOOSER_IMPL2(name, n)
#define MACRO_CHOOSER_IMPL2(name, n) name##n

#define VAR1(value) NHamKaas::NamedValue(#value, value)
#define VAR2(name, value) NHamKaas::NamedValue(#name, value)

#define VAR(...) MACRO_CHOOSER(VAR, __VA_ARGS__)(__VA_ARGS__)

#define CUDA_CHECK_ERROR0() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            THROW("CUDA error", VAR(error, cudaGetErrorString(error))); \
        } \
    } while (0)

#define CUDA_CHECK_ERROR1(error) \
    do { \
        if (error != cudaSuccess) { \
            THROW("CUDA error", VAR(error, cudaGetErrorString(error))); \
        } \
    } while (0)

#define CUDA_CHECK_ERROR(...) MACRO_CHOOSER(CUDA_CHECK_ERROR, __VA_ARGS__)(__VA_ARGS__)

#define CUDA_ASSERT(error) \
    do { \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK_ERROR(error) \
    do { \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            THROW("cuBLAS error", VAR(error)); \
        } \
    } while (0)

#define CUBLAS_ASSERT(error) \
    do { \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, error); \
            exit(1); \
        } \
    } while (0)

#ifdef USE_CUDNN

#define CUDNN_CHECK_ERROR(error) \
    do { \
        if (error != CUDNN_STATUS_SUCCESS) { \
            THROW("CUDNN error", VAR(std::to_string(error))); \
        } \
    } while (0)

#define CUDNN_ASSERT(error) \
    do { \
        if (error != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "CUDNN error in %s:%d: %d\n", __FILE__, __LINE__, error); \
            exit(1); \
        } \
    } while (0)

#define CUDNN_FE_CHECK_ERROR(status) \
    do { \
        if (status.is_bad()) { \
            THROW("CUDNN frontend error", VAR(status.get_message())); \
        } \
    } while (0)

#endif
