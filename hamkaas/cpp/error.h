#include <stdexcept>
#include <sstream>

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

#define THROW(message, ...) \
    throw HamkaasException(__FILE__, __LINE__, message, ##__VA_ARGS__)

#define NVAR(name, value) NamedValue(#name, value)
#define VAR(var) NamedValue(#var, var)
