#include "parser.h"

#include <optional>
#include <stdexcept>
#include <vector>

// Removes space symbols from the script and splits it into expressions.
std::vector<std::string> PreprocessScript(const std::string& script)
{
    std::vector<std::string> expressions;
    std::string expression;

    for (char c : script) {
        if (c == ';') {
            if (expression.empty()) {
                throw std::runtime_error("Empty expression");
            }

            expressions.push_back(expression);
            expression.clear();
        } else if (!std::isspace(c)) {
            expression.push_back(c);
        }
    }

    if (!expression.empty()) {
        throw std::runtime_error("Unexpected end of script");
    }

    return expressions;
}

TNodeBasePtr ParseScript(const TScript& script)
{
    std::unordered_map<int, TNodeBasePtr> nodes;
    std::optional<int> outputNodeIndex;

    for (const auto& expression : PreprocessScript(script.Script)) {
        int ptr = 0;

        auto skip = [&] (char c) {
            if (ptr == expression.size()) {
                throw std::runtime_error(std::string{"Unexpected end of expression, expected \'"} + c + "\'");
            }
            if (expression[ptr] != c) {
                throw std::runtime_error(std::string{"Unexpected symbol, expected \'"} + c + "\', got \'" + expression[ptr] + "\'");
            }
            ++ptr;
        };

        auto parseInt = [&] {
            if (ptr == expression.size()) {
                throw std::runtime_error("Unexpected end of expression, expected integer");
            }
            if (!std::isdigit(expression[ptr])) {
                throw std::runtime_error("Unexpected symbol, expected integer");
            }
            int result = 0;
            while (ptr < expression.size() && std::isdigit(expression[ptr])) {
                result = result * 10 + expression[ptr] - '0';
                ++ptr;
            }

            return result;
        };

        auto parseIntList = [&] {
            if (ptr == expression.size()) {
                throw std::runtime_error("Unexpected end of expression");
            }

            std::vector<int> result;
            skip('[');
            if (expression[ptr] == ']') {
                ++ptr;
                return result;
            }

            while (true) {
                result.push_back(parseInt());
                if (expression[ptr] == ']') {
                    ++ptr;
                    return result;
                }

                skip(',');
            }
        };

        auto parseStringArg = [&] {
            if (ptr == expression.size()) {
                throw std::runtime_error("Unexpected end of expression");
            }

            std::string result;
            while (expression[ptr] != ',' && expression[ptr] != ')') {
                result.push_back(expression[ptr]);
                ++ptr;
            }

            return result;
        };

        auto parseValueTypeArg = [&] {
            auto valueType = parseStringArg();
            if (valueType == "float16") {
                return EValueType::Float16;
            } else if (valueType == "float32") {
                return EValueType::Float32;
            } else if (valueType == "float64") {
                return EValueType::Float64;
            } else if (valueType == "int16") {
                return EValueType::Int16;
            } else if (valueType == "int32") {
                return EValueType::Int32;
            } else if (valueType == "int64") {
                return EValueType::Int64;
            } else {                
                throw std::runtime_error("Unknown value type: " + valueType);
            }
        };

        const std::string Result = "result";
        if (expression.substr(0, Result.size()) == Result) {
            ptr += Result.size();
            skip('=');
            outputNodeIndex = parseInt();
            if (ptr < expression.size()) {
                throw std::runtime_error("Unexpected symbols after the end of expression");
            }

            continue;
        }

        TNodeBasePtr node;

        auto outputNodeIndex = parseInt();
        skip('=');

        std::string nodeType;
        while (ptr < expression.size() && std::isalpha(expression[ptr])) {
            nodeType.push_back(expression[ptr]);
            ++ptr;
        }

        skip('(');
        if (nodeType == "InputTensor") {
            auto name = parseStringArg();
            skip(',');
            auto type = parseValueTypeArg();
            skip(',');
            auto shape = parseIntList();

            node = std::make_shared<TInputNode>(name, TTensorMeta{type, shape});
        } else if (nodeType == "ConstantTensor") {
            auto name = parseStringArg();
            skip(',');
            auto type = parseValueTypeArg();
            skip(',');
            auto shape = parseIntList();

            auto constantIt = script.Constants.find(name);
            if (constantIt == script.Constants.end()) {
                throw std::runtime_error("Constant \"" + name + "\" not found");
            }

            const auto& constant = constantIt->second;
            node = std::make_shared<TConstantNode>(TTensorMeta{type, shape}, constant);
        } else if (nodeType == "SumNode") {
            auto lhs = parseInt();
            skip(',');
            auto rhs = parseInt();

            auto lhsIt = nodes.find(lhs);
            auto rhsIt = nodes.find(rhs);
            if (!lhs || !rhs) {
                throw std::runtime_error("Expression references unknown node");
            }

            node = std::make_shared<TSumNode>(lhsIt->second, rhsIt->second);
        } else if (nodeType == "MulNode") {
            auto lhs = parseInt();
            skip(',');
            auto rhs = parseInt();

            auto lhsIt = nodes.find(lhs);
            auto rhsIt = nodes.find(rhs);
            if (lhsIt == nodes.end() || rhsIt == nodes.end()) {
                throw std::runtime_error("Expression references unknown node");
            }

            node = std::make_shared<TMulNode>(lhsIt->second, rhsIt->second);
        } else if (nodeType == "ReLUNode") {
            auto input = parseInt();

            auto inputIt = nodes.find(input);
            if (inputIt == nodes.end()) {
                throw std::runtime_error("Expression references unknown node");
            }

            node = std::make_shared<TReLUNode>(inputIt->second);
        } else if (nodeType == "SiLUNode") {
            auto input = parseInt();

            auto inputIt = nodes.find(input);
            if (inputIt == nodes.end()) {
                throw std::runtime_error("Expression references unknown node");
            }

            node = std::make_shared<TSiLUNode>(inputIt->second);
        } else {
            throw std::runtime_error("Unknown node type: " + nodeType);
        }

        skip(')');

        if (ptr < expression.size()) {
            throw std::runtime_error("Unexpected symbols after the end of expression");
        }

        if (!nodes.emplace(outputNodeIndex, node).second) {
            throw std::runtime_error("Node is defined twice");
        }
    }

    if (!outputNodeIndex) {
        throw std::runtime_error("Output node index is not defined");
    }

    auto outputNodeIt = nodes.find(*outputNodeIndex);
    if (outputNodeIt == nodes.end()) {
        throw std::runtime_error("Output node references unknown node");
    }

    return outputNodeIt->second;
}
