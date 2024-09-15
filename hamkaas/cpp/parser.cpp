#include "parser.h"

#include "error.h"

#include <optional>
#include <vector>

// Removes space symbols from the script and splits it into expressions.
std::vector<std::string> PreprocessScript(const std::string& script)
{
    std::vector<std::string> expressions;
    std::string expression;

    for (char c : script) {
        if (c == ';') {
            if (expression.empty()) {
                THROW("Empty expression");
            }

            expressions.push_back(expression);
            expression.clear();
        } else if (!std::isspace(c)) {
            expression.push_back(c);
        }
    }

    if (!expression.empty()) {
        THROW("Unexpected end of expression", VAR(expression));
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
                THROW("Unexpected end of expression", NVAR(expected, c), NVAR(position, ptr));
            }
            if (expression[ptr] != c) {
                THROW("Unexpected symbol", NVAR(expected, c), NVAR(got, expression[ptr]), NVAR(position, ptr));
            }
            ++ptr;
        };

        auto parseInt = [&] {
            if (ptr == expression.size()) {
                THROW("Unexpected end of expression, expected integer");
            }
            if (!std::isdigit(expression[ptr])) {
                THROW("Unexpected end of expression, expected integer", NVAR(got, expression[ptr]), NVAR(position, ptr));
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
                THROW("Unexpected end of expression");
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
                THROW("Unexpected end of expression");
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
                THROW("Unknown value type", VAR(valueType));
            }
        };

        const std::string Result = "result";
        if (expression.substr(0, Result.size()) == Result) {
            ptr += Result.size();
            skip('=');
            outputNodeIndex = parseInt();
            if (ptr < expression.size()) {
                THROW("Unexpected symbols after the end of expression", NVAR(suffix, expression.substr(ptr)));
            }

            continue;
        }

        TNodeBasePtr node;

        auto nodeIndex = parseInt();
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
                THROW("Constant not found", VAR(name));
            }

            const auto& constant = constantIt->second;
            node = std::make_shared<TConstantNode>(TTensorMeta{type, shape}, constant);
        } else if (nodeType == "SumNode") {
            auto lhs = parseInt();
            skip(',');
            auto rhs = parseInt();

            auto lhsIt = nodes.find(lhs);
            if (lhsIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(lhs));
            }
            auto rhsIt = nodes.find(rhs);
            if (rhsIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(rhs));
            }

            node = std::make_shared<TSumNode>(lhsIt->second, rhsIt->second);
        } else if (nodeType == "MulNode") {
            auto lhs = parseInt();
            skip(',');
            auto rhs = parseInt();

            auto lhsIt = nodes.find(lhs);
            if (lhsIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(lhs));
            }
            auto rhsIt = nodes.find(rhs);
            if (rhsIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(rhs));
            }

            node = std::make_shared<TMulNode>(lhsIt->second, rhsIt->second);
        } else if (nodeType == "ReLUNode") {
            auto input = parseInt();

            auto inputIt = nodes.find(input);
            if (inputIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(input));
            }

            node = std::make_shared<TReLUNode>(inputIt->second);
        } else if (nodeType == "SiLUNode") {
            auto input = parseInt();

            auto inputIt = nodes.find(input);
            if (inputIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex), VAR(input));
            }

            node = std::make_shared<TSiLUNode>(inputIt->second);
        } else {
            THROW("Unknown node type", VAR(nodeIndex), VAR(nodeType));
        }

        skip(')');

        if (ptr < expression.size()) {
            THROW("Unexpected symbols after the end of expression", NVAR(suffix, expression.substr(ptr)));
        }

        if (!nodes.emplace(nodeIndex, node).second) {
            THROW("Node is defined twice", VAR(nodeIndex));
        }
    }

    if (!outputNodeIndex) {
        THROW("Output node index is not defined");
    }

    auto outputNodeIt = nodes.find(*outputNodeIndex);
    if (outputNodeIt == nodes.end()) {
        THROW("Output node references unknown node", VAR(*outputNodeIndex));
    }

    return outputNodeIt->second;
}
