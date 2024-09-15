#include "parser.h"

#include "error.h"

#include <optional>
#include <vector>

namespace NHamKaas {

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

TNodeBasePtr ParseScript(const std::string& script)
{
    std::unordered_map<int, TNodeBasePtr> nodes;
    TNodeBasePtr outputNode;

    for (const auto& expression : PreprocessScript(script)) {
        int ptr = 0;

        auto skip = [&] (char c) {
            if (ptr == expression.size()) {
                THROW("Unexpected end of expression", VAR(expected, c), VAR(position, ptr));
            }
            if (expression[ptr] != c) {
                THROW("Unexpected symbol", VAR(expected, c), VAR(got, expression[ptr]), VAR(position, ptr), VAR(expression));
            }
            ++ptr;
        };

        auto parseInt = [&] {
            if (ptr == expression.size()) {
                THROW("Unexpected end of expression, expected integer");
            }
            if (!std::isdigit(expression[ptr])) {
                THROW("Unexpected end of expression, expected integer", VAR(got, expression[ptr]), VAR(position, ptr));
            }

            int64_t result = 0;
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

            std::vector<int64_t> result;
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
            if (valueType == "float32") {
                return EValueType::Float32;
            } else if (valueType == "float64") {
                return EValueType::Float64;
            } else if (valueType == "int64") {
                return EValueType::Int64;
            } else {      
                THROW("Unknown value type", VAR(valueType));
            }
        };

        auto parseNodeIndex = [&] {
            skip('$');
            return parseInt();
        };

        auto parseNodeRef = [&] {
            auto nodeIndex = parseNodeIndex();
            auto nodeIt = nodes.find(nodeIndex);
            if (nodeIt == nodes.end()) {
                THROW("Expression references unknown node", VAR(nodeIndex));
            }

            return nodeIt->second;
        };

        const std::string Result = "result";
        if (expression.substr(0, Result.size()) == Result) {
            ptr += Result.size();
            skip('=');
            outputNode = parseNodeRef();
            if (ptr < expression.size()) {
                THROW("Unexpected symbols after the end of expression", VAR(suffix, expression.substr(ptr)));
            }

            continue;
        }

        TNodeBasePtr node;

        auto nodeIndex = parseNodeIndex();
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

            node = std::make_shared<TInputNode>(name, TTensorMeta{type, std::move(shape)});
        } else if (nodeType == "ConstantTensor") {
            auto name = parseStringArg();
            skip(',');
            auto type = parseValueTypeArg();
            skip(',');
            auto shape = parseIntList();

            node = std::make_shared<TConstantNode>(TTensorMeta{type, std::move(shape)}, name);
        } else if (nodeType == "BufferTensor") {
            auto type = parseValueTypeArg();
            skip(',');
            auto shape = parseIntList();

            node = std::make_shared<TBufferNode>(TTensorMeta{type, std::move(shape)});
        } else if (nodeType == "SumNode") {
            auto lhs = parseNodeRef();
            skip(',');
            auto rhs = parseNodeRef();

            node = std::make_shared<TSumNode>(std::move(lhs), std::move(rhs));
        } else if (nodeType == "MatMulNode") {
            auto lhs = parseNodeRef();
            skip(',');
            auto rhs = parseNodeRef();
        
            node = std::make_shared<TMatMulNode>(std::move(lhs), std::move(rhs));
        } else if (nodeType == "ReLUNode") {
            auto input = parseNodeRef();

            node = std::make_shared<TReLUNode>(std::move(input));
        } else if (nodeType == "SiLUNode") {
            auto input = parseNodeRef();

            node = std::make_shared<TSiLUNode>(std::move(input));
        } else if (nodeType == "SliceNode") {
            auto input = parseNodeRef();
            skip(',');
            auto begin = parseInt();
            skip(',');
            auto end = parseInt();

            node = std::make_shared<TSliceNode>(std::move(input), begin, end);
        } else if (nodeType == "ReshapeNode") {
            auto input = parseNodeRef();
            skip(',');
            auto shape = parseIntList();

            node = std::make_shared<TReshapeNode>(std::move(input), std::move(shape));
        } else if (nodeType == "PermuteNode") {
            auto input = parseNodeRef();
            skip(',');
            auto permutation = parseIntList();

            node = std::make_shared<TPermuteNode>(std::move(input), std::move(permutation));
        } else if (nodeType == "ReplaceSliceNode") {
            auto input = parseNodeRef();
            skip(',');
            auto replacement = parseNodeRef();
            skip(',');
            auto begin = parseNodeRef();
            skip(',');
            auto end = parseNodeRef();

            node = std::make_shared<TReplaceSliceNode>(std::move(input), std::move(replacement), std::move(begin), std::move(end));
        } else {
            THROW("Unknown node type", VAR(nodeType));
        }

        skip(')');

        if (ptr < expression.size()) {
            THROW("Unexpected symbols after the end of expression", VAR(suffix, expression.substr(ptr)));
        }

        if (!nodes.emplace(nodeIndex, node).second) {
            THROW("Node is defined twice", VAR(nodeIndex));
        }
    }

    if (!outputNode) {
        THROW("Output node is not defined");
    }

    return outputNode;
}

} // namespace NHamKaas
