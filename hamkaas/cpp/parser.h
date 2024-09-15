#pragma once

#include "node.h"

#include <string>
#include <unordered_map>

struct TScript
{
    std::string Script;

    std::unordered_map<std::string, const void*> Constants;
};

TNodeBasePtr ParseScript(const TScript& script);
