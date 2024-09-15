#pragma once

#include "node.h"

#include <string>
#include <unordered_map>

namespace NHamKaas {

TNodeBasePtr ParseScript(const std::string& script);

} // namespace NHamKaas
