#pragma once

#include <string>

namespace fmt {

template <typename... Args>
std::string format(const std::string& fmt_str, Args&&...) {
    return fmt_str;
}

template <typename... Args>
std::string format(const char* fmt_str, Args&&...) {
    return std::string(fmt_str);
}

} // namespace fmt
