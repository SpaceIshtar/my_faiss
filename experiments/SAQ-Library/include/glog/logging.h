#pragma once

#include <cstdlib>
#include <iostream>

namespace saq_glog_stub {

class LogStream {
   public:
    LogStream(const char* level, const char* file, int line, bool enabled = true)
            : enabled_(enabled) {
        if (enabled_) {
            std::cerr << "[" << level << "] " << file << ":" << line << " ";
        }
    }

    ~LogStream() {
        if (enabled_) {
            std::cerr << std::endl;
        }
    }

    template <typename T>
    LogStream& operator<<(const T& v) {
        if (enabled_) {
            std::cerr << v;
        }
        return *this;
    }

   private:
    bool enabled_;
};

class CheckStream {
   public:
    CheckStream(bool ok, const char* expr, const char* file, int line)
            : ok_(ok) {
        if (!ok_) {
            std::cerr << "CHECK failed (" << expr << ") " << file << ":"
                      << line << " ";
        }
    }

    ~CheckStream() {
        if (!ok_) {
            std::cerr << std::endl;
            std::abort();
        }
    }

    template <typename T>
    CheckStream& operator<<(const T& v) {
        if (!ok_) {
            std::cerr << v;
        }
        return *this;
    }

   private:
    bool ok_;
};

} // namespace saq_glog_stub

#define LOG(level) ::saq_glog_stub::LogStream(#level, __FILE__, __LINE__)
#define LOG_IF(level, cond) \
    ::saq_glog_stub::LogStream(#level, __FILE__, __LINE__, static_cast<bool>(cond))
#define LOG_FIRST_N(level, n) \
    ::saq_glog_stub::LogStream(#level, __FILE__, __LINE__)

#define CHECK(cond) \
    ::saq_glog_stub::CheckStream(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define DCHECK(cond) CHECK(cond)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))

#define DCHECK_EQ(a, b) DCHECK((a) == (b))
#define DCHECK_NE(a, b) DCHECK((a) != (b))
#define DCHECK_LT(a, b) DCHECK((a) < (b))
#define DCHECK_LE(a, b) DCHECK((a) <= (b))
#define DCHECK_GT(a, b) DCHECK((a) > (b))
#define DCHECK_GE(a, b) DCHECK((a) >= (b))
