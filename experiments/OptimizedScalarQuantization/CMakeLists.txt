cmake_minimum_required(VERSION 3.16)
project(OptimizedScalarQuantization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(osq
  OptimizedScalarQuantizer.cpp
  OSQIndex.cpp
)

target_include_directories(osq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(osq PRIVATE -O3 -march=native -DNDEBUG)

add_executable(osq_example example_main.cpp)
target_link_libraries(osq_example PRIVATE osq)
target_compile_options(osq_example PRIVATE -O3 -march=native -DNDEBUG)
