# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HNSW experiments - quantization-based search with reranking

find_package(OpenMP REQUIRED)

# Include directory for headers
set(HNSW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Build HNSW index
add_executable(build_hnsw_index EXCLUDE_FROM_ALL build_hnsw_index.cpp)
target_link_libraries(build_hnsw_index PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark HNSW with quantization + rerank (old version, kept for reference)
add_executable(bench_hnsw_quant_rerank EXCLUDE_FROM_ALL bench_hnsw_quant_rerank.cpp)
target_link_libraries(bench_hnsw_quant_rerank PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark standard HNSW (exact distances)
add_executable(bench_hnsw_standard EXCLUDE_FROM_ALL bench_hnsw_standard.cpp)
target_link_libraries(bench_hnsw_standard PRIVATE faiss OpenMP::OpenMP_CXX)

# ============================================================
# New benchmark framework with configurable quantization methods
# ============================================================
add_executable(bench_hnsw_quant EXCLUDE_FROM_ALL bench_hnsw_quant.cpp)
target_include_directories(bench_hnsw_quant PRIVATE ${HNSW_INCLUDE_DIR})
target_link_libraries(bench_hnsw_quant PRIVATE faiss OpenMP::OpenMP_CXX)

# Copy config files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
