# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HNSW experiments - quantization-based search with reranking

find_package(OpenMP REQUIRED)

# Build HNSW index
add_executable(build_hnsw_index EXCLUDE_FROM_ALL build_hnsw_index.cpp)
target_link_libraries(build_hnsw_index PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark HNSW with quantization + rerank
add_executable(bench_hnsw_quant_rerank EXCLUDE_FROM_ALL bench_hnsw_quant_rerank.cpp)
target_link_libraries(bench_hnsw_quant_rerank PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark standard HNSW (exact distances)
add_executable(bench_hnsw_standard EXCLUDE_FROM_ALL bench_hnsw_standard.cpp)
target_link_libraries(bench_hnsw_standard PRIVATE faiss OpenMP::OpenMP_CXX)
