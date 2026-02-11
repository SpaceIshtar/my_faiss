# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HNSW experiments - quantization-based search with reranking

find_package(OpenMP REQUIRED)

# Include directory for headers
set(HNSW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# RaBitQ library include directory
set(RABITQ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../RaBitQ-Library/include)

# VAQ library directories
set(VAQ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../VAQ)
set(VAQ_INCLUDE_DIR ${VAQ_DIR}/bitvecengine)
set(VAQ_EIGEN_DIR ${VAQ_DIR}/external/eigen)

# Build VAQ as a static library (avoids multiple-definition errors from
# non-inline functions in VAQ headers like Math.hpp, KMeans.hpp, etc.)
# add_library(vaq_lib STATIC
#     ${VAQ_DIR}/bitvecengine/VAQ.cpp
#     ${VAQ_DIR}/bitvecengine/utils/Heap.cpp
#     ${VAQ_DIR}/bitvecengine/utils/FPGrowth/fptree.cpp
# )
# target_include_directories(vaq_lib PUBLIC
#     ${VAQ_INCLUDE_DIR}
#     ${VAQ_EIGEN_DIR}
#     ${HNSW_INCLUDE_DIR}
# )
# target_compile_options(vaq_lib PRIVATE -mavx2 -mfma -O3 -ffast-math -Wno-deprecated-enum-enum-conversion)
# target_compile_definitions(vaq_lib PRIVATE VAQ_OPTIMIZE EIGEN_DONT_PARALLELIZE)
# target_link_libraries(vaq_lib PUBLIC faiss glpk armadillo blas OpenMP::OpenMP_CXX)

# Build HNSW index
add_executable(build_hnsw_index EXCLUDE_FROM_ALL build_hnsw_index.cpp)
target_include_directories(build_hnsw_index PRIVATE ${HNSW_INCLUDE_DIR})
target_link_libraries(build_hnsw_index PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark HNSW with quantization + rerank (old version, kept for reference)
add_executable(bench_hnsw_quant_rerank EXCLUDE_FROM_ALL bench_hnsw_quant_rerank.cpp)
target_link_libraries(bench_hnsw_quant_rerank PRIVATE faiss OpenMP::OpenMP_CXX)

# Benchmark standard HNSW (exact distances)
add_executable(bench_hnsw_standard EXCLUDE_FROM_ALL bench_hnsw_standard.cpp)
target_include_directories(bench_hnsw_standard PRIVATE ${HNSW_INCLUDE_DIR})
target_link_libraries(bench_hnsw_standard PRIVATE faiss OpenMP::OpenMP_CXX)

# ============================================================
# New benchmark framework with configurable quantization methods
# ============================================================
add_executable(bench_hnsw_quant EXCLUDE_FROM_ALL bench_hnsw_quant.cpp)
target_include_directories(bench_hnsw_quant PRIVATE
    ${HNSW_INCLUDE_DIR}
    ${RABITQ_INCLUDE_DIR}
    ${VAQ_INCLUDE_DIR}
    ${VAQ_EIGEN_DIR}
)
target_link_libraries(bench_hnsw_quant PRIVATE faiss -lglpk -larmadillo OpenMP::OpenMP_CXX)
target_compile_options(bench_hnsw_quant PRIVATE -mavx2 -mfma)
target_compile_definitions(bench_hnsw_quant PRIVATE VAQ_OPTIMIZE EIGEN_DONT_PARALLELIZE)

# ============================================================
# Native RaBitQ benchmark (authentic RaBitQ search with FAISS HNSW graph)
# ============================================================
add_executable(bench_hnsw_rabitq_native EXCLUDE_FROM_ALL bench_hnsw_rabitq_native.cpp)
target_include_directories(bench_hnsw_rabitq_native PRIVATE
    ${HNSW_INCLUDE_DIR}
    ${RABITQ_INCLUDE_DIR}
)
target_link_libraries(bench_hnsw_rabitq_native PRIVATE faiss OpenMP::OpenMP_CXX)
target_compile_options(bench_hnsw_rabitq_native PRIVATE -mavx2 -mfma)  # Required for RaBitQ

# Copy config files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
