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

# SAQ library include directories
set(SAQ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SAQ-Library/include)
set(SAQ_CORE_INCLUDE_DIR ${SAQ_INCLUDE_DIR}/saqlib)

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
target_compile_options(build_hnsw_index PRIVATE -march=native -mfma)
target_link_libraries(build_hnsw_index PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# Benchmark HNSW with quantization + rerank (old version, kept for reference)
add_executable(bench_hnsw_quant_rerank EXCLUDE_FROM_ALL bench_hnsw_quant_rerank.cpp)
target_link_libraries(bench_hnsw_quant_rerank PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# Benchmark standard HNSW (exact distances)
add_executable(bench_hnsw_standard EXCLUDE_FROM_ALL bench_hnsw_standard.cpp)
target_include_directories(bench_hnsw_standard PRIVATE ${HNSW_INCLUDE_DIR})
target_compile_options(bench_hnsw_standard PRIVATE -march=native -mfma)
target_link_libraries(bench_hnsw_standard PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# ============================================================
# HNSW construction benchmark (quantized vs exact construction)
# ============================================================
add_executable(bench_construction_hnsw EXCLUDE_FROM_ALL bench_construction_hnsw.cpp)
target_include_directories(bench_construction_hnsw PRIVATE ${HNSW_INCLUDE_DIR})
target_compile_options(bench_construction_hnsw PRIVATE -march=native -mfma)
target_link_libraries(bench_construction_hnsw PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# ============================================================
# IndexHNSWPQ benchmark (PQ search + exact rerank via IndexRefine)
# ============================================================
add_executable(bench_hnswpq EXCLUDE_FROM_ALL bench_hnswpq.cpp)
target_include_directories(bench_hnswpq PRIVATE ${HNSW_INCLUDE_DIR})
target_link_libraries(bench_hnswpq PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# ============================================================
# New benchmark framework with configurable quantization methods
# ============================================================
add_executable(bench_hnsw_quant EXCLUDE_FROM_ALL bench_hnsw_quant.cpp)
target_include_directories(bench_hnsw_quant PRIVATE
    ${HNSW_INCLUDE_DIR}
    ${RABITQ_INCLUDE_DIR}
    ${SAQ_INCLUDE_DIR}
    ${SAQ_CORE_INCLUDE_DIR}
    ${VAQ_INCLUDE_DIR}
    ${VAQ_EIGEN_DIR}
)
target_link_libraries(bench_hnsw_quant PRIVATE faiss_avx512 -lglpk -larmadillo OpenMP::OpenMP_CXX)
target_compile_options(bench_hnsw_quant PRIVATE -march=native -mfma)
target_compile_definitions(bench_hnsw_quant PRIVATE VAQ_OPTIMIZE EIGEN_DONT_PARALLELIZE)

# ============================================================
# Cache profiling tool (train/profile modes for cachegrind)
# ============================================================
add_executable(bench_hnsw_quant_profile EXCLUDE_FROM_ALL bench_hnsw_quant_profile.cpp)
target_include_directories(bench_hnsw_quant_profile PRIVATE ${HNSW_INCLUDE_DIR})
target_link_libraries(bench_hnsw_quant_profile PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)

# ============================================================
# Native RaBitQ benchmark (authentic RaBitQ search with FAISS HNSW graph)
# ============================================================
add_executable(bench_hnsw_rabitq_native EXCLUDE_FROM_ALL bench_hnsw_rabitq_native.cpp)
target_include_directories(bench_hnsw_rabitq_native PRIVATE
    ${HNSW_INCLUDE_DIR}
    ${RABITQ_INCLUDE_DIR}
)
target_link_libraries(bench_hnsw_rabitq_native PRIVATE faiss_avx512 OpenMP::OpenMP_CXX)
target_compile_options(bench_hnsw_rabitq_native PRIVATE -march=native -mfma)  # Required for RaBitQ

add_custom_target(all_hnsw_benchmarks
    DEPENDS
        build_hnsw_index
        bench_hnsw_quant_rerank
        bench_hnsw_standard
        bench_construction_hnsw
        bench_hnswpq
        bench_hnsw_quant
        bench_hnsw_quant_profile
        bench_hnsw_rabitq_native
)

# Copy config files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
