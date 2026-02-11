# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# DiskANN experiments - integration with FAISS quantization

set(DISKANN_DIR ${CMAKE_SOURCE_DIR}/DiskANN)
set(DISKANN_INCLUDE_DIR ${DISKANN_DIR}/include)

# DiskANN source files needed for PQFlashIndex
set(DISKANN_SOURCES
    ${DISKANN_DIR}/src/abstract_data_store.cpp
    ${DISKANN_DIR}/src/ann_exception.cpp
    ${DISKANN_DIR}/src/disk_utils.cpp
    ${DISKANN_DIR}/src/distance.cpp
    ${DISKANN_DIR}/src/index.cpp
    ${DISKANN_DIR}/src/in_mem_graph_store.cpp
    ${DISKANN_DIR}/src/in_mem_data_store.cpp
    ${DISKANN_DIR}/src/linux_aligned_file_reader.cpp
    ${DISKANN_DIR}/src/math_utils.cpp
    ${DISKANN_DIR}/src/natural_number_map.cpp
    ${DISKANN_DIR}/src/natural_number_set.cpp
    ${DISKANN_DIR}/src/memory_mapper.cpp
    ${DISKANN_DIR}/src/partition.cpp
    ${DISKANN_DIR}/src/pq.cpp
    ${DISKANN_DIR}/src/pq_flash_index.cpp
    ${DISKANN_DIR}/src/scratch.cpp
    ${DISKANN_DIR}/src/logger.cpp
    ${DISKANN_DIR}/src/utils.cpp
    ${DISKANN_DIR}/src/filter_utils.cpp
    ${DISKANN_DIR}/src/index_factory.cpp
    ${DISKANN_DIR}/src/abstract_index.cpp
)

# Find MKL (required by DiskANN)
# find_package(MKL)
# if(NOT MKL_FOUND)
    # Try to find MKL in common locations
    set(POSSIBLE_MKL_PATHS
        "/common/home/jl3288/intel/oneapi/mkl/latest"
        "/usr/lib/x86_64-linux-gnu"
    )
    foreach(MKL_PATH ${POSSIBLE_MKL_PATHS})
        if(EXISTS "${MKL_PATH}/lib/intel64/libmkl_core.so" OR EXISTS "${MKL_PATH}/libmkl_core.so")
            set(MKL_FOUND TRUE)
            if(EXISTS "${MKL_PATH}/lib/intel64")
                set(MKL_LIBRARY_DIR "${MKL_PATH}/lib/intel64")
            else()
                set(MKL_LIBRARY_DIR "${MKL_PATH}")
            endif()
            if(EXISTS "${MKL_PATH}/include")
                set(MKL_INCLUDE_DIR "${MKL_PATH}/include")
            else()
                set(MKL_INCLUDE_DIR "/usr/include/mkl")
            endif()
            break()
        endif()
    endforeach()
# endif()

message(STATUS "MKL INCLUDE DIR: ${MKL_INCLUDE_DIR}")

# Find Boost
find_package(Boost COMPONENTS program_options)

# Common compile definitions for DiskANN
set(DISKANN_COMPILE_DEFS -DMKL_ILP64)

# Common compile options for DiskANN (SSE/AVX intrinsics)
set(DISKANN_COMPILE_OPTS -march=native)

# Common libraries for DiskANN on Linux
set(DISKANN_LINK_LIBS
    aio
    pthread
    m
    dl
)

if(MKL_FOUND)
    list(APPEND DISKANN_LINK_LIBS
        mkl_intel_ilp64
        mkl_intel_thread
        mkl_core
        iomp5
    )
endif()

# HNSW include dir (for shared bench_config.h from diskann_bench_config.h)
set(HNSW_EXP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../hnsw)

# Build DiskANN index builder
add_executable(build_diskann_index EXCLUDE_FROM_ALL build_diskann_index.cpp ${DISKANN_SOURCES})
target_include_directories(build_diskann_index PRIVATE
    ${DISKANN_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
    ${HNSW_EXP_DIR}
)
target_compile_definitions(build_diskann_index PRIVATE ${DISKANN_COMPILE_DEFS})
target_compile_options(build_diskann_index PRIVATE -fopenmp ${DISKANN_COMPILE_OPTS} -include xmmintrin.h)
target_link_libraries(build_diskann_index PRIVATE
    faiss
    ${DISKANN_LINK_LIBS}
)
if(MKL_FOUND AND MKL_LIBRARY_DIR)
    target_link_directories(build_diskann_index PRIVATE ${MKL_LIBRARY_DIR})
    # Also add Intel OpenMP library path
    target_link_directories(build_diskann_index PRIVATE
        /common/home/jl3288/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin)
endif()
if(Boost_FOUND)
    target_link_libraries(build_diskann_index PRIVATE Boost::program_options)
endif()

# Benchmark DiskANN with FAISS SQ
add_executable(bench_diskann_faiss_sq EXCLUDE_FROM_ALL bench_diskann_faiss_sq.cpp ${DISKANN_SOURCES})
target_include_directories(bench_diskann_faiss_sq PRIVATE
    ${DISKANN_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
)
target_compile_definitions(bench_diskann_faiss_sq PRIVATE ${DISKANN_COMPILE_DEFS})
target_compile_options(bench_diskann_faiss_sq PRIVATE -fopenmp ${DISKANN_COMPILE_OPTS} -include xmmintrin.h)
target_link_libraries(bench_diskann_faiss_sq PRIVATE
    faiss
    ${DISKANN_LINK_LIBS}
)
if(MKL_FOUND AND MKL_LIBRARY_DIR)
    target_link_directories(bench_diskann_faiss_sq PRIVATE ${MKL_LIBRARY_DIR})
    # Also add Intel OpenMP library path
    target_link_directories(bench_diskann_faiss_sq PRIVATE
        /common/home/jl3288/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin)
endif()
if(Boost_FOUND)
    target_link_libraries(bench_diskann_faiss_sq PRIVATE Boost::program_options)
endif()

# ============================================================
# Generic DiskANN + Quantization Benchmark Framework
# ============================================================

# RaBitQ library
set(RABITQ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../RaBitQ-Library/include)

# VAQ library
set(VAQ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../VAQ)
set(VAQ_INCLUDE_DIR ${VAQ_DIR}/bitvecengine)
set(VAQ_EIGEN_DIR ${VAQ_DIR}/external/eigen)

add_executable(bench_diskann_quant EXCLUDE_FROM_ALL bench_diskann_quant.cpp ${DISKANN_SOURCES})
target_include_directories(bench_diskann_quant PRIVATE
    ${DISKANN_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
    ${HNSW_EXP_DIR}
    ${RABITQ_INCLUDE_DIR}
    ${VAQ_INCLUDE_DIR}
    ${VAQ_EIGEN_DIR}
)
target_compile_definitions(bench_diskann_quant PRIVATE ${DISKANN_COMPILE_DEFS} VAQ_OPTIMIZE EIGEN_DONT_PARALLELIZE)
target_compile_options(bench_diskann_quant PRIVATE -fopenmp ${DISKANN_COMPILE_OPTS} -include xmmintrin.h -mavx2 -mfma)
target_link_libraries(bench_diskann_quant PRIVATE
    faiss
    -lglpk -larmadillo
    ${DISKANN_LINK_LIBS}
)
if(MKL_FOUND AND MKL_LIBRARY_DIR)
    target_link_directories(bench_diskann_quant PRIVATE ${MKL_LIBRARY_DIR})
    target_link_directories(bench_diskann_quant PRIVATE
        /common/home/jl3288/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin)
endif()
if(Boost_FOUND)
    target_link_libraries(bench_diskann_quant PRIVATE Boost::program_options)
endif()

# ============================================================
# Standard DiskANN Benchmark (Native PQ, no custom quantizers)
# ============================================================
add_executable(bench_diskann_standard EXCLUDE_FROM_ALL bench_diskann_standard.cpp ${DISKANN_SOURCES})
target_include_directories(bench_diskann_standard PRIVATE
    ${DISKANN_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
    ${HNSW_EXP_DIR}
)
target_compile_definitions(bench_diskann_standard PRIVATE ${DISKANN_COMPILE_DEFS})
target_compile_options(bench_diskann_standard PRIVATE -fopenmp ${DISKANN_COMPILE_OPTS} -include xmmintrin.h)
target_link_libraries(bench_diskann_standard PRIVATE
    faiss
    ${DISKANN_LINK_LIBS}
)
if(MKL_FOUND AND MKL_LIBRARY_DIR)
    target_link_directories(bench_diskann_standard PRIVATE ${MKL_LIBRARY_DIR})
    target_link_directories(bench_diskann_standard PRIVATE
        /common/home/jl3288/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin)
endif()
if(Boost_FOUND)
    target_link_libraries(bench_diskann_standard PRIVATE Boost::program_options)
endif()

# Copy config files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
