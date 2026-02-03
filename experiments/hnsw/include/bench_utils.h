/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

namespace hnsw_bench {

/*****************************************************
 * Timing utilities
 *****************************************************/

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

inline float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        perror("");
        return nullptr;
    }

    int d;
    size_t nr = fread(&d, sizeof(int), 1, f);
    if (nr != 1 || d <= 0 || d >= 1000000) {
        fprintf(stderr, "Invalid dimension in %s\n", fname);
        fclose(f);
        return nullptr;
    }

    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;

    if (sz % ((d + 1) * sizeof(float)) != 0) {
        fprintf(stderr, "Invalid file size in %s\n", fname);
        fclose(f);
        return nullptr;
    }

    size_t n = sz / ((d + 1) * sizeof(float));

    *d_out = d;
    *n_out = n;

    float* x = new float[n * (d + 1)];
    nr = fread(x, sizeof(float), n * (d + 1), f);
    if (nr != n * (d + 1)) {
        fprintf(stderr, "Could not read all data from %s\n", fname);
        delete[] x;
        fclose(f);
        return nullptr;
    }

    // Shift array to remove row headers
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(float));
    }

    fclose(f);
    return x;
}

inline int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

/*****************************************************
 * Recall computation
 *****************************************************/

inline float compute_recall_at_k(
        size_t nq,
        size_t k,
        const faiss::idx_t* I,
        const int* gt,
        size_t gt_k) {
    int64_t total_hits = 0;
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < k; j++) {
            faiss::idx_t result_id = I[i * k + j];
            for (size_t l = 0; l < k && l < gt_k; l++) {
                if (result_id == (faiss::idx_t)gt[i * gt_k + l]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return total_hits / float(nq * k);
}

/*****************************************************
 * HNSW index loading/building
 *****************************************************/

inline faiss::IndexHNSW* load_or_build_hnsw(
        const std::string& index_path,
        size_t d,
        int M,
        int efConstruction,
        size_t nb,
        const float* xb,
        bool verbose = true) {

    // Try to load existing index
    FILE* f = fopen(index_path.c_str(), "rb");
    if (f) {
        fclose(f);
        if (verbose) {
            std::cout << "[Loading HNSW index from " << index_path << "...]" << std::endl;
        }
        Timer timer;
        std::unique_ptr<faiss::Index> loaded_index(faiss::read_index(index_path.c_str()));
        faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(loaded_index.get());
        if (hnsw) {
            loaded_index.release();
            if (verbose) {
                std::cout << "HNSW load time: " << timer.elapsed_ms() << " ms" << std::endl;
            }
            return hnsw;
        }
    }

    // Build new index
    if (verbose) {
        std::cout << "[Building HNSW index (M=" << M << ", efConstruction=" << efConstruction << ")...]" << std::endl;
    }

    Timer timer;
    auto* hnsw = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_L2);
    hnsw->hnsw.efConstruction = efConstruction;
    hnsw->add(nb, xb);

    if (verbose) {
        std::cout << "HNSW build time: " << timer.elapsed_sec() << " seconds" << std::endl;
    }

    // Save index
    // Create directory if needed
    size_t last_slash = index_path.rfind('/');
    if (last_slash != std::string::npos) {
        std::string dir = index_path.substr(0, last_slash);
        std::string cmd = "mkdir -p " + dir;
        int ret = system(cmd.c_str());
        (void)ret;
    }

    if (verbose) {
        std::cout << "[Saving HNSW index to " << index_path << "...]" << std::endl;
    }
    timer.reset();
    faiss::write_index(hnsw, index_path.c_str());
    if (verbose) {
        std::cout << "HNSW save time: " << timer.elapsed_sec() << " seconds" << std::endl;
    }

    return hnsw;
}

/*****************************************************
 * File existence check
 *****************************************************/

inline bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

} // namespace hnsw_bench
