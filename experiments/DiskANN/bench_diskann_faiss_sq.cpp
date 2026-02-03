/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark DiskANN with FAISS ScalarQuantizer on SIFT1M dataset.
 *
 * This benchmark:
 * 1. Loads SIFT data and queries
 * 2. Trains FAISS IndexScalarQuantizer on the base data
 * 3. Loads a pre-built DiskANN disk index
 * 4. Uses FAISS SQ for distance estimation instead of DiskANN's PQ
 * 5. Measures search performance and recall
 *
 * Usage:
 *   ./bench_diskann_faiss_sq [data_path] [index_prefix]
 *
 * Default:
 *   data_path: /data/local/embedding_dataset/sift1M
 *   index_prefix: {data_path}/index/diskann_R64_L100/diskann_R64_L100
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/stat.h>

// FAISS headers
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/DistanceComputer.h>

// DiskANN headers
#include "linux_aligned_file_reader.h"
#include "pq_flash_index.h"
#include "utils.h"

/*****************************************************
 * I/O functions for fvecs and ivecs (same as bench_rabitq_sift.cpp)
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

/*****************************************************
 * Timing utilities
 *****************************************************/

class Timer {
   public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

   private:
    std::chrono::high_resolution_clock::time_point start_;
};

/*****************************************************
 * Recall computation
 *****************************************************/

float compute_recall_at_k(
        size_t nq,
        size_t k,
        const uint64_t* I,
        const int* gt,
        size_t gt_k) {
    int64_t total_hits = 0;

    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < k; j++) {
            uint64_t result_id = I[i * k + j];
            for (size_t l = 0; l < k && l < gt_k; l++) {
                if (result_id == (uint64_t)gt[i * gt_k + l]) {
                    total_hits++;
                    break;
                }
            }
        }
    }

    return total_hits / float(nq * k);
}

/*****************************************************
 * Main
 *****************************************************/

int main(int argc, char* argv[]) {
    omp_set_num_threads(16);

    std::string data_path = "/data/local/embedding_dataset/sift1M";
    std::string index_prefix = "";

    if (argc > 1) {
        data_path = argv[1];
    }
    if (argc > 2) {
        index_prefix = argv[2];
    }

    // Default index prefix
    if (index_prefix.empty()) {
        index_prefix = data_path + "/index/diskann_R64_L100/diskann_R64_L100";
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "DiskANN + FAISS ScalarQuantizer Benchmark on SIFT1M" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "Index prefix: " << index_prefix << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset
    std::cout << "\n[Loading data...]" << std::endl;

    size_t d, nb, nq, k_gt;
    float* xb = fvecs_read((data_path + "/sift_base.fvecs").c_str(), &d, &nb);
    float* xq = fvecs_read((data_path + "/sift_query.fvecs").c_str(), &d, &nq);
    int* gt = ivecs_read((data_path + "/sift_groundtruth.ivecs").c_str(), &k_gt, &nq);

    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;
    std::cout << "Queries: " << nq << " vectors" << std::endl;
    std::cout << "Ground truth k: " << k_gt << std::endl;

    // Train FAISS ScalarQuantizer
    std::cout << "\n[Training FAISS ScalarQuantizer...]" << std::endl;
    Timer timer;

    faiss::IndexScalarQuantizer faiss_sq_index(d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    faiss_sq_index.train(nb, xb);
    std::cout << "SQ training time: " << timer.elapsed_ms() << " ms" << std::endl;

    timer.reset();
    faiss_sq_index.add(nb, xb);
    std::cout << "SQ add time: " << timer.elapsed_ms() << " ms" << std::endl;
    std::cout << "SQ code_size: " << faiss_sq_index.code_size << " bytes" << std::endl;

    // Load DiskANN index
    std::cout << "\n[Loading DiskANN disk index...]" << std::endl;

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<float>> flash_index(
        new diskann::PQFlashIndex<float>(reader, diskann::Metric::L2));

    timer.reset();
    int load_result = flash_index->load(omp_get_max_threads(), index_prefix.c_str());
    if (load_result != 0) {
        std::cerr << "Error loading DiskANN index: " << load_result << std::endl;
        return load_result;
    }
    std::cout << "DiskANN index load time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Cache some nodes for faster search
    std::vector<uint32_t> node_list;
    uint64_t num_nodes_to_cache = 10000;
    flash_index->cache_bfs_levels(num_nodes_to_cache, node_list);
    flash_index->load_cache_list(node_list);
    std::cout << "Cached " << num_nodes_to_cache << " nodes" << std::endl;

    // Set FAISS index for distance estimation
    std::cout << "\n[Setting FAISS SQ for distance estimation...]" << std::endl;
    flash_index->set_faiss_index(&faiss_sq_index);

    // Search parameters
    const size_t k = 10;  // recall@10
    const uint32_t beam_width = 4;
    std::vector<uint32_t> L_values = {10, 20, 30, 50, 70, 100, 150, 200};

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Search Results (DiskANN + FAISS SQ)" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::setw(8) << "L"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall@10"
              << std::setw(15) << "Latency(ms)" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    for (uint32_t L : L_values) {
        if (L < k) continue;

        std::vector<uint64_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);

        // Warmup
        for (size_t i = 0; i < std::min((size_t)100, nq); i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                beam_width, false, nullptr);
        }

        // Timed search
        timer.reset();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)nq; i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                beam_width, false, nullptr);
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, k_gt);

        std::cout << std::setw(8) << L
                  << std::setw(15) << std::fixed << std::setprecision(0) << qps
                  << std::setw(15) << std::setprecision(4) << recall
                  << std::setw(15) << std::setprecision(3) << latency << std::endl;
    }

    // Compare with original PQ (disable FAISS)
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Search Results (DiskANN Original PQ)" << std::endl;
    std::cout << "==================================================" << std::endl;

    flash_index->set_faiss_index(nullptr);  // Disable FAISS, use original PQ
    std::cout << std::setw(8) << "L"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall@10"
              << std::setw(15) << "Latency(ms)" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    for (uint32_t L : L_values) {
        if (L < k) continue;

        std::vector<uint64_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);

        // Warmup
        for (size_t i = 0; i < std::min((size_t)100, nq); i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                beam_width, false, nullptr);
        }

        // Timed search
        timer.reset();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)nq; i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                beam_width, false, nullptr);
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, k_gt);

        std::cout << std::setw(8) << L
                  << std::setw(15) << std::fixed << std::setprecision(0) << qps
                  << std::setw(15) << std::setprecision(4) << recall
                  << std::setw(15) << std::setprecision(3) << latency << std::endl;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
