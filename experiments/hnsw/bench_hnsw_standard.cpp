/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark standard HNSW with exact distances.
 *
 * Usage:
 *   ./bench_hnsw_standard [data_path] [index_path]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/stat.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

/*****************************************************
 * I/O functions for fvecs and ivecs
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
 * Result storage
 *****************************************************/

struct BenchmarkResult {
    size_t ef;
    double qps;
    float recall;
    double latency_ms;
};

void create_directory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    system(cmd.c_str());
}

// Extract dataset name from path (e.g., "/data/local/embedding_dataset/sift1M" -> "sift1M")
std::string get_dataset_name(const std::string& path) {
    size_t pos = path.rfind('/');
    if (pos != std::string::npos && pos + 1 < path.size()) {
        return path.substr(pos + 1);
    }
    return path;
}

// Extract M and efConstruction from index path (e.g., "hnsw_M32_ef200.faissindex")
std::pair<int, int> parse_index_params(const std::string& index_path) {
    int M = 32, efC = 200;  // defaults

    // Find "M" followed by digits
    size_t m_pos = index_path.find("_M");
    if (m_pos != std::string::npos) {
        m_pos += 2;  // skip "_M"
        size_t end = m_pos;
        while (end < index_path.size() && std::isdigit(index_path[end])) end++;
        if (end > m_pos) {
            M = std::stoi(index_path.substr(m_pos, end - m_pos));
        }
    }

    // Find "ef" followed by digits
    size_t ef_pos = index_path.find("_ef");
    if (ef_pos != std::string::npos) {
        ef_pos += 3;  // skip "_ef"
        size_t end = ef_pos;
        while (end < index_path.size() && std::isdigit(index_path[end])) end++;
        if (end > ef_pos) {
            efC = std::stoi(index_path.substr(ef_pos, end - ef_pos));
        }
    }

    return {M, efC};
}

/*****************************************************
 * Main
 *****************************************************/

int main(int argc, char* argv[]) {
    omp_set_num_threads(16);

    std::string data_path = "/data/local/embedding_dataset/sift1M";
    std::string index_path = "";

    if (argc > 1) {
        data_path = argv[1];
    }
    if (argc > 2) {
        index_path = argv[2];
    }

    if (index_path.empty()) {
        index_path = data_path + "/index/hnsw_M32_ef200.faissindex";
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "Standard HNSW Benchmark (Exact Distances)" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "Index path: " << index_path << std::endl;
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

    // Load HNSW index
    std::cout << "\n[Loading HNSW index...]" << std::endl;
    Timer timer;
    std::unique_ptr<faiss::Index> loaded_index(faiss::read_index(index_path.c_str()));
    faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(loaded_index.get());
    if (!hnsw_index) {
        std::cerr << "Error: Index is not an IndexHNSW" << std::endl;
        return 1;
    }
    std::cout << "HNSW load time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Parse HNSW parameters from index path
    auto [hnsw_M, hnsw_efC] = parse_index_params(index_path);
    std::string dataset_name = get_dataset_name(data_path);

    // Search parameters
    const size_t k = 10;
    std::vector<size_t> ef_values = {10, 20, 30, 40, 50, 75, 100, 150, 200};
    std::vector<BenchmarkResult> results;

    // ==========================================
    // Benchmark: Standard HNSW (exact distances)
    // ==========================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Standard HNSW Search Results" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::setw(8) << "ef"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall@10"
              << std::setw(15) << "Latency(ms)" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    for (size_t ef : ef_values) {
        std::vector<faiss::idx_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);

        faiss::SearchParametersHNSW params;
        params.efSearch = ef;

        // Warmup
        hnsw_index->search(std::min((faiss::idx_t)100, (faiss::idx_t)nq),
                          xq, k, result_dists.data(), result_ids.data(), &params);

        // Timed search
        timer.reset();
        hnsw_index->search(nq, xq, k, result_dists.data(), result_ids.data(), &params);

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, k_gt);

        results.push_back({ef, qps, recall, latency});

        std::cout << std::setw(8) << ef
                  << std::setw(15) << std::fixed << std::setprecision(0) << qps
                  << std::setw(15) << std::setprecision(4) << recall
                  << std::setw(15) << std::setprecision(3) << latency << std::endl;
    }

    // Save results to file
    std::string result_dir = "experiments/hnsw/results/" + dataset_name + "/standard";
    create_directory(result_dir);
    std::string result_file = result_dir + "/M" + std::to_string(hnsw_M) +
        "_efc" + std::to_string(hnsw_efC) + ".txt";

    std::ofstream ofs(result_file);
    if (ofs.is_open()) {
        ofs << "# Standard HNSW Benchmark (Exact Distances)\n"
            << "# Dataset: " << dataset_name << "\n"
            << "# HNSW M: " << hnsw_M << ", efConstruction: " << hnsw_efC << "\n"
            << "# k: " << k << "\n\n"
            << std::setw(8) << "ef"
            << std::setw(15) << "QPS"
            << std::setw(15) << "Recall"
            << std::setw(15) << "Latency(ms)" << "\n";

        for (const auto& r : results) {
            ofs << std::setw(8) << r.ef
                << std::setw(15) << std::fixed << std::setprecision(0) << r.qps
                << std::setw(15) << std::setprecision(4) << r.recall
                << std::setw(15) << std::setprecision(3) << r.latency_ms << "\n";
        }
        ofs.close();
        std::cout << "\nResults saved to: " << result_file << std::endl;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
