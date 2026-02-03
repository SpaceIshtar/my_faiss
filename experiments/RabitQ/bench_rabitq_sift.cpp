/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark for RaBitQ indices on SIFT1M dataset.
 *
 * Tests the following index types:
 * - IndexRaBitQ
 * - IndexIVFRaBitQ
 * - IndexRaBitQFastScan
 * - IndexIVFRaBitQFastScan
 *
 * Usage:
 *   ./bench_rabitq_sift [data_path]
 *
 * Default data_path: /data/local/embedding_dataset/sift1M
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>

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
 * Recall@k = (number of results in top-k that appear in ground truth top-k) / k
 *****************************************************/

float compute_recall_at_k(
        faiss::idx_t nq,
        faiss::idx_t k,
        const faiss::idx_t* I,
        const faiss::idx_t* gt) {
    int64_t total_hits = 0;

    for (faiss::idx_t i = 0; i < nq; i++) {
        // For each query, count how many of the returned k results
        // appear in the ground truth top-k
        for (faiss::idx_t j = 0; j < k; j++) {
            faiss::idx_t result_id = I[i * k + j];
            for (faiss::idx_t l = 0; l < k; l++) {
                if (result_id == gt[i * k + l]) {
                    total_hits++;
                    break;
                }
            }
        }
    }

    return total_hits / float(nq * k);
}

/*****************************************************
 * Benchmark runner
 *****************************************************/

struct BenchmarkResult {
    std::string index_name;
    double train_time_ms;
    double add_time_ms;
    double search_time_ms;
    float recall;
    double qps; // queries per second
};

void print_header() {
    std::cout << std::setw(35) << std::left << "Index"
              << std::setw(12) << std::right << "Train(ms)"
              << std::setw(12) << "Add(ms)"
              << std::setw(12) << "Search(ms)"
              << std::setw(10) << "Recall"
              << std::setw(12) << "QPS" << std::endl;
    std::cout << std::string(93, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(35) << std::left << r.index_name
              << std::setw(12) << std::right << std::fixed << std::setprecision(1)
              << r.train_time_ms << std::setw(12) << r.add_time_ms
              << std::setw(12) << r.search_time_ms << std::setw(10)
              << std::setprecision(4) << r.recall
              << std::setw(12) << std::setprecision(0) << r.qps << std::endl;
}

BenchmarkResult benchmark_index(
        faiss::Index* index,
        const std::string& name,
        faiss::idx_t nb,
        const float* xb,
        faiss::idx_t nq,
        const float* xq,
        faiss::idx_t k,
        const faiss::idx_t* gt) {
    BenchmarkResult result;
    result.index_name = name;
    Timer timer;

    // Train
    timer.reset();
    index->train(nb, xb);
    result.train_time_ms = timer.elapsed_ms();

    // Add
    timer.reset();
    index->add(nb, xb);
    result.add_time_ms = timer.elapsed_ms();

    // Search
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    // Warm-up search
    index->search(nq, xq, k, D.data(), I.data());

    // Timed search
    timer.reset();
    index->search(nq, xq, k, D.data(), I.data());
    result.search_time_ms = timer.elapsed_ms();

    // Compute recall
    result.recall = compute_recall_at_k(nq, k, I.data(), gt);

    // Compute QPS
    result.qps = (nq * 1000.0) / result.search_time_ms;

    return result;
}

/*****************************************************
 * Main
 *****************************************************/

int main(int argc, char* argv[]) {
    // Set number of threads to 16
    omp_set_num_threads(16);

    std::string data_path = "/data/local/embedding_dataset/sift1M";
    if (argc > 1) {
        data_path = argv[1];
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "RaBitQ Benchmark on SIFT1M Dataset" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset
    std::cout << "\n[Loading data...]" << std::endl;

    size_t d, nb, nq, k;
    float* xb = fvecs_read((data_path + "/sift_base.fvecs").c_str(), &d, &nb);
    float* xq = fvecs_read((data_path + "/sift_query.fvecs").c_str(), &d, &nq);
    int* gt_int = ivecs_read(
            (data_path + "/sift_groundtruth.ivecs").c_str(), &k, &nq);

    // Convert ground truth to idx_t
    std::vector<faiss::idx_t> gt(k * nq);
    for (size_t i = 0; i < k * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;

    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;
    std::cout << "Queries: " << nq << " vectors" << std::endl;
    std::cout << "k (from ground truth): " << k << std::endl;

    const size_t nlist = 4096;  // Number of IVF clusters
    const size_t nprobe = 16;   // Number of clusters to probe

    std::vector<BenchmarkResult> results;

    // ==========================================
    // 1. IndexRaBitQ
    // ==========================================
    std::cout << "\n[Testing IndexRaBitQ...]" << std::endl;
    {
        faiss::IndexRaBitQ index(d, faiss::METRIC_L2);
        results.push_back(benchmark_index(
                &index, "IndexRaBitQ", nb, xb, nq, xq, k, gt.data()));
    }

    // Test with different nb_bits
    {
        faiss::IndexRaBitQ index(d, faiss::METRIC_L2, 4);
        results.push_back(benchmark_index(
                &index, "IndexRaBitQ (4-bit)", nb, xb, nq, xq, k, gt.data()));
    }

    // ==========================================
    // 2. IndexIVFRaBitQ
    // ==========================================
    std::cout << "\n[Testing IndexIVFRaBitQ...]" << std::endl;
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFRaBitQ index(&quantizer, d, nlist, faiss::METRIC_L2);
        index.nprobe = nprobe;

        results.push_back(benchmark_index(
                &index, "IndexIVFRaBitQ (nprobe=16)", nb, xb, nq, xq, k, gt.data()));
    }

    // Test with different nprobe values
    for (size_t np : {4, 32, 64}) {
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFRaBitQ index(&quantizer, d, nlist, faiss::METRIC_L2);
        index.nprobe = np;

        std::string name = "IndexIVFRaBitQ (nprobe=" + std::to_string(np) + ")";
        results.push_back(benchmark_index(
                &index, name, nb, xb, nq, xq, k, gt.data()));
    }

    // ==========================================
    // 3. IndexRaBitQFastScan
    // ==========================================
    std::cout << "\n[Testing IndexRaBitQFastScan...]" << std::endl;
    {
        faiss::IndexRaBitQFastScan index(d, faiss::METRIC_L2);
        results.push_back(benchmark_index(
                &index, "IndexRaBitQFastScan", nb, xb, nq, xq, k, gt.data()));
    }

    // Test with different nb_bits
    {
        faiss::IndexRaBitQFastScan index(d, faiss::METRIC_L2, 32, 4);
        results.push_back(benchmark_index(
                &index, "IndexRaBitQFastScan (4-bit)", nb, xb, nq, xq, k, gt.data()));
    }

    // ==========================================
    // 4. IndexIVFRaBitQFastScan
    // ==========================================
    std::cout << "\n[Testing IndexIVFRaBitQFastScan...]" << std::endl;
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFRaBitQFastScan index(
                &quantizer, d, nlist, faiss::METRIC_L2);
        index.nprobe = nprobe;

        results.push_back(benchmark_index(
                &index, "IndexIVFRaBitQFastScan (nprobe=16)", nb, xb, nq, xq, k, gt.data()));
    }

    // Test with different nprobe values
    for (size_t np : {4, 32, 64}) {
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFRaBitQFastScan index(
                &quantizer, d, nlist, faiss::METRIC_L2);
        index.nprobe = np;

        std::string name = "IndexIVFRaBitQFastScan (nprobe=" + std::to_string(np) + ")";
        results.push_back(benchmark_index(
                &index, name, nb, xb, nq, xq, k, gt.data()));
    }

    // ==========================================
    // Print summary
    // ==========================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Benchmark Summary" << std::endl;
    std::cout << "==================================================" << std::endl;

    print_header();
    for (const auto& r : results) {
        print_result(r);
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq;

    return 0;
}
