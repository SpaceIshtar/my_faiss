/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark HNSW with quantization-based search + exact distance reranking.
 *
 * This implements:
 * 1. Build: HNSW with exact distances (IndexHNSWFlat)
 * 2. Search: Use quantization (SQ) for graph traversal distance estimation
 * 3. Rerank: Use exact distances for final top-k selection
 *
 * Usage:
 *   ./bench_hnsw_quant_rerank [data_path] [index_path]
 */

#include <algorithm>
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

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>

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
 * HNSW search with quantization + rerank
 *
 * 1. Use quant_dc for HNSW graph traversal (get top-ef candidates)
 * 2. Rerank candidates using exact_dc (from HNSW's storage)
 *****************************************************/

void hnsw_search_with_rerank(
        const faiss::IndexHNSW* hnsw_index,
        faiss::FlatCodesDistanceComputer* quant_dc,
        faiss::DistanceComputer* exact_dc,
        const float* query,
        size_t k,
        size_t ef,
        float* distances,
        faiss::idx_t* labels) {
    const faiss::HNSW& hnsw = hnsw_index->hnsw;

    // Set query for both distance computers
    quant_dc->set_query(query);
    exact_dc->set_query(query);

    // Prepare heap for ef candidates using HeapBlockResultHandler
    // This is how FAISS internally uses the result handler for HNSW search
    using C = faiss::CMax<float, faiss::idx_t>;
    std::vector<float> ef_distances(ef, std::numeric_limits<float>::max());
    std::vector<faiss::idx_t> ef_labels(ef, -1);

    // Create block result handler for a single query
    faiss::HeapBlockResultHandler<C, false> bres(1, ef_distances.data(), ef_labels.data(), ef);

    // Create single result handler from block handler
    typename faiss::HeapBlockResultHandler<C, false>::SingleResultHandler res(bres);

    // Use a VisitedTable for the search
    faiss::VisitedTable vt(hnsw_index->ntotal);

    // Search parameters
    faiss::SearchParametersHNSW params;
    params.efSearch = ef;

    // Begin search for query 0
    res.begin(0);

    // HNSW search using quantized distances
    hnsw.search(*quant_dc, hnsw_index, res, vt, &params);

    // End search (this reorders the heap)
    res.end();


    // Rerank using exact distances (one by one, no nested parallelism)
    size_t n_candidates = ef_labels.size();
    std::vector<std::pair<float, faiss::idx_t>> reranked(n_candidates);
    for (size_t i = 0; i < n_candidates; i++) {
        float dist = (*exact_dc)(ef_labels[i]);
        reranked[i] = {dist, ef_labels[i]};
    }

    // Sort candidates by exact distance
    std::sort(reranked.begin(), reranked.end());

    // Return top-k
    for (size_t i = 0; i < k; i++) {
        if (i < reranked.size()) {
            distances[i] = reranked[i].first;
            labels[i] = reranked[i].second;
        } else {
            distances[i] = std::numeric_limits<float>::max();
            labels[i] = -1;
        }
    }
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
    std::cout << "HNSW + Quantization + Rerank Benchmark on SIFT1M" << std::endl;
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

    // Get the flat storage for reranking
    faiss::IndexFlat* flat_index = dynamic_cast<faiss::IndexFlat*>(hnsw_index->storage);
    if (!flat_index) {
        std::cerr << "Error: HNSW storage is not IndexFlat" << std::endl;
        return 1;
    }

    // Train ScalarQuantizer index
    std::cout << "\n[Training ScalarQuantizer...]" << std::endl;
    timer.reset();
    faiss::IndexScalarQuantizer sq_index(d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    sq_index.train(nb, xb);
    sq_index.add(nb, xb);
    std::cout << "SQ train+add time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Search parameters
    const size_t k = 10;
    std::vector<size_t> ef_values = {10, 20, 30, 40, 50, 75, 100};

    // ==========================================
    // Benchmark: HNSW + SQ search + exact rerank
    // ==========================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "HNSW + SQ Search + Exact Rerank" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::setw(8) << "ef"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall@10"
              << std::setw(15) << "Latency(ms)" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    for (size_t ef : ef_values) {
        std::vector<faiss::idx_t> result_ids(nq * k, 0);
        std::vector<float> result_dists(nq * k, std::numeric_limits<float>::max());

        // Warmup
        for (size_t i = 0; i < std::min((size_t)10, nq); i++) {
            std::unique_ptr<faiss::FlatCodesDistanceComputer> quant_dc(
                sq_index.get_FlatCodesDistanceComputer());
            std::unique_ptr<faiss::DistanceComputer> exact_dc(
                flat_index->get_distance_computer());
            hnsw_search_with_rerank(
                hnsw_index, quant_dc.get(), exact_dc.get(),
                xq + i * d, k, ef,
                result_dists.data() + i * k,
                result_ids.data() + i * k);
        }

        // Timed search
        timer.reset();

        // Debug: print actual thread count (only for first ef)
        if (ef == ef_values[0]) {
            std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;
        }

        #pragma omp parallel
        {
            // Each thread creates its own distance computers (once per thread)
            std::unique_ptr<faiss::FlatCodesDistanceComputer> quant_dc(
                sq_index.get_FlatCodesDistanceComputer());
            std::unique_ptr<faiss::DistanceComputer> exact_dc(
                flat_index->get_distance_computer());

            #pragma omp for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)nq; i++) {
                hnsw_search_with_rerank(
                    hnsw_index, quant_dc.get(), exact_dc.get(),
                    xq + i * d, k, ef,
                    result_dists.data() + i * k,
                    result_ids.data() + i * k);
            }
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, k_gt);

        std::cout << std::setw(8) << ef
                  << std::setw(15) << std::fixed << std::setprecision(0) << qps
                  << std::setw(15) << std::setprecision(4) << recall
                  << std::setw(15) << std::setprecision(3) << latency << std::endl;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
