/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark FAISS IndexHNSWPQ with exact-distance reranking.
 *
 * Builds an IndexHNSWPQ (HNSW graph with PQ storage), then for each ef_search
 * value, searches ef candidates using PQ distances and reranks them with exact
 * L2 distances via IndexRefine.
 *
 * Usage:
 *   ./bench_hnswpq --dataset <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1m, gist1M)
 *   --config-dir <path>    Config directory (default: ./config)
 *   --data-path <path>     Override dataset base path
 *   --threads <n>          Number of threads
 *   --ef <list>            Comma-separated ef values to test
 *   --help                 Show this help
 *
 * Example:
 *   ./bench_hnswpq --dataset sift1m
 *   ./bench_hnswpq --dataset gist1M --threads 8
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexRefine.h>
#include <faiss/index_io.h>
#include <faiss/impl/HNSW.h>

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1m";
    std::string config_dir = "./config";
    std::string data_path;
    int threads = -1;
    std::vector<size_t> ef_values;
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>       Dataset name (e.g., sift1m, gist1M)\n"
              << "  --config-dir <path>    Config directory (default: ./config)\n"
              << "  --data-path <path>     Override dataset base path\n"
              << "  --threads <n>          Number of threads\n"
              << "  --ef <list>            Comma-separated ef values to test\n"
              << "  --help                 Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1m\n"
              << "  " << prog << " --dataset gist1M --ef 10,20,50,100,200\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            opts.dataset = argv[++i];
        } else if (arg == "--config-dir" && i + 1 < argc) {
            opts.config_dir = argv[++i];
        } else if (arg == "--data-path" && i + 1 < argc) {
            opts.data_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        } else if (arg == "--ef" && i + 1 < argc) {
            std::string ef_str = argv[++i];
            std::istringstream iss(ef_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                opts.ef_values.push_back(std::stoul(token));
            }
        }
    }

    return opts;
}

/*****************************************************
 * Result structures and output
 *****************************************************/

struct BenchmarkResult {
    size_t ef;
    double qps;
    float recall;
    double latency_ms;
    double total_time_ms;
    double ndis_mean;
    double nhops_mean;
};

void save_results_to_file(
        const std::string& filepath,
        const std::string& dataset_name,
        int pq_M,
        int pq_nbits,
        int hnsw_M,
        int hnsw_efConstruction,
        size_t nq,
        size_t k,
        const std::vector<BenchmarkResult>& results) {

    size_t last_slash = filepath.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(filepath.substr(0, last_slash));
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }

    ofs << "========================================\n"
        << "HNSW + PQ (IndexHNSWPQ) Benchmark Results\n"
        << "========================================\n\n";

    ofs << "[Configuration]\n"
        << "  Dataset:           " << dataset_name << "\n"
        << "  Algorithm:         IndexHNSWPQ\n"
        << "  PQ M:              " << pq_M << "\n"
        << "  PQ nbits:          " << pq_nbits << "\n"
        << "  HNSW M:            " << hnsw_M << "\n"
        << "  HNSW efConstruct:  " << hnsw_efConstruction << "\n"
        << "  Num Queries:       " << nq << "\n"
        << "  Recall@k:          " << k << "\n\n";

    ofs << "[Summary]\n"
        << std::setw(8)  << "ef"
        << std::setw(12) << "QPS"
        << std::setw(12) << "Recall"
        << std::setw(12) << "Latency(ms)"
        << std::setw(12) << "ndis_mean"
        << std::setw(12) << "nhops_mean"
        << "\n"
        << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        ofs << std::setw(8)  << r.ef
            << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
            << std::setw(12) << std::setprecision(4) << r.recall
            << std::setw(12) << std::setprecision(3) << r.latency_ms
            << std::setw(12) << std::setprecision(1) << r.ndis_mean
            << std::setw(12) << std::setprecision(1) << r.nhops_mean
            << "\n";
    }

    ofs << "\n========================================\n"
        << "End of Results\n"
        << "========================================\n";

    ofs.close();
    std::cout << "Results saved to: " << filepath << std::endl;
}

/*****************************************************
 * Load or build IndexHNSWPQ
 *****************************************************/

faiss::IndexHNSWPQ* load_or_build_hnswpq(
        const std::string& index_path,
        size_t d,
        int pq_M,
        int pq_nbits,
        int hnsw_M,
        int hnsw_efConstruction,
        size_t nb,
        const float* xb) {

    // Try to load existing index
    FILE* f = fopen(index_path.c_str(), "rb");
    if (f) {
        fclose(f);
        std::cout << "[Loading IndexHNSWPQ from " << index_path << "...]" << std::endl;
        Timer timer;
        std::unique_ptr<faiss::Index> loaded(faiss::read_index(index_path.c_str()));
        faiss::IndexHNSWPQ* idx = dynamic_cast<faiss::IndexHNSWPQ*>(loaded.get());
        if (idx) {
            loaded.release();
            std::cout << "Load time: " << std::fixed << std::setprecision(1)
                      << timer.elapsed_sec() << "s" << std::endl;
            return idx;
        }
        std::cerr << "Warning: loaded index is not IndexHNSWPQ, rebuilding" << std::endl;
    }

    // Build new index
    std::cout << "[Building IndexHNSWPQ (PQ M=" << pq_M
              << ", nbits=" << pq_nbits << ")...]" << std::endl;

    Timer timer;
    auto* idx = new faiss::IndexHNSWPQ(d, pq_M, hnsw_M, pq_nbits);
    idx->hnsw.efConstruction = hnsw_efConstruction;
    idx->train(nb, xb);
    idx->add(nb, xb);

    std::cout << "Build time: " << std::fixed << std::setprecision(1)
              << timer.elapsed_sec() << "s" << std::endl;

    // Save
    size_t last_slash = index_path.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(index_path.substr(0, last_slash));
    }
    std::cout << "[Saving index to " << index_path << "...]" << std::endl;
    faiss::write_index(idx, index_path.c_str());

    return idx;
}

/*****************************************************
 * Main
 *****************************************************/

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    // Load dataset config
    std::string datasets_config = opts.config_dir + "/datasets.conf";
    auto dataset_configs = ConfigParser::parse_datasets(datasets_config);

    DatasetConfig ds_cfg;
    auto it = dataset_configs.find(opts.dataset);
    if (it != dataset_configs.end()) {
        ds_cfg = it->second;
    } else {
        std::cerr << "Warning: Dataset '" << opts.dataset
                  << "' not found in " << datasets_config
                  << ", using defaults" << std::endl;
        ds_cfg.name = opts.dataset;
        ds_cfg.base_path = "/data/local/embedding_dataset/" + opts.dataset;
    }

    // Apply command line overrides
    if (!opts.data_path.empty()) {
        ds_cfg.base_path = opts.data_path;
    }
    if (opts.threads > 0) {
        ds_cfg.threads = opts.threads;
    }
    if (!opts.ef_values.empty()) {
        ds_cfg.ef_search_values = opts.ef_values;
    }
    if (ds_cfg.ef_search_values.empty()) {
        ds_cfg.ef_search_values = get_default_ef_search();
    }

    omp_set_num_threads(ds_cfg.threads);

    // Load PQ config
    std::string pq_config = opts.config_dir + "/pq.conf";
    auto algo_params = ConfigParser::parse_algorithm(pq_config);

    std::vector<AlgorithmParamSet> param_sets;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end()) {
        param_sets = ait->second;
    }

    if (param_sets.empty()) {
        std::cerr << "Error: No PQ configs found for dataset '"
                  << opts.dataset << "' in " << pq_config << std::endl;
        return 1;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "IndexHNSWPQ Benchmark (PQ search + exact rerank)" << std::endl;
    std::cout << "Dataset:           " << opts.dataset << std::endl;
    std::cout << "Data path:         " << ds_cfg.base_path << std::endl;
    std::cout << "HNSW M:            " << ds_cfg.hnsw_M << std::endl;
    std::cout << "HNSW efConstruct:  " << ds_cfg.hnsw_efConstruction << std::endl;
    std::cout << "Threads:           " << ds_cfg.threads << std::endl;
    std::cout << "PQ configs:        " << param_sets.size() << " parameter sets" << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load data
    std::cout << "\n[Loading data...]" << std::endl;
    size_t d, nb, nq, gt_k;
    float* xb = fvecs_read(ds_cfg.get_path(ds_cfg.base_file).c_str(), &d, &nb);
    float* xq = fvecs_read(ds_cfg.get_path(ds_cfg.query_file).c_str(), &d, &nq);
    int* gt = ivecs_read(ds_cfg.get_path(ds_cfg.groundtruth_file).c_str(), &gt_k, &nq);

    if (!xb || !xq || !gt) {
        std::cerr << "Error: Failed to load data files" << std::endl;
        return 1;
    }

    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;
    std::cout << "Queries:  " << nq << " vectors" << std::endl;

    const size_t k = ds_cfg.k;

    // Run benchmarks for each PQ parameter set
    for (const auto& ps : param_sets) {
        int pq_M = std::stoi(ps.params.at("M"));
        int pq_nbits = 8;
        auto nbits_it = ps.params.find("nbits");
        if (nbits_it != ps.params.end()) {
            pq_nbits = std::stoi(nbits_it->second);
        }

        std::cout << "\n==================================================" << std::endl;
        std::cout << "IndexHNSWPQ | PQ M=" << pq_M << ", nbits=" << pq_nbits << std::endl;
        std::cout << "==================================================" << std::endl;

        // Check dimension divisibility
        if (d % pq_M != 0) {
            std::cout << "[SKIP] dimension " << d
                      << " not divisible by PQ M=" << pq_M << std::endl;
            continue;
        }

        // Build or load IndexHNSWPQ
        std::ostringstream idx_oss;
        idx_oss << ds_cfg.base_path << "/index/hnswpq_M" << pq_M
                << "_nbits" << pq_nbits
                << "_hM" << ds_cfg.hnsw_M
                << "_efc" << ds_cfg.hnsw_efConstruction << ".faissindex";
        std::string index_path = idx_oss.str();

        faiss::IndexHNSWPQ* hnswpq_index = load_or_build_hnswpq(
            index_path, d, pq_M, pq_nbits,
            ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction, nb, xb);

        if (!hnswpq_index) {
            std::cerr << "Error: Failed to load/build IndexHNSWPQ" << std::endl;
            continue;
        }

        // Build flat index for reranking
        std::cout << "[Building IndexFlatL2 for reranking...]" << std::endl;
        faiss::IndexFlatL2 flat_index(d);
        flat_index.add(nb, xb);

        // Wrap with IndexRefine: PQ search + exact rerank
        faiss::IndexRefine refine_index(hnswpq_index, &flat_index);
        refine_index.own_fields = false;  // we manage memory ourselves

        // Print header
        std::cout << "\n" << std::setw(8) << "ef"
                  << std::setw(12) << "QPS"
                  << std::setw(12) << "Recall@" << k
                  << std::setw(12) << "Latency(ms)"
                  << std::setw(12) << "ndis_mean"
                  << std::setw(12) << "nhops_mean" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        std::vector<BenchmarkResult> results;

        for (size_t ef : ds_cfg.ef_search_values) {
            hnswpq_index->hnsw.efSearch = ef;
            refine_index.k_factor = (float)ef / k;

            std::vector<faiss::idx_t> result_ids(nq * k);
            std::vector<float> result_dists(nq * k);

            // Reset HNSW stats
            faiss::hnsw_stats.reset();

            // Timed search
            Timer timer;
            refine_index.search(nq, xq, k, result_dists.data(), result_ids.data());
            double search_time = timer.elapsed_ms();

            // Read accumulated stats
            double ndis_mean = (double)faiss::hnsw_stats.ndis / nq;
            double nhops_mean = (double)faiss::hnsw_stats.nhops / nq;

            double qps = nq * 1000.0 / search_time;
            double latency = search_time / nq;
            float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, gt_k);

            results.push_back({ef, qps, recall, latency, search_time,
                               ndis_mean, nhops_mean});

            std::cout << std::setw(8)  << ef
                      << std::setw(12) << std::fixed << std::setprecision(0) << qps
                      << std::setw(12) << std::setprecision(4) << recall
                      << std::setw(12) << std::setprecision(3) << latency
                      << std::setw(12) << std::setprecision(1) << ndis_mean
                      << std::setw(12) << std::setprecision(1) << nhops_mean
                      << std::endl;
        }

        // Save results
        std::ostringstream res_oss;
        res_oss << "experiments/hnsw/results/" << opts.dataset << "/hnswpq/"
                << "hnswpq_M" << pq_M << "_nbits" << pq_nbits
                << "_hM" << ds_cfg.hnsw_M
                << "_efc" << ds_cfg.hnsw_efConstruction << ".txt";

        save_results_to_file(
            res_oss.str(), opts.dataset, pq_M, pq_nbits,
            ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction,
            nq, k, results);

        delete hnswpq_index;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
