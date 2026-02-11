/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark standard HNSW with exact distances.
 *
 * This benchmark tests HNSW's built-in exact distance computation
 * (no custom quantizer), serving as a baseline for comparison with
 * bench_hnsw_quant (custom quantizers).
 *
 * Usage:
 *   ./bench_hnsw_standard --dataset <name> [options]
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
 *   ./bench_hnsw_standard --dataset sift1m
 *   ./bench_hnsw_standard --dataset gist1M --ef 10,20,50,100,200
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

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1m";
    std::string config_dir = "./config";
    std::string data_path;        // override
    int threads = -1;             // -1 = use config
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
 * Result structures
 *****************************************************/

struct BenchmarkResult {
    size_t ef;
    double qps;
    float recall;
    double latency_ms;
    double total_time_ms;
    SearchStats stats;
};

/*****************************************************
 * Result file output
 *****************************************************/

void print_percentile_stats(std::ostream& os, const std::string& name, const PercentileStats& stats) {
    os << "  " << name << ":\n"
       << "    min:      " << std::fixed << std::setprecision(2) << stats.min << "\n"
       << "    p10:      " << stats.p10 << "\n"
       << "    p25:      " << stats.p25 << "\n"
       << "    p50:      " << stats.p50 << "\n"
       << "    p75:      " << stats.p75 << "\n"
       << "    p90:      " << stats.p90 << "\n"
       << "    max:      " << stats.max << "\n"
       << "    mean:     " << stats.mean << "\n"
       << "    variance: " << stats.variance << "\n"
       << "    stddev:   " << stats.stddev << "\n";
}

void save_results_to_file(
        const std::string& filepath,
        const std::string& dataset_name,
        int hnsw_M,
        int hnsw_efConstruction,
        size_t nq,
        size_t k,
        const std::vector<BenchmarkResult>& results) {

    // Create directory if needed
    size_t last_slash = filepath.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(filepath.substr(0, last_slash));
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }

    // Header
    ofs << "========================================\n"
        << "HNSW Standard Benchmark Results\n"
        << "========================================\n\n";

    // Configuration
    ofs << "[Configuration]\n"
        << "  Dataset:           " << dataset_name << "\n"
        << "  Algorithm:         standard (exact distances)\n"
        << "  HNSW M:            " << hnsw_M << "\n"
        << "  HNSW efConstruct:  " << hnsw_efConstruction << "\n"
        << "  Num Queries:       " << nq << "\n"
        << "  Recall@k:          " << k << "\n\n";

    // Summary table
    ofs << "[Summary]\n"
        << std::setw(8) << "ef"
        << std::setw(12) << "QPS"
        << std::setw(12) << "Recall"
        << std::setw(12) << "Latency(ms)"
        << std::setw(12) << "ndis_mean"
        << std::setw(12) << "nhops_mean"
        << "\n"
        << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        ofs << std::setw(8) << r.ef
            << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
            << std::setw(12) << std::setprecision(4) << r.recall
            << std::setw(12) << std::setprecision(3) << r.latency_ms
            << std::setw(12) << std::setprecision(1) << r.stats.ndis_stats.mean
            << std::setw(12) << std::setprecision(1) << r.stats.nhops_stats.mean
            << "\n";
    }
    ofs << "\n";

    // Detailed statistics for each ef value
    ofs << "[Detailed Statistics]\n";
    for (const auto& r : results) {
        ofs << "\n--- ef = " << r.ef << " ---\n"
            << "  QPS:          " << std::fixed << std::setprecision(2) << r.qps << "\n"
            << "  Recall@" << k << ":    " << std::setprecision(4) << r.recall << "\n"
            << "  Total Time:   " << std::setprecision(2) << r.total_time_ms << " ms\n"
            << "  Avg Latency:  " << std::setprecision(4) << r.latency_ms << " ms\n\n";

        print_percentile_stats(ofs, "ndis (distance computations)", r.stats.ndis_stats);
        ofs << "\n";
        print_percentile_stats(ofs, "nhops (graph edges traversed)", r.stats.nhops_stats);
    }

    ofs << "\n========================================\n"
        << "End of Results\n"
        << "========================================\n";

    ofs.close();
    std::cout << "Results saved to: " << filepath << std::endl;
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

    std::cout << "==================================================" << std::endl;
    std::cout << "Standard HNSW Benchmark (Exact Distances)" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "HNSW M: " << ds_cfg.hnsw_M << std::endl;
    std::cout << "HNSW efConstruction: " << ds_cfg.hnsw_efConstruction << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset
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
    std::cout << "Queries: " << nq << " vectors" << std::endl;

    // Load or build HNSW index
    std::string hnsw_path = ds_cfg.get_hnsw_index_path();
    faiss::IndexHNSW* hnsw_index = load_or_build_hnsw(
        hnsw_path, d, ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction, nb, xb);

    if (!hnsw_index) {
        std::cerr << "Error: Failed to load/build HNSW index" << std::endl;
        return 1;
    }

    // Benchmark
    const size_t k = ds_cfg.k;
    std::vector<BenchmarkResult> results;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Standard HNSW Search Results (Exact Distances)" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::setw(8) << "ef"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "Latency(ms)"
              << std::setw(12) << "ndis_mean"
              << std::setw(12) << "nhops_mean" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (size_t ef : ds_cfg.ef_search_values) {
        std::vector<faiss::idx_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);

        faiss::SearchParametersHNSW params;
        params.efSearch = ef;

        // Warmup
        hnsw_index->search(std::min((faiss::idx_t)100, (faiss::idx_t)nq),
                          xq, k, result_dists.data(), result_ids.data(), &params);

        // Timed search with stats collection
        faiss::hnsw_stats.reset();
        Timer timer;
        hnsw_index->search(nq, xq, k, result_dists.data(), result_ids.data(), &params);

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, gt_k);

        // Collect ndis/nhop from HNSW global stats
        // hnsw_stats accumulates totals across all queries; compute mean per query
        SearchStats ss;
        double mean_ndis = (double)faiss::hnsw_stats.ndis / nq;
        double mean_nhops = (double)faiss::hnsw_stats.nhops / nq;
        ss.ndis_stats.mean = mean_ndis;
        ss.ndis_stats.p50 = mean_ndis;  // approximation (no per-query distribution)
        ss.nhops_stats.mean = mean_nhops;
        ss.nhops_stats.p50 = mean_nhops;

        results.push_back({ef, qps, recall, latency, search_time, ss});

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << std::setw(12) << std::setprecision(4) << recall
                  << std::setw(12) << std::setprecision(3) << latency
                  << std::setw(12) << std::setprecision(1) << ss.ndis_stats.mean
                  << std::setw(12) << std::setprecision(1) << ss.nhops_stats.mean
                  << std::endl;
    }

    // Save results to file
    std::string result_filename = "standard_M" + std::to_string(ds_cfg.hnsw_M) +
        "_efc" + std::to_string(ds_cfg.hnsw_efConstruction) + ".txt";
    std::string result_dir = "experiments/hnsw/results/" + opts.dataset + "/standard";
    std::string result_path = result_dir + "/" + result_filename;

    save_results_to_file(
        result_path, opts.dataset,
        ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction,
        nq, k, results);

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete hnsw_index;

    return 0;
}
