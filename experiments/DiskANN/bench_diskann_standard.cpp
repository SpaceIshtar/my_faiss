/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark standard DiskANN with native PQ distances.
 *
 * This benchmark tests DiskANN's built-in PQ-based distance estimation
 * (no custom quantizer), serving as a baseline for comparison with
 * bench_diskann_quant (custom quantizers).
 *
 * Usage:
 *   ./bench_diskann_standard --dataset <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1M)
 *   --config-dir <path>    Config directory (default: ./config)
 *   --data-path <path>     Override dataset base path
 *   --threads <n>          Number of threads
 *   --L <list>             Comma-separated L values to test
 *   --beam-width <n>       Override beam width
 *   --help                 Show this help
 *
 * Example:
 *   ./bench_diskann_standard --dataset sift1M
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

// DiskANN headers
#include "linux_aligned_file_reader.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "utils.h"

// Shared utilities (Timer, fvecs_read, PercentileStats, etc.)
#include "../hnsw/include/bench_utils.h"

// DiskANN-specific config
#include "include/diskann_bench_config.h"

using namespace hnsw_bench;
using namespace diskann_bench;

/*****************************************************
 * Recall computation (DiskANN returns uint64_t IDs)
 *****************************************************/

float compute_recall_at_k_u64(
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
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1M";
    std::string config_dir = "./config";
    std::string data_path;        // override
    int threads = -1;             // -1 = use config
    std::vector<uint32_t> L_values;
    uint32_t beam_width = 0;     // 0 = use config
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>       Dataset name (e.g., sift1M)\n"
              << "  --config-dir <path>    Config directory (default: ./config)\n"
              << "  --data-path <path>     Override dataset base path\n"
              << "  --threads <n>          Number of threads\n"
              << "  --L <list>             Comma-separated L values to test\n"
              << "  --beam-width <n>       Override beam width\n"
              << "  --help                 Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1M\n"
              << "  " << prog << " --dataset gist1M --beam-width 8\n";
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
        } else if (arg == "--L" && i + 1 < argc) {
            std::string L_str = argv[++i];
            std::istringstream iss(L_str);
            std::string token;
            while (std::getline(iss, token, ',')) {
                opts.L_values.push_back(std::stoul(token));
            }
        } else if (arg == "--beam-width" && i + 1 < argc) {
            opts.beam_width = std::stoul(argv[++i]);
        }
    }

    return opts;
}

/*****************************************************
 * Result structures
 *****************************************************/

struct BenchmarkResult {
    uint32_t L;
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
    auto dataset_configs = parse_diskann_datasets(datasets_config);

    DiskANNDatasetConfig ds_cfg;
    auto it = dataset_configs.find(opts.dataset);
    if (it != dataset_configs.end()) {
        ds_cfg = it->second;
    } else {
        ds_cfg.name = opts.dataset;
        ds_cfg.base_path = "/data/local/embedding_dataset/" + opts.dataset;
        ds_cfg.L_search_values = get_default_L_search();
    }

    // Apply command line overrides
    if (!opts.data_path.empty()) {
        ds_cfg.base_path = opts.data_path;
    }
    if (opts.threads > 0) {
        ds_cfg.threads = opts.threads;
    }
    if (!opts.L_values.empty()) {
        ds_cfg.L_search_values = opts.L_values;
    }
    if (opts.beam_width > 0) {
        ds_cfg.beam_width = opts.beam_width;
    }
    if (ds_cfg.L_search_values.empty()) {
        ds_cfg.L_search_values = get_default_L_search();
    }

    omp_set_num_threads(ds_cfg.threads);

    std::cout << "==================================================" << std::endl;
    std::cout << "Standard DiskANN Benchmark (Native PQ Distances)" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "DiskANN R: " << ds_cfg.diskann_R << std::endl;
    std::cout << "DiskANN L_build: " << ds_cfg.diskann_L_build << std::endl;
    std::cout << "Beam width: " << ds_cfg.beam_width << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset
    std::cout << "\n[Loading data...]" << std::endl;
    size_t d, nb, nq, gt_k;
    float* xq = fvecs_read(ds_cfg.get_path(ds_cfg.query_file).c_str(), &d, &nq);
    int* gt = ivecs_read(ds_cfg.get_path(ds_cfg.groundtruth_file).c_str(), &gt_k, &nq);

    if (!xq || !gt) {
        std::cerr << "Error: Failed to load query/groundtruth files" << std::endl;
        return 1;
    }

    std::cout << "Queries: " << nq << " vectors, dimension " << d << std::endl;

    // Load DiskANN disk index
    std::string index_prefix = ds_cfg.get_diskann_index_prefix();
    std::cout << "\n[Loading DiskANN disk index...]" << std::endl;
    std::cout << "Index prefix: " << index_prefix << std::endl;

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<float>> flash_index(
        new diskann::PQFlashIndex<float>(reader, diskann::Metric::L2));

    Timer timer;
    int load_result = flash_index->load(ds_cfg.threads, index_prefix.c_str());
    if (load_result != 0) {
        std::cerr << "Error loading DiskANN index: " << load_result << std::endl;
        return load_result;
    }
    std::cout << "DiskANN index load time: " << timer.elapsed_ms() << " ms" << std::endl;

    // Cache BFS levels for faster search
    std::vector<uint32_t> node_list;
    flash_index->cache_bfs_levels(ds_cfg.num_nodes_to_cache, node_list);
    flash_index->load_cache_list(node_list);
    std::cout << "Cached " << ds_cfg.num_nodes_to_cache << " nodes" << std::endl;

    // Benchmark
    std::vector<BenchmarkResult> results;
    const size_t k = ds_cfg.k;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Standard DiskANN Search Results (Native PQ)" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::setw(8) << "L"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "Latency(ms)"
              << std::setw(12) << "ndis_mean"
              << std::setw(12) << "nhops_mean" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (uint32_t L : ds_cfg.L_search_values) {
        if (L < k) continue;

        std::vector<uint64_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);
        std::vector<diskann::QueryStats> query_stats(nq);

        // Warmup
        size_t warmup_n = std::min((size_t)100, nq);
        for (size_t i = 0; i < warmup_n; i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                ds_cfg.beam_width, false, nullptr);
        }

        // Reset stats
        std::memset(query_stats.data(), 0, nq * sizeof(diskann::QueryStats));

        // Timed search
        timer.reset();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)nq; i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                ds_cfg.beam_width, false, &query_stats[i]);
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k_u64(nq, k, result_ids.data(), gt, gt_k);

        // Aggregate ndis/nhop statistics
        std::vector<size_t> ndis_values(nq), nhop_values(nq);
        for (size_t i = 0; i < nq; i++) {
            ndis_values[i] = query_stats[i].n_cmps;
            nhop_values[i] = query_stats[i].n_hops;
        }
        SearchStats ss;
        ss.ndis_stats = compute_percentile_stats(ndis_values);
        ss.nhops_stats = compute_percentile_stats(nhop_values);

        results.push_back({L, qps, recall, latency, search_time, ss});

        std::cout << std::setw(8) << L
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << std::setw(12) << std::setprecision(4) << recall
                  << std::setw(12) << std::setprecision(3) << latency
                  << std::setw(12) << std::setprecision(1) << ss.ndis_stats.mean
                  << std::setw(12) << std::setprecision(1) << ss.nhops_stats.mean
                  << std::endl;
    }

    // Save results to file
    std::string result_dir = "experiments/DiskANN/results/" + opts.dataset + "/standard";
    create_directory(result_dir);
    std::string result_file = result_dir + "/R" + std::to_string(ds_cfg.diskann_R) +
        "_L" + std::to_string(ds_cfg.diskann_L_build) + ".txt";

    std::ofstream ofs(result_file);
    if (ofs.is_open()) {
        ofs << "# Standard DiskANN Benchmark (Native PQ Distances)\n"
            << "# Dataset: " << opts.dataset << "\n"
            << "# DiskANN R: " << ds_cfg.diskann_R
            << ", L_build: " << ds_cfg.diskann_L_build << "\n"
            << "# Beam width: " << ds_cfg.beam_width << "\n"
            << "# k: " << k << "\n\n"
            << std::setw(8) << "L"
            << std::setw(12) << "QPS"
            << std::setw(12) << "Recall"
            << std::setw(12) << "Latency(ms)"
            << std::setw(12) << "ndis_mean"
            << std::setw(12) << "nhops_mean" << "\n"
            << std::string(68, '-') << "\n";

        for (const auto& r : results) {
            ofs << std::setw(8) << r.L
                << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
                << std::setw(12) << std::setprecision(4) << r.recall
                << std::setw(12) << std::setprecision(3) << r.latency_ms
                << std::setw(12) << std::setprecision(1) << r.stats.ndis_stats.mean
                << std::setw(12) << std::setprecision(1) << r.stats.nhops_stats.mean
                << "\n";
        }

        // Detailed stats
        ofs << "\n[Detailed Statistics]\n";
        for (const auto& r : results) {
            ofs << "\n--- L = " << r.L << " ---\n"
                << "  QPS:          " << std::fixed << std::setprecision(2) << r.qps << "\n"
                << "  Recall@" << k << ":    " << std::setprecision(4) << r.recall << "\n"
                << "  Total Time:   " << std::setprecision(2) << r.total_time_ms << " ms\n"
                << "  Avg Latency:  " << std::setprecision(4) << r.latency_ms << " ms\n";
            print_percentile_stats(ofs, "ndis (distance computations)", r.stats.ndis_stats);
            print_percentile_stats(ofs, "nhops (graph hops)", r.stats.nhops_stats);
        }

        ofs.close();
        std::cout << "\nResults saved to: " << result_file << std::endl;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    delete[] xq;
    delete[] gt;

    return 0;
}