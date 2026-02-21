/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * DiskANN + Quantization Benchmark Framework
 *
 * This benchmark tests DiskANN graph traversal with various quantization methods
 * for distance estimation via a generic DistanceComputer factory.
 *
 * Usage:
 *   ./bench_diskann_quant --dataset <name> --algorithm <name> [options]
 *
 * Options:
 *   --dataset <name>           Dataset name (e.g., sift1M)
 *   --algorithm <name>         Algorithm name (pq, sq, opq, rq, lsq, prq, plsq, vaq, rabitq)
 *   --config-dir <path>        Config directory (default: ./config)
 *   --algo-config-dir <path>   Algorithm config directory (default: config-dir)
 *   --data-path <path>         Override dataset base path
 *   --threads <n>              Number of threads
 *   --L <list>                 Comma-separated L values to test
 *   --beam-width <n>           Override beam width
 *   --help                     Show this help
 *
 * Example:
 *   ./bench_diskann_quant --dataset sift1M --algorithm pq
 *   ./bench_diskann_quant --dataset sift1M --algorithm sq --data-path /path/to/data
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

// FAISS headers
#include <faiss/IndexFlat.h>
#include <faiss/impl/DistanceComputer.h>

// DiskANN headers
#include "linux_aligned_file_reader.h"
#include "percentile_stats.h"
#include "pq_flash_index.h"
#include "utils.h"

// Shared quantization wrappers (from HNSW framework)
#include "../hnsw/include/aq_wrapper.h"
#include "../hnsw/include/opq_wrapper.h"
#include "../hnsw/include/pq_wrapper.h"
#include "../hnsw/include/quant_wrapper.h"
#include "../hnsw/include/rabitq_wrapper.h"
#include "../hnsw/include/saq_wrapper.h"
#include "../hnsw/include/sq_wrapper.h"
#include "../hnsw/include/vaq_wrapper.h"

// Shared utilities (Timer, fvecs_read, etc.)
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
 * Benchmark runner
 *****************************************************/

struct BenchmarkResult {
    uint32_t L;
    double qps;
    float recall;
    double latency_ms;
    double total_time_ms;
    SearchStats stats;
};

std::vector<BenchmarkResult> run_benchmark(
        diskann::PQFlashIndex<float>* flash_index,
        const float* xq,
        size_t nq,
        size_t d,
        const int* gt,
        size_t gt_k,
        size_t k,
        const std::vector<uint32_t>& L_values,
        uint32_t beam_width,
        int num_threads,
        bool verbose = true) {

    std::vector<BenchmarkResult> results;
    Timer timer;

    for (uint32_t L : L_values) {
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
                beam_width, false, nullptr);
        }

        // Reset stats
        std::memset(query_stats.data(), 0, nq * sizeof(diskann::QueryStats));

        // Timed search
        timer.reset();

        #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
        for (int64_t i = 0; i < (int64_t)nq; i++) {
            flash_index->cached_beam_search(
                xq + i * d, k, L,
                result_ids.data() + i * k,
                result_dists.data() + i * k,
                beam_width, false, &query_stats[i]);
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

        if (verbose) {
            std::cout << std::setw(8) << L
                      << std::setw(12) << std::fixed << std::setprecision(0) << qps
                      << std::setw(12) << std::setprecision(4) << recall
                      << std::setw(12) << std::setprecision(3) << latency
                      << std::setw(12) << std::setprecision(1) << ss.ndis_stats.mean
                      << std::setw(12) << std::setprecision(1) << ss.nhops_stats.mean
                      << std::endl;
        }
    }

    return results;
}

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
        const std::string& algorithm_name,
        const std::string& quant_params,
        int diskann_R,
        int diskann_L_build,
        uint32_t beam_width,
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
        << "DiskANN + Quantization Benchmark Results\n"
        << "========================================\n\n";

    // Configuration
    ofs << "[Configuration]\n"
        << "  Dataset:           " << dataset_name << "\n"
        << "  Algorithm:         " << algorithm_name << "\n"
        << "  Quant Params:      " << quant_params << "\n"
        << "  DiskANN R:         " << diskann_R << "\n"
        << "  DiskANN L_build:   " << diskann_L_build << "\n"
        << "  Beam Width:        " << beam_width << "\n"
        << "  Num Queries:       " << nq << "\n"
        << "  Recall@k:          " << k << "\n\n";

    // Summary table
    ofs << "[Summary]\n"
        << std::setw(8) << "L"
        << std::setw(12) << "QPS"
        << std::setw(12) << "Recall"
        << std::setw(12) << "Latency(ms)"
        << std::setw(12) << "ndis_mean"
        << std::setw(12) << "nhops_mean"
        << "\n"
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
    ofs << "\n";

    // Detailed per-L stats
    ofs << "[Detailed Statistics]\n";
    for (const auto& r : results) {
        ofs << "\n--- L = " << r.L << " ---\n"
            << "  QPS:          " << std::fixed << std::setprecision(2) << r.qps << "\n"
            << "  Recall@" << k << ":    " << std::setprecision(4) << r.recall << "\n"
            << "  Total Time:   " << std::setprecision(2) << r.total_time_ms << " ms\n"
            << "  Avg Latency:  " << std::setprecision(4) << r.latency_ms << " ms\n";
        print_percentile_stats(ofs, "ndis (distance computations)", r.stats.ndis_stats);
        print_percentile_stats(ofs, "nhops (graph hops)", r.stats.nhops_stats);
    }

    ofs << "\n========================================\n"
        << "End of Results\n"
        << "========================================\n";

    ofs.close();
    std::cout << "Results saved to: " << filepath << std::endl;
}

std::string generate_result_filename(
        const std::string& algorithm,
        const std::string& quant_params,
        int diskann_R,
        int diskann_L_build) {
    std::ostringstream oss;
    oss << algorithm << "_" << quant_params
        << "_R" << diskann_R << "_L" << diskann_L_build << ".txt";
    return oss.str();
}

/*****************************************************
 * Algorithm factory
 *****************************************************/

std::unique_ptr<QuantWrapper> create_wrapper(
        const std::string& algorithm,
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {

    if (algorithm == "pq") {
        return create_pq_wrapper(d, metric, params);
    } else if (algorithm == "opq") {
        return create_opq_wrapper(d, metric, params);
    } else if (algorithm == "sq") {
        return create_sq_wrapper(d, metric, params);
    } else if (algorithm == "rq") {
        return create_rq_wrapper(d, metric, params);
    } else if (algorithm == "lsq") {
        return create_lsq_wrapper(d, metric, params);
    } else if (algorithm == "prq") {
        return create_prq_wrapper(d, metric, params);
    } else if (algorithm == "plsq") {
        return create_plsq_wrapper(d, metric, params);
    } else if (algorithm == "vaq") {
        return create_vaq_wrapper(d, metric, params);
    } else if (algorithm == "rabitq") {
        return create_rabitq_wrapper(d, metric, params);
    } else if (algorithm == "saq") {
        return create_saq_wrapper(d, metric, params);
    } else {
        throw std::runtime_error("Unknown algorithm: " + algorithm);
    }
}

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1M";
    std::string algorithm = "sq";
    std::string config_dir = "./config";
    std::string algo_config_dir;  // defaults to config_dir
    std::string data_path;        // override
    int threads = -1;             // -1 = use config
    std::vector<uint32_t> L_values;
    uint32_t beam_width = 0;     // 0 = use config
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> --algorithm <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>           Dataset name (e.g., sift1M)\n"
              << "  --algorithm <name>         Algorithm (pq, opq, sq, rq, lsq, prq, plsq, vaq, rabitq, saq)\n"
              << "  --config-dir <path>        Config directory (default: ./config)\n"
              << "  --algo-config-dir <path>   Algorithm config directory (default: config-dir)\n"
              << "  --data-path <path>         Override dataset base path\n"
              << "  --threads <n>              Number of threads\n"
              << "  --L <list>                 Comma-separated L values to test\n"
              << "  --beam-width <n>           Override beam width\n"
              << "  --help                     Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1M --algorithm pq\n"
              << "  " << prog << " --dataset sift1M --algorithm sq --data-path /path/to/data\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            opts.dataset = argv[++i];
        } else if (arg == "--algorithm" && i + 1 < argc) {
            opts.algorithm = argv[++i];
        } else if (arg == "--config-dir" && i + 1 < argc) {
            opts.config_dir = argv[++i];
        } else if (arg == "--algo-config-dir" && i + 1 < argc) {
            opts.algo_config_dir = argv[++i];
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
 * Main
 *****************************************************/

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "DiskANN + Quantization Benchmark Framework" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Algorithm: " << opts.algorithm << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset config
    std::string datasets_config = opts.config_dir + "/datasets.conf";
    auto dataset_configs = parse_diskann_datasets(datasets_config);

    DiskANNDatasetConfig ds_cfg;
    auto it = dataset_configs.find(opts.dataset);
    if (it != dataset_configs.end()) {
        ds_cfg = it->second;
    } else {
        // Use defaults
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
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;

    // Load algorithm config
    // Try algo-config-dir first, then config-dir
    std::string algo_config_dir = opts.algo_config_dir.empty()
        ? opts.config_dir : opts.algo_config_dir;
    std::string algo_config = algo_config_dir + "/" + opts.algorithm + ".conf";
    auto algo_params = ConfigParser::parse_algorithm(algo_config);

    std::vector<AlgorithmParamSet> param_sets;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end()) {
        param_sets = ait->second;
    }

    // If no config, use defaults (same as HNSW framework)
    if (param_sets.empty()) {
        if (opts.algorithm == "pq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "32"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "opq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "32"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "sq") {
            param_sets.push_back({.params = {{"qtype", "QT_8bit"}}});
            param_sets.push_back({.params = {{"qtype", "QT_4bit"}}});
        } else if (opts.algorithm == "rq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "lsq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "prq") {
            param_sets.push_back({.params = {{"nsplits", "2"}, {"Msub", "4"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"nsplits", "4"}, {"Msub", "4"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "plsq") {
            param_sets.push_back({.params = {{"nsplits", "2"}, {"Msub", "4"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"nsplits", "4"}, {"Msub", "4"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "vaq") {
            param_sets.push_back({.params = {{"bits", "128"}, {"nsub", "16"}, {"minbps", "7"}, {"maxbps", "9"}}});
            param_sets.push_back({.params = {{"bits", "256"}, {"nsub", "32"}, {"minbps", "7"}, {"maxbps", "9"}}});
        } else if (opts.algorithm == "rabitq") {
            param_sets.push_back({.params = {{"bits", "1"}}});
            param_sets.push_back({.params = {{"bits", "2"}}});
            param_sets.push_back({.params = {{"bits", "4"}}});
        } else if (opts.algorithm == "saq") {
            param_sets.push_back({.params = {{"bits", "1"}, {"clusters", "4096"}}});
            param_sets.push_back({.params = {{"bits", "2"}, {"clusters", "4096"}}});
            param_sets.push_back({.params = {{"bits", "4"}, {"clusters", "4096"}}});
        }
    }

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
    std::cout << "Queries: " << nq << " vectors" << std::endl;

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

    // Run benchmarks for each parameter set
    for (const auto& ps : param_sets) {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Algorithm: " << opts.algorithm << " | Params: " << ps.to_string() << std::endl;
        std::cout << "==================================================" << std::endl;

        // Create quantizer wrapper
        auto quant = create_wrapper(opts.algorithm, d, faiss::METRIC_L2, ps.params);

        // Train and add vectors
        std::cout << "[Training " << quant->get_name() << "...]" << std::endl;
        timer.reset();
        quant->train(nb, xb);
        quant->add(nb, xb);
        std::cout << "Train+add time: " << timer.elapsed_ms() << " ms" << std::endl;

        // Set the distance computer factory on the DiskANN index
        flash_index->set_distance_computer_factory([&quant]() {
            return quant->get_distance_computer();
        });

        // Print results header
        std::cout << "\n" << std::setw(8) << "L"
                  << std::setw(12) << "QPS"
                  << std::setw(12) << "Recall@" << ds_cfg.k
                  << std::setw(12) << "Latency(ms)"
                  << std::setw(12) << "ndis_mean"
                  << std::setw(12) << "nhops_mean" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        // Run benchmark
        auto results = run_benchmark(
            flash_index.get(), xq, nq, d, gt, gt_k, ds_cfg.k,
            ds_cfg.L_search_values, ds_cfg.beam_width, ds_cfg.threads);

        // Save results to file
        std::string quant_params = quant->get_params_string();
        std::string result_filename = generate_result_filename(
            opts.algorithm, quant_params, ds_cfg.diskann_R, ds_cfg.diskann_L_build);
        std::string result_dir = "experiments/DiskANN/results/" + opts.dataset + "/" + opts.algorithm;
        std::string result_path = result_dir + "/" + result_filename;

        save_results_to_file(
            result_path, opts.dataset, opts.algorithm, quant_params,
            ds_cfg.diskann_R, ds_cfg.diskann_L_build, ds_cfg.beam_width,
            nq, ds_cfg.k, results);
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
