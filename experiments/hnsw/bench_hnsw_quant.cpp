/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * HNSW + Quantization Benchmark Framework
 *
 * This benchmark tests HNSW graph traversal with various quantization methods
 * for distance estimation, followed by exact distance reranking.
 *
 * Usage:
 *   ./bench_hnsw_quant --dataset <name> --algorithm <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1m)
 *   --algorithm <name>     Algorithm name (pq, sq)
 *   --config-dir <path>    Config directory (default: ./config)
 *   --data-path <path>     Override dataset base path
 *   --threads <n>          Number of threads
 *   --ef <list>            Comma-separated ef values to test
 *   --help                 Show this help
 *
 * Example:
 *   ./bench_hnsw_quant --dataset sift1m --algorithm pq
 *   ./bench_hnsw_quant --dataset sift1m --algorithm sq --data-path /path/to/data
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>

#include "include/aq_wrapper.h"
#include "include/bench_config.h"
#include "include/bench_utils.h"
#include "include/opq_wrapper.h"
#include "include/pq_wrapper.h"
#include "include/quant_wrapper.h"
#include "include/rabitq_wrapper.h"
#include "include/sq_wrapper.h"
#include "include/vaq_wrapper.h"

using namespace hnsw_bench;

/*****************************************************
 * HNSW search with quantization + rerank
 *****************************************************/

QueryStats hnsw_search_with_rerank(
        const faiss::IndexHNSW* hnsw_index,
        faiss::DistanceComputer* quant_dc,
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

    // Prepare heap for ef candidates
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    std::vector<float> ef_distances(ef, std::numeric_limits<float>::max());
    std::vector<faiss::idx_t> ef_labels(ef, -1);

    // Create result handler
    RH bres(1, ef_distances.data(), ef_labels.data(), ef);
    RH::SingleResultHandler res(bres);

    // VisitedTable for search
    faiss::VisitedTable vt(hnsw_index->ntotal);

    // Search parameters
    faiss::SearchParametersHNSW params;
    params.efSearch = ef;

    // HNSW search using quantized distances (returns HNSWStats)
    res.begin(0);
    faiss::HNSWStats stats = hnsw.search(*quant_dc, hnsw_index, res, vt, &params);
    res.end();

    // Rerank using exact distances
    std::vector<std::pair<float, faiss::idx_t>> reranked;
    reranked.reserve(ef);
    for (size_t i = 0; i < ef; i++) {
        if (ef_labels[i] >= 0) {
            float dist = (*exact_dc)(ef_labels[i]);
            reranked.push_back({dist, ef_labels[i]});
        }
    }

    // Sort by exact distance
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

    // Return query stats
    QueryStats qs;
    qs.ndis = stats.ndis;
    qs.nhops = stats.nhops;
    return qs;
}

/*****************************************************
 * Benchmark runner
 *****************************************************/

struct BenchmarkResult {
    size_t ef;
    double qps;
    float recall;
    double latency_ms;
    double total_time_ms;
    SearchStats stats;
};

std::vector<BenchmarkResult> run_benchmark(
        const faiss::IndexHNSW* hnsw_index,
        const faiss::IndexFlat* flat_index,
        QuantWrapper* quant,
        const float* xq,
        size_t nq,
        size_t d,
        const int* gt,
        size_t gt_k,
        size_t k,
        const std::vector<size_t>& ef_values,
        int num_threads,
        bool verbose = true) {

    std::vector<BenchmarkResult> results;
    Timer timer;

    for (size_t ef : ef_values) {
        std::vector<faiss::idx_t> result_ids(nq * k, -1);
        std::vector<float> result_dists(nq * k, std::numeric_limits<float>::max());
        std::vector<QueryStats> query_stats(nq);

        // Warmup
        size_t warmup_n = std::min((size_t)10, nq);
        for (size_t i = 0; i < warmup_n; i++) {
            auto quant_dc = quant->get_distance_computer();
            auto exact_dc = std::unique_ptr<faiss::DistanceComputer>(
                flat_index->get_distance_computer());
            hnsw_search_with_rerank(
                hnsw_index, quant_dc.get(), exact_dc.get(),
                xq + i * d, k, ef,
                result_dists.data() + i * k,
                result_ids.data() + i * k);
        }

        // Timed search with stats collection
        timer.reset();

        #pragma omp parallel num_threads(num_threads)
        {
            auto quant_dc = quant->get_distance_computer();
            auto exact_dc = std::unique_ptr<faiss::DistanceComputer>(
                flat_index->get_distance_computer());

            #pragma omp for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)nq; i++) {
                query_stats[i] = hnsw_search_with_rerank(
                    hnsw_index, quant_dc.get(), exact_dc.get(),
                    xq + i * d, k, ef,
                    result_dists.data() + i * k,
                    result_ids.data() + i * k);
            }
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, gt_k);
        SearchStats stats = SearchStats::compute(query_stats);

        results.push_back({ef, qps, recall, latency, search_time, stats});

        if (verbose) {
            std::cout << std::setw(8) << ef
                      << std::setw(12) << std::fixed << std::setprecision(0) << qps
                      << std::setw(12) << std::setprecision(4) << recall
                      << std::setw(12) << std::setprecision(3) << latency
                      << std::setw(12) << std::setprecision(1) << stats.ndis_stats.mean
                      << std::setw(12) << std::setprecision(1) << stats.nhops_stats.mean
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
        << "HNSW + Quantization Benchmark Results\n"
        << "========================================\n\n";

    // Configuration
    ofs << "[Configuration]\n"
        << "  Dataset:           " << dataset_name << "\n"
        << "  Algorithm:         " << algorithm_name << "\n"
        << "  Quant Params:      " << quant_params << "\n"
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

std::string generate_result_filename(
        const std::string& algorithm,
        const std::string& quant_params,
        int hnsw_M,
        int hnsw_efConstruction) {
    std::ostringstream oss;
    oss << algorithm << "_" << quant_params
        << "_M" << hnsw_M << "_efc" << hnsw_efConstruction << ".txt";
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
    } else {
        throw std::runtime_error("Unknown algorithm: " + algorithm);
    }
}

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1m";
    std::string algorithm = "sq";
    std::string config_dir = "./config";
    std::string data_path;  // Override
    int threads = -1;       // -1 means use config
    std::vector<size_t> ef_values;  // Empty means use config
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> --algorithm <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>       Dataset name (e.g., sift1m)\n"
              << "  --algorithm <name>     Algorithm name (pq, sq)\n"
              << "  --config-dir <path>    Config directory (default: ./config)\n"
              << "  --data-path <path>     Override dataset base path\n"
              << "  --threads <n>          Number of threads\n"
              << "  --ef <list>            Comma-separated ef values to test\n"
              << "  --help                 Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1m --algorithm pq\n"
              << "  " << prog << " --dataset sift1m --algorithm sq --data-path /path/to/data\n";
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
 * Main
 *****************************************************/

int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);

    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "HNSW + Quantization Benchmark Framework" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Algorithm: " << opts.algorithm << std::endl;
    std::cout << "==================================================" << std::endl;

    Timer global_timer;

    // Load dataset config
    std::string datasets_config = opts.config_dir + "/datasets.conf";
    auto dataset_configs = ConfigParser::parse_datasets(datasets_config);

    DatasetConfig ds_cfg;
    auto it = dataset_configs.find(opts.dataset);
    if (it != dataset_configs.end()) {
        ds_cfg = it->second;
    } else {
        // Use defaults
        ds_cfg.name = opts.dataset;
        ds_cfg.base_path = "/data/local/embedding_dataset/" + opts.dataset;
        ds_cfg.ef_search_values = get_default_ef_search();
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
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;

    // Load algorithm config
    std::string algo_config = opts.config_dir + "/" + opts.algorithm + ".conf";
    auto algo_params = ConfigParser::parse_algorithm(algo_config);

    std::vector<AlgorithmParamSet> param_sets;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end()) {
        param_sets = ait->second;
    }

    // If no config, use defaults
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

    // Load or build HNSW index
    std::string hnsw_path = ds_cfg.get_hnsw_index_path();
    faiss::IndexHNSW* hnsw_index = load_or_build_hnsw(
        hnsw_path, d, ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction, nb, xb);

    if (!hnsw_index) {
        std::cerr << "Error: Failed to load/build HNSW index" << std::endl;
        return 1;
    }

    // Get flat storage for exact distances
    faiss::IndexFlat* flat_index = dynamic_cast<faiss::IndexFlat*>(hnsw_index->storage);
    if (!flat_index) {
        std::cerr << "Error: HNSW storage is not IndexFlat" << std::endl;
        return 1;
    }

    // Run benchmarks for each parameter set
    for (const auto& ps : param_sets) {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Algorithm: " << opts.algorithm << " | Params: " << ps.to_string() << std::endl;
        std::cout << "==================================================" << std::endl;

        // Create quantizer wrapper
        auto quant = create_wrapper(opts.algorithm, d, faiss::METRIC_L2, ps.params);

        // Train and add vectors
        std::cout << "[Training " << quant->get_name() << "...]" << std::endl;
        Timer timer;
        quant->train(nb, xb);
        quant->add(nb, xb);
        std::cout << "Train+add time: " << timer.elapsed_ms() << " ms" << std::endl;

        // Print results header
        std::cout << "\n" << std::setw(8) << "ef"
                  << std::setw(12) << "QPS"
                  << std::setw(12) << "Recall@" << ds_cfg.k
                  << std::setw(12) << "Latency(ms)"
                  << std::setw(12) << "ndis_mean"
                  << std::setw(12) << "nhops_mean" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        // Run benchmark
        auto results = run_benchmark(
            hnsw_index, flat_index, quant.get(),
            xq, nq, d, gt, gt_k, ds_cfg.k,
            ds_cfg.ef_search_values, ds_cfg.threads);

        // Save results to file
        std::string quant_params = quant->get_params_string();
        std::string result_filename = generate_result_filename(
            opts.algorithm, quant_params, ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction);
        std::string result_dir = "experiments/hnsw/results/" + opts.dataset + "/" + opts.algorithm;
        std::string result_path = result_dir + "/" + result_filename;

        save_results_to_file(
            result_path, opts.dataset, opts.algorithm, quant_params,
            ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction,
            nq, ds_cfg.k, results);
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete hnsw_index;

    return 0;
}
