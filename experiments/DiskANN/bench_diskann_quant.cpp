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
 *   --algorithm <name>         Algorithm name (pq, sq, opq, osq, rq, lsq, prq, plsq, vaq, rabitq)
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
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
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
#include "../hnsw/include/osq_wrapper.h"
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

namespace {

std::string to_lower_copy(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string infer_vector_format(const std::string& configured, const std::string& path) {
    std::string fmt = to_lower_copy(configured);
    if (!fmt.empty() && fmt != "auto") {
        return fmt;
    }
    std::string lower_path = to_lower_copy(path);
    if (ends_with(lower_path, ".fvecs")) return "fvecs";
    if (ends_with(lower_path, ".u8bin")) return "u8bin";
    if (ends_with(lower_path, ".fbin")) return "fbin";
    return "fvecs";
}

std::string infer_groundtruth_format(const std::string& configured, const std::string& path) {
    std::string fmt = to_lower_copy(configured);
    if (!fmt.empty() && fmt != "auto") {
        return fmt;
    }
    std::string lower_path = to_lower_copy(path);
    if (ends_with(lower_path, ".ivecs")) return "ivecs";
    if (ends_with(lower_path, ".ibin")) return "ibin";
    return "truthset";
}

void read_u8bin_header(const std::string& path, size_t& n, size_t& d) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open u8bin file: " + path);
    }
    int32_t n_i32 = 0, d_i32 = 0;
    ifs.read(reinterpret_cast<char*>(&n_i32), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&d_i32), sizeof(int32_t));
    if (!ifs || n_i32 <= 0 || d_i32 <= 0) {
        throw std::runtime_error("Invalid u8bin header: " + path);
    }
    n = static_cast<size_t>(n_i32);
    d = static_cast<size_t>(d_i32);
}

void read_fbin_header(const std::string& path, size_t& n, size_t& d) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open fbin file: " + path);
    }
    int32_t n_i32 = 0, d_i32 = 0;
    ifs.read(reinterpret_cast<char*>(&n_i32), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&d_i32), sizeof(int32_t));
    if (!ifs || n_i32 <= 0 || d_i32 <= 0) {
        throw std::runtime_error("Invalid fbin header: " + path);
    }
    n = static_cast<size_t>(n_i32);
    d = static_cast<size_t>(d_i32);
}

float* load_query_as_float(
        const std::string& query_path,
        const std::string& configured_format,
        size_t& d,
        size_t& nq) {
    const std::string fmt = infer_vector_format(configured_format, query_path);
    if (fmt == "fvecs") {
        return fvecs_read(query_path.c_str(), &d, &nq);
    }
    if (fmt == "fbin") {
        float* xq = nullptr;
        diskann::load_bin<float>(query_path, xq, nq, d);
        return xq;
    }
    if (fmt == "u8bin") {
        uint8_t* xq_u8 = nullptr;
        diskann::load_bin<uint8_t>(query_path, xq_u8, nq, d);
        float* xq = new float[nq * d];
        for (size_t i = 0; i < nq * d; ++i) {
            xq[i] = static_cast<float>(xq_u8[i]);
        }
        delete[] xq_u8;
        return xq;
    }
    throw std::runtime_error("Unsupported query format: " + fmt);
}

uint8_t* load_query_as_u8(
        const std::string& query_path,
        const std::string& configured_format,
        size_t& d,
        size_t& nq) {
    const std::string fmt = infer_vector_format(configured_format, query_path);
    if (fmt == "u8bin") {
        uint8_t* xq = nullptr;
        diskann::load_bin<uint8_t>(query_path, xq, nq, d);
        return xq;
    }
    if (fmt == "fvecs" || fmt == "fbin") {
        float* xqf = nullptr;
        if (fmt == "fvecs") {
            xqf = fvecs_read(query_path.c_str(), &d, &nq);
        } else {
            diskann::load_bin<float>(query_path, xqf, nq, d);
        }
        if (!xqf) {
            return nullptr;
        }
        uint8_t* xq = new uint8_t[nq * d];
        for (size_t i = 0; i < nq * d; ++i) {
            float v = xqf[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            xq[i] = static_cast<uint8_t>(v);
        }
        delete[] xqf;
        return xq;
    }
    throw std::runtime_error("Unsupported query format: " + fmt);
}

int* load_groundtruth_ids(
        const std::string& gt_path,
        const std::string& configured_format,
        size_t& gt_k,
        size_t expected_nq) {
    const std::string fmt = infer_groundtruth_format(configured_format, gt_path);
    size_t gt_nq = 0;

    if (fmt == "ivecs") {
        int* gt = ivecs_read(gt_path.c_str(), &gt_k, &gt_nq);
        if (!gt) {
            throw std::runtime_error("Failed to load ivecs groundtruth: " + gt_path);
        }
        if (gt_nq != expected_nq) {
            throw std::runtime_error("Groundtruth nq mismatch with query count");
        }
        return gt;
    }

    if (fmt == "ibin") {
        uint32_t* gt_u32 = nullptr;
        diskann::load_bin<uint32_t>(gt_path, gt_u32, gt_nq, gt_k);
        if (gt_nq != expected_nq) {
            delete[] gt_u32;
            throw std::runtime_error("Groundtruth nq mismatch with query count");
        }
        int* gt = new int[gt_nq * gt_k];
        for (size_t i = 0; i < gt_nq * gt_k; ++i) {
            if (gt_u32[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                delete[] gt_u32;
                delete[] gt;
                throw std::runtime_error("Groundtruth id exceeds int range");
            }
            gt[i] = static_cast<int>(gt_u32[i]);
        }
        delete[] gt_u32;
        return gt;
    }

    if (fmt == "truthset") {
        uint32_t* gt_ids = nullptr;
        float* gt_dists = nullptr;
        diskann::load_truthset(gt_path, gt_ids, gt_dists, gt_nq, gt_k);
        if (gt_nq != expected_nq) {
            delete[] gt_ids;
            delete[] gt_dists;
            throw std::runtime_error("Groundtruth nq mismatch with query count");
        }
        int* gt = new int[gt_nq * gt_k];
        for (size_t i = 0; i < gt_nq * gt_k; ++i) {
            if (gt_ids[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                delete[] gt_ids;
                delete[] gt_dists;
                delete[] gt;
                throw std::runtime_error("Groundtruth id exceeds int range");
            }
            gt[i] = static_cast<int>(gt_ids[i]);
        }
        delete[] gt_ids;
        delete[] gt_dists;
        return gt;
    }

    throw std::runtime_error("Unsupported groundtruth format: " + fmt);
}

} // namespace

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

template <typename T>
std::vector<BenchmarkResult> run_benchmark(
        diskann::PQFlashIndex<T>* flash_index,
        const T* xq,
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
    } else if (algorithm == "osq") {
        return create_osq_wrapper(d, metric, params);
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
              << "  --algorithm <name>         Algorithm (pq, opq, osq, sq, rq, lsq, prq, plsq, vaq, rabitq, saq)\n"
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
        } else if (opts.algorithm == "osq") {
            param_sets.push_back({.params = {{"encoding", "PACKED_NIBBLE"}, {"similarity", "EUCLIDEAN"}}});
            param_sets.push_back({.params = {{"encoding", "SEVEN_BIT"}, {"similarity", "EUCLIDEAN"}}});
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
    size_t d = 0, nb = 0, nq = 0, gt_k = 0;
    size_t query_d = 0;
    float* xb = nullptr;
    float* xq_float = nullptr;
    uint8_t* xq_u8 = nullptr;
    int* gt = nullptr;
    const std::string base_path = ds_cfg.get_path(ds_cfg.base_file);
    const std::string query_path = ds_cfg.get_path(ds_cfg.query_file);
    const std::string gt_path = ds_cfg.get_path(ds_cfg.groundtruth_file);
    const std::string base_fmt = infer_vector_format(ds_cfg.base_format, base_path);
    const std::string data_type = to_lower_copy(ds_cfg.data_type);
    const bool use_uint8_index = (data_type == "uint8" || data_type == "u8");

    try {
        if (base_fmt == "u8bin") {
            read_u8bin_header(base_path, nb, d);
            std::cout << "Base format: u8bin (streaming mode)" << std::endl;
        } else if (base_fmt == "fbin") {
            read_fbin_header(base_path, nb, d);
            xb = nullptr;
            std::cout << "Base format: fbin (streaming mode)" << std::endl;
        } else {
            xb = fvecs_read(base_path.c_str(), &d, &nb);
        }

        if (use_uint8_index) {
            xq_u8 = load_query_as_u8(query_path, ds_cfg.query_format, query_d, nq);
        } else {
            xq_float = load_query_as_float(query_path, ds_cfg.query_format, query_d, nq);
        }
        if (query_d != d) {
            throw std::runtime_error(
                "Dimension mismatch: base d=" + std::to_string(d) +
                ", query d=" + std::to_string(query_d));
        }
        gt = load_groundtruth_ids(gt_path, ds_cfg.groundtruth_format, gt_k, nq);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data files: " << e.what() << std::endl;
        delete[] xb;
        delete[] xq_float;
        delete[] xq_u8;
        delete[] gt;
        return 1;
    }

    if (((use_uint8_index && !xq_u8) || (!use_uint8_index && !xq_float)) ||
        !gt || ((base_fmt != "u8bin" && base_fmt != "fbin") && !xb)) {
        std::cerr << "Error: Failed to load data files" << std::endl;
        delete[] xb;
        delete[] xq_float;
        delete[] xq_u8;
        delete[] gt;
        return 1;
    }

    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;
    std::cout << "Queries: " << nq << " vectors" << std::endl;

    // Load DiskANN disk index
    std::string index_prefix = ds_cfg.get_diskann_index_prefix();
    std::cout << "\n[Loading DiskANN disk index...]" << std::endl;
    std::cout << "Index prefix: " << index_prefix << std::endl;

    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    Timer timer;

    auto run_with_index = [&](auto& flash_index, const auto* xq) -> int {
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
            if (base_fmt == "u8bin") {
                size_t train_n = ds_cfg.sample_num > 0 ? std::min(ds_cfg.sample_num, nb) : 0;
                if (train_n == 0) {
                    train_n = std::min<size_t>(nb, 200000);
                    std::cout << "sample_num is not set; auto-using sample_num=" << train_n
                              << " for quantizer training on u8bin data" << std::endl;
                }

                std::ifstream base_ifs(base_path, std::ios::binary);
                if (!base_ifs.is_open()) {
                    std::cerr << "Error: failed to open base u8bin file: " << base_path << std::endl;
                    return 1;
                }
                base_ifs.seekg(2 * sizeof(int32_t), std::ios::beg);

                std::vector<uint8_t> sample_u8(train_n * d);
                base_ifs.read(reinterpret_cast<char*>(sample_u8.data()), sample_u8.size());
                if (!base_ifs || static_cast<size_t>(base_ifs.gcount()) != sample_u8.size()) {
                    std::cerr << "Error: failed to read sample vectors from u8bin base file" << std::endl;
                    return 1;
                }
                std::vector<float> sample_f(train_n * d);
                for (size_t i = 0; i < sample_u8.size(); ++i) {
                    sample_f[i] = static_cast<float>(sample_u8[i]);
                }
                quant->train(train_n, sample_f.data());

                const size_t batch_size = std::max<size_t>(1, ds_cfg.add_batch_size);
                std::vector<uint8_t> batch_u8(batch_size * d);
                std::vector<float> batch_f(batch_size * d);
                base_ifs.clear();
                base_ifs.seekg(2 * sizeof(int32_t), std::ios::beg);

                size_t added = 0;
                while (added < nb) {
                    const size_t cur = std::min(batch_size, nb - added);
                    const size_t cur_bytes = cur * d;
                    base_ifs.read(reinterpret_cast<char*>(batch_u8.data()), cur_bytes);
                    if (!base_ifs || static_cast<size_t>(base_ifs.gcount()) != cur_bytes) {
                        std::cerr << "Error: failed while batch-reading u8bin base file at vector " << added << std::endl;
                        return 1;
                    }
                    for (size_t i = 0; i < cur_bytes; ++i) {
                        batch_f[i] = static_cast<float>(batch_u8[i]);
                    }
                    quant->add(cur, batch_f.data());
                    added += cur;
                }
            } else if (base_fmt == "fbin") {
                size_t train_n = ds_cfg.sample_num > 0 ? std::min(ds_cfg.sample_num, nb) : 0;
                if (train_n == 0) {
                    train_n = std::min<size_t>(nb, 200000);
                    std::cout << "sample_num is not set; auto-using sample_num=" << train_n
                              << " for quantizer training on fbin data" << std::endl;
                }

                std::ifstream base_ifs(base_path, std::ios::binary);
                if (!base_ifs.is_open()) {
                    std::cerr << "Error: failed to open base fbin file: " << base_path << std::endl;
                    return 1;
                }
                base_ifs.seekg(2 * sizeof(int32_t), std::ios::beg);

                std::vector<float> sample_f(train_n * d);
                const size_t sample_bytes = sample_f.size() * sizeof(float);
                base_ifs.read(reinterpret_cast<char*>(sample_f.data()), sample_bytes);
                if (!base_ifs || static_cast<size_t>(base_ifs.gcount()) != sample_bytes) {
                    std::cerr << "Error: failed to read sample vectors from fbin base file" << std::endl;
                    return 1;
                }
                quant->train(train_n, sample_f.data());

                const size_t batch_size = std::max<size_t>(1, ds_cfg.add_batch_size);
                std::vector<float> batch_f(batch_size * d);
                base_ifs.clear();
                base_ifs.seekg(2 * sizeof(int32_t), std::ios::beg);

                size_t added = 0;
                while (added < nb) {
                    const size_t cur = std::min(batch_size, nb - added);
                    const size_t cur_bytes = cur * d * sizeof(float);
                    base_ifs.read(reinterpret_cast<char*>(batch_f.data()), cur_bytes);
                    if (!base_ifs || static_cast<size_t>(base_ifs.gcount()) != cur_bytes) {
                        std::cerr << "Error: failed while batch-reading fbin base file at vector " << added << std::endl;
                        return 1;
                    }
                    quant->add(cur, batch_f.data());
                    added += cur;
                }
            } else {
                const size_t train_n = ds_cfg.sample_num > 0 ? std::min(ds_cfg.sample_num, nb) : nb;
                quant->train(train_n, xb);
                std::cout << "Training ends on samples, start one-shot add" << std::endl;
                quant->add(nb, xb);
            }
            std::cout << "Train+add time: " << timer.elapsed_ms() << " ms" << std::endl;

            flash_index->set_distance_computer_factory([&quant]() {
                return quant->get_distance_computer();
            });

            std::cout << "\n" << std::setw(8) << "L"
                      << std::setw(12) << "QPS"
                      << std::setw(12) << "Recall@" << ds_cfg.k
                      << std::setw(12) << "Latency(ms)"
                      << std::setw(12) << "ndis_mean"
                      << std::setw(12) << "nhops_mean" << std::endl;
            std::cout << std::string(68, '-') << std::endl;

            auto results = run_benchmark(
                flash_index.get(), xq, nq, d, gt, gt_k, ds_cfg.k,
                ds_cfg.L_search_values, ds_cfg.beam_width, ds_cfg.threads);

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
        return 0;
    };

    int status = 0;
    if (use_uint8_index) {
        auto flash_index = std::make_unique<diskann::PQFlashIndex<uint8_t>>(reader, diskann::Metric::L2);
        flash_index->set_skip_in_memory_pq_data(true);
        status = run_with_index(flash_index, xq_u8);
    } else {
        auto flash_index = std::make_unique<diskann::PQFlashIndex<float>>(reader, diskann::Metric::L2);
        flash_index->set_skip_in_memory_pq_data(true);
        status = run_with_index(flash_index, xq_float);
    }
    if (status != 0) {
        delete[] xb;
        delete[] xq_float;
        delete[] xq_u8;
        delete[] gt;
        return status;
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq_float;
    delete[] xq_u8;
    delete[] gt;

    return 0;
}
