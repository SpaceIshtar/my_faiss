/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * HNSW Construction Benchmark
 *
 * Tests how different distance computation strategies during HNSW construction
 * affect the resulting graph quality. During construction, HNSW uses:
 *   - Search phase: find neighbor candidates (uses operator() on DistanceComputer)
 *   - Pruning phase: select best neighbors (uses symmetric_dis() on DistanceComputer)
 *
 * Construction modes tested (besides exact+exact in build_hnsw_index):
 *   - adc_sdc:   ADC for search + SDC for pruning (standard quantized HNSW)
 *   - adc_exact:  ADC for search + Exact for pruning
 *   - exact_sdc:  Exact for search + SDC for pruning
 *
 * After construction, all indexes are searched with exact distances to
 * evaluate graph quality.
 *
 * Usage:
 *   ./bench_construction_hnsw --dataset <name> --algorithm <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1m)
 *   --algorithm <name>     Algorithm (pq, sq, opq)
 *   --config-dir <path>    Config directory (default: ./config)
 *   --data-path <path>     Override dataset base path
 *   --threads <n>          Number of threads
 *   --ef <list>            Comma-separated ef search values
 *   --mode <list>          Comma-separated modes (adc_sdc,adc_exact,exact_sdc)
 *   --timeout <seconds>    Train+add timeout (default: 3600)
 *   --help                 Show this help
 *
 * Example:
 *   ./bench_construction_hnsw --dataset sift1m --algorithm pq
 *   ./bench_construction_hnsw --dataset sift1m --algorithm sq --mode adc_exact
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cxxabi.h>
#include <typeinfo>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

/*****************************************************
 * Construction modes
 *****************************************************/

enum class ConstructionMode {
    ADC_SDC,   // Quantized search (ADC) + Quantized pruning (SDC)
    ADC_EXACT, // Quantized search (ADC) + Exact pruning
    EXACT_SDC  // Exact search + Quantized pruning (SDC)
};

std::string mode_to_string(ConstructionMode mode) {
    switch (mode) {
        case ConstructionMode::ADC_SDC:
            return "adc_sdc";
        case ConstructionMode::ADC_EXACT:
            return "adc_exact";
        case ConstructionMode::EXACT_SDC:
            return "exact_sdc";
    }
    return "unknown";
}

std::string mode_description(ConstructionMode mode) {
    switch (mode) {
        case ConstructionMode::ADC_SDC:
            return "ADC search + SDC pruning";
        case ConstructionMode::ADC_EXACT:
            return "ADC search + Exact pruning";
        case ConstructionMode::EXACT_SDC:
            return "Exact search + SDC pruning";
    }
    return "unknown";
}

/*****************************************************
 * Hybrid Distance Computers
 *
 * These wrap two DistanceComputers to use different
 * distance strategies for search vs pruning phases.
 *****************************************************/

/// ADC for search (operator()), exact for pruning (symmetric_dis)
struct ADCExactDistanceComputer : faiss::DistanceComputer {
    std::unique_ptr<faiss::DistanceComputer> adc_dc;
    std::unique_ptr<faiss::DistanceComputer> exact_dc;

    ADCExactDistanceComputer(
            std::unique_ptr<faiss::DistanceComputer> adc,
            std::unique_ptr<faiss::DistanceComputer> exact)
            : adc_dc(std::move(adc)), exact_dc(std::move(exact)) {}

    void set_query(const float* x) override {
        adc_dc->set_query(x);
        exact_dc->set_query(x);
    }

    float operator()(faiss::idx_t i) override {
        return (*adc_dc)(i);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return exact_dc->symmetric_dis(i, j);
    }

    void distances_batch_4(
            const faiss::idx_t i0,
            const faiss::idx_t i1,
            const faiss::idx_t i2,
            const faiss::idx_t i3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        adc_dc->distances_batch_4(i0, i1, i2, i3, d0, d1, d2, d3);
    }
};

/// Exact for search (operator()), SDC for pruning (symmetric_dis)
/// Search uses exact (flat) distances, pruning uses quantized SDC.
struct ExactSDCDistanceComputer : faiss::DistanceComputer {
    std::unique_ptr<faiss::DistanceComputer> exact_dc;
    std::unique_ptr<faiss::DistanceComputer> quant_dc;

    ExactSDCDistanceComputer(
            std::unique_ptr<faiss::DistanceComputer> exact,
            std::unique_ptr<faiss::DistanceComputer> quant)
            : exact_dc(std::move(exact)), quant_dc(std::move(quant)) {}

    void set_query(const float* x) override {
        exact_dc->set_query(x);
        quant_dc->set_query(x);
    }

    float operator()(faiss::idx_t i) override {
        return (*exact_dc)(i);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return quant_dc->symmetric_dis(i, j);
    }

    void distances_batch_4(
            const faiss::idx_t i0,
            const faiss::idx_t i1,
            const faiss::idx_t i2,
            const faiss::idx_t i3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        exact_dc->distances_batch_4(i0, i1, i2, i3, d0, d1, d2, d3);
    }
};

/*****************************************************
 * Custom storage Index for HNSW construction
 *
 * Stores vectors in flat format (for later exact search)
 * but returns hybrid DistanceComputers during construction.
 * The quantized index is managed externally and must remain
 * valid for the lifetime of this object.
 *****************************************************/

struct ConstructionStorageIndex : faiss::Index {
    faiss::IndexFlat flat_index;
    faiss::Index* quant_index; // NOT owned
    ConstructionMode mode;
    mutable bool construction_phase_ = true;

    ConstructionStorageIndex(
            int d,
            faiss::MetricType metric,
            faiss::Index* quant_idx,
            ConstructionMode m)
            : faiss::Index(d, metric),
              flat_index(d, metric),
              quant_index(quant_idx),
              mode(m) {
        is_trained = true;
    }

    void add(faiss::idx_t n, const float* x) override {
        flat_index.add(n, x);
        ntotal = flat_index.ntotal;
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        // auto debug_dc = [](const char* label, const faiss::Index* idx, faiss::DistanceComputer* dc) {
        //     int status;
        //     char* idx_name = abi::__cxa_demangle(typeid(*idx).name(), nullptr, nullptr, &status);
        //     char* dc_name  = abi::__cxa_demangle(typeid(*dc).name(),  nullptr, nullptr, &status);
        //     fprintf(stderr, "[DEBUG get_distance_computer] %s: index type = %s, DC type = %s\n",
        //             label,
        //             idx_name ? idx_name : typeid(*idx).name(),
        //             dc_name  ? dc_name  : typeid(*dc).name());
        //     free(idx_name);
        //     free(dc_name);
        // };

        if (construction_phase_) {
            switch (mode) {
                case ConstructionMode::ADC_SDC: {
                    // Both search and pruning use quantized distances
                    auto* dc = quant_index->get_distance_computer();
                    // debug_dc(/"ADC_SDC quant_index", quant_index, dc);
                    return dc;
                }

                case ConstructionMode::ADC_EXACT: {
                    auto quant_dc = std::unique_ptr<faiss::DistanceComputer>(
                            quant_index->get_distance_computer());
                    auto flat_dc = std::unique_ptr<faiss::DistanceComputer>(
                            flat_index.get_distance_computer());
                    return new ADCExactDistanceComputer(
                            std::move(quant_dc), std::move(flat_dc));
                }

                case ConstructionMode::EXACT_SDC: {
                    auto flat_dc = std::unique_ptr<faiss::DistanceComputer>(
                            flat_index.get_distance_computer());
                    auto quant_dc = std::unique_ptr<faiss::DistanceComputer>(
                            quant_index->get_distance_computer());
                    return new ExactSDCDistanceComputer(
                            std::move(flat_dc), std::move(quant_dc));
                }
            }
        }
        // After construction: use flat DC for exact search
        return flat_index.get_distance_computer();
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        flat_index.search(n, x, k, distances, labels, params);
    }

    void reset() override {
        flat_index.reset();
        ntotal = 0;
    }
};

/*****************************************************
 * Quantized index factory
 *****************************************************/

struct QuantizerInfo {
    std::string name;
    std::string params_string;
    faiss::Index* index; // caller takes ownership
};

QuantizerInfo create_quantizer(
        const std::string& algorithm,
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    QuantizerInfo info;

    if (algorithm == "pq") {
        size_t M = 8, nbits = 8;
        auto it = params.find("M");
        if (it != params.end())
            M = std::stoul(it->second);
        it = params.find("nbits");
        if (it != params.end())
            nbits = std::stoul(it->second);

        info.index = new faiss::IndexPQ(d, M, nbits, metric);
        info.name = "PQ";
        std::ostringstream oss;
        oss << "M" << M << "_nbits" << nbits;
        info.params_string = oss.str();

    } else if (algorithm == "sq") {
        auto qtype = faiss::ScalarQuantizer::QT_8bit;
        auto it = params.find("qtype");
        if (it != params.end()) {
            const std::string& s = it->second;
            if (s == "QT_8bit" || s == "8bit")
                qtype = faiss::ScalarQuantizer::QT_8bit;
            else if (s == "QT_4bit" || s == "4bit")
                qtype = faiss::ScalarQuantizer::QT_4bit;
            else if (s == "QT_fp16" || s == "fp16")
                qtype = faiss::ScalarQuantizer::QT_fp16;
            else if (s == "QT_6bit" || s == "6bit")
                qtype = faiss::ScalarQuantizer::QT_6bit;
            else if (s == "QT_8bit_uniform")
                qtype = faiss::ScalarQuantizer::QT_8bit_uniform;
            else if (s == "QT_4bit_uniform")
                qtype = faiss::ScalarQuantizer::QT_4bit_uniform;
        }

        info.index = new faiss::IndexScalarQuantizer(d, qtype, metric);
        info.name = "SQ";
        auto it2 = params.find("qtype");
        info.params_string = (it2 != params.end()) ? it2->second : "QT_8bit";

    } else if (algorithm == "opq") {
        size_t M = 8, nbits = 8;
        auto it = params.find("M");
        if (it != params.end())
            M = std::stoul(it->second);
        it = params.find("nbits");
        if (it != params.end())
            nbits = std::stoul(it->second);

        auto* opq = new faiss::OPQMatrix(d, M);
        auto* pq_idx = new faiss::IndexPQ(d, M, nbits, metric);
        auto* idx = new faiss::IndexPreTransform(opq, pq_idx);
        idx->own_fields = true;

        info.index = idx;
        info.name = "OPQ";
        std::ostringstream oss;
        oss << "M" << M << "_nbits" << nbits;
        info.params_string = oss.str();

    } else {
        throw std::runtime_error(
                "Unsupported algorithm for construction benchmark: " +
                algorithm + ". Supported: pq, sq, opq");
    }

    return info;
}

/// Ensure SDC tables are computed for PQ-based quantizers.
/// PQ's symmetric_dis() requires precomputed SDC tables.
void prepare_sdc(faiss::Index* index) {
    // Direct IndexPQ
    auto* pq_idx = dynamic_cast<faiss::IndexPQ*>(index);
    if (pq_idx) {
        pq_idx->pq.compute_sdc_table();
        return;
    }
    // IndexPreTransform wrapping IndexPQ (for OPQ)
    auto* pretrans = dynamic_cast<faiss::IndexPreTransform*>(index);
    if (pretrans) {
        auto* inner_pq = dynamic_cast<faiss::IndexPQ*>(pretrans->index);
        if (inner_pq) {
            inner_pq->pq.compute_sdc_table();
        }
    }
    // SQ: symmetric_dis works without extra setup (computes code_distance)
}

/*****************************************************
 * Build HNSW with specified construction mode
 *****************************************************/

faiss::IndexHNSW* build_hnsw_with_mode(
        ConstructionMode mode,
        faiss::Index* quant_index,
        int d,
        int M,
        int efConstruction,
        size_t nb,
        const float* xb,
        bool verbose = true) {
    if (verbose) {
        std::cout << "[Building HNSW: " << mode_description(mode) << " (M=" << M
                  << ", efC=" << efConstruction << ")...]" << std::endl;
    }

    Timer timer;

    // Create custom storage that provides hybrid DCs during construction
    auto* storage = new ConstructionStorageIndex(
            d, faiss::METRIC_L2, quant_index, mode);

    // Create HNSW with custom storage
    auto* hnsw_index = new faiss::IndexHNSW(storage, M);
    hnsw_index->hnsw.efConstruction = efConstruction;
    hnsw_index->own_fields = true;

    // Build the graph
    hnsw_index->add(nb, xb);

    // Switch to flat-only mode for exact distance search
    storage->construction_phase_ = false;

    if (verbose) {
        std::cout << "Build time: " << std::fixed << std::setprecision(1)
                  << timer.elapsed_sec() << " seconds" << std::endl;
    }

    return hnsw_index;
}

/// Save HNSW index by copying graph to a standard IndexHNSWFlat.
/// ConstructionStorageIndex can't be serialized directly by FAISS.
void save_hnsw_index(
        faiss::IndexHNSW* hnsw_index,
        const std::string& path,
        int d,
        int M,
        size_t nb,
        const float* xb) {
    // Create directory
    size_t last_slash = path.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(path.substr(0, last_slash));
    }

    // Create a standard IndexHNSWFlat and copy the graph
    auto* save_idx = new faiss::IndexHNSWFlat(d, M, faiss::METRIC_L2);
    save_idx->hnsw = hnsw_index->hnsw; // deep copy of HNSW graph
    dynamic_cast<faiss::IndexFlat*>(save_idx->storage)->add(nb, xb);
    save_idx->ntotal = nb;

    faiss::write_index(save_idx, path.c_str());
    delete save_idx;

    std::cout << "Index saved to: " << path << std::endl;
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
    SearchStats stats;
};

void print_percentile_stats(
        std::ostream& os,
        const std::string& name,
        const PercentileStats& stats) {
    os << "  " << name << ":\n"
       << "    min:      " << std::fixed << std::setprecision(2) << stats.min
       << "\n"
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
        ConstructionMode mode,
        int hnsw_M,
        int hnsw_efConstruction,
        double build_time_sec,
        size_t nq,
        size_t k,
        const std::vector<BenchmarkResult>& results) {
    size_t last_slash = filepath.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(filepath.substr(0, last_slash));
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << std::endl;
        return;
    }

    ofs << "========================================\n"
        << "HNSW Construction Benchmark Results\n"
        << "========================================\n\n";

    ofs << "[Configuration]\n"
        << "  Dataset:           " << dataset_name << "\n"
        << "  Algorithm:         " << algorithm_name << "\n"
        << "  Quant Params:      " << quant_params << "\n"
        << "  Construction Mode: " << mode_to_string(mode) << " ("
        << mode_description(mode) << ")\n"
        << "  HNSW M:            " << hnsw_M << "\n"
        << "  HNSW efConstruct:  " << hnsw_efConstruction << "\n"
        << "  Build Time:        " << std::fixed << std::setprecision(1)
        << build_time_sec << " s\n"
        << "  Num Queries:       " << nq << "\n"
        << "  Recall@k:          " << k << "\n\n";

    ofs << "[Summary]\n"
        << std::setw(8) << "ef" << std::setw(12) << "QPS" << std::setw(12)
        << "Recall" << std::setw(12) << "Latency(ms)" << std::setw(12)
        << "ndis_mean" << std::setw(12) << "nhops_mean"
        << "\n"
        << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        ofs << std::setw(8) << r.ef << std::setw(12) << std::fixed
            << std::setprecision(0) << r.qps << std::setw(12)
            << std::setprecision(4) << r.recall << std::setw(12)
            << std::setprecision(3) << r.latency_ms << std::setw(12)
            << std::setprecision(1) << r.stats.ndis_stats.mean << std::setw(12)
            << std::setprecision(1) << r.stats.nhops_stats.mean << "\n";
    }
    ofs << "\n";

    ofs << "[Detailed Statistics]\n";
    for (const auto& r : results) {
        ofs << "\n--- ef = " << r.ef << " ---\n"
            << "  QPS:          " << std::fixed << std::setprecision(2)
            << r.qps << "\n"
            << "  Recall@" << k << ":    " << std::setprecision(4) << r.recall
            << "\n"
            << "  Total Time:   " << std::setprecision(2) << r.total_time_ms
            << " ms\n"
            << "  Avg Latency:  " << std::setprecision(4) << r.latency_ms
            << " ms\n\n";

        print_percentile_stats(
                ofs, "ndis (distance computations)", r.stats.ndis_stats);
        ofs << "\n";
        print_percentile_stats(
                ofs, "nhops (graph edges traversed)", r.stats.nhops_stats);
    }

    ofs << "\n========================================\n"
        << "End of Results\n"
        << "========================================\n";

    ofs.close();
    std::cout << "Results saved to: " << filepath << std::endl;
}

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1m";
    std::string algorithm = "pq";
    std::string config_dir = "./config";
    std::string data_path;
    int threads = -1;
    std::vector<size_t> ef_values;
    std::vector<std::string> modes; // empty = all modes
    double timeout_sec = 3600.0;
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout
            << "Usage: " << prog
            << " --dataset <name> --algorithm <name> [options]\n\n"
            << "Options:\n"
            << "  --dataset <name>       Dataset name (e.g., sift1m)\n"
            << "  --algorithm <name>     Algorithm (pq, sq, opq)\n"
            << "  --config-dir <path>    Config directory (default: ./config)\n"
            << "  --data-path <path>     Override dataset base path\n"
            << "  --threads <n>          Number of threads\n"
            << "  --ef <list>            Comma-separated ef search values\n"
            << "  --mode <list>          Comma-separated modes "
               "(adc_sdc,adc_exact,exact_sdc)\n"
            << "  --timeout <seconds>    Train+add timeout (default: 3600)\n"
            << "  --help                 Show this help\n\n"
            << "Construction Modes:\n"
            << "  adc_sdc    Quantized ADC search + Quantized SDC pruning\n"
            << "  adc_exact  Quantized ADC search + Exact distance pruning\n"
            << "  exact_sdc  Exact distance search + Quantized SDC pruning\n\n"
            << "Example:\n"
            << "  " << prog << " --dataset sift1m --algorithm pq\n"
            << "  " << prog
            << " --dataset sift1m --algorithm sq --mode adc_exact,exact_sdc\n";
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
            std::string s = argv[++i];
            std::istringstream iss(s);
            std::string token;
            while (std::getline(iss, token, ',')) {
                opts.ef_values.push_back(std::stoul(token));
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string s = argv[++i];
            std::istringstream iss(s);
            std::string token;
            while (std::getline(iss, token, ',')) {
                opts.modes.push_back(token);
            }
        } else if (arg == "--timeout" && i + 1 < argc) {
            opts.timeout_sec = std::stod(argv[++i]);
        }
    }

    return opts;
}

std::vector<ConstructionMode> parse_modes(
        const std::vector<std::string>& mode_strs) {
    if (mode_strs.empty()) {
        return {ConstructionMode::ADC_SDC,
                ConstructionMode::ADC_EXACT,
                ConstructionMode::EXACT_SDC};
    }

    std::vector<ConstructionMode> modes;
    for (const auto& s : mode_strs) {
        if (s == "adc_sdc")
            modes.push_back(ConstructionMode::ADC_SDC);
        else if (s == "adc_exact")
            modes.push_back(ConstructionMode::ADC_EXACT);
        else if (s == "exact_sdc")
            modes.push_back(ConstructionMode::EXACT_SDC);
        else
            std::cerr << "Warning: Unknown mode '" << s << "', skipping"
                      << std::endl;
    }
    return modes;
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
    std::cout << "HNSW Construction Benchmark" << std::endl;
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
        ds_cfg.name = opts.dataset;
        ds_cfg.base_path = "/data/local/embedding_dataset/" + opts.dataset;
        ds_cfg.ef_search_values = get_default_ef_search();
    }

    // Apply overrides
    if (!opts.data_path.empty())
        ds_cfg.base_path = opts.data_path;
    if (opts.threads > 0)
        ds_cfg.threads = opts.threads;
    if (!opts.ef_values.empty())
        ds_cfg.ef_search_values = opts.ef_values;
    if (ds_cfg.ef_search_values.empty())
        ds_cfg.ef_search_values = get_default_ef_search();

    omp_set_num_threads(ds_cfg.threads);

    std::vector<ConstructionMode> modes = parse_modes(opts.modes);

    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "HNSW M: " << ds_cfg.hnsw_M << std::endl;
    std::cout << "HNSW efConstruction: " << ds_cfg.hnsw_efConstruction
              << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;
    std::cout << "Modes: ";
    for (size_t i = 0; i < modes.size(); i++) {
        if (i > 0)
            std::cout << ", ";
        std::cout << mode_to_string(modes[i]);
    }
    std::cout << std::endl;

    // Load algorithm config
    std::string algo_config = opts.config_dir + "/" + opts.algorithm + ".conf";
    auto algo_params = ConfigParser::parse_algorithm(algo_config);

    std::vector<AlgorithmParamSet> param_sets;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end()) {
        param_sets = ait->second;
    }

    // Defaults if no config
    if (param_sets.empty()) {
        if (opts.algorithm == "pq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "32"}, {"nbits", "8"}}});
        } else if (opts.algorithm == "sq") {
            param_sets.push_back({.params = {{"qtype", "QT_8bit"}}});
            param_sets.push_back({.params = {{"qtype", "QT_4bit"}}});
        } else if (opts.algorithm == "opq") {
            param_sets.push_back({.params = {{"M", "8"}, {"nbits", "8"}}});
            param_sets.push_back({.params = {{"M", "16"}, {"nbits", "8"}}});
        }
    }

    // Load data
    std::cout << "\n[Loading data...]" << std::endl;
    size_t d, nb, nq, gt_k;
    float* xb = fvecs_read(ds_cfg.get_path(ds_cfg.base_file).c_str(), &d, &nb);
    float* xq =
            fvecs_read(ds_cfg.get_path(ds_cfg.query_file).c_str(), &d, &nq);
    int* gt = ivecs_read(
            ds_cfg.get_path(ds_cfg.groundtruth_file).c_str(), &gt_k, &nq);

    if (!xb || !xq || !gt) {
        std::cerr << "Error: Failed to load data files" << std::endl;
        return 1;
    }

    std::cout << "Database: " << nb << " vectors, dimension " << d
              << std::endl;
    std::cout << "Queries: " << nq << " vectors" << std::endl;

    const size_t k = ds_cfg.k;

    // Run for each parameter set
    for (const auto& ps : param_sets) {
        std::cout << "\n=================================================="
                  << std::endl;
        std::cout << "Quantizer: " << opts.algorithm
                  << " | Params: " << ps.to_string() << std::endl;
        std::cout << "=================================================="
                  << std::endl;

        // Create quantizer
        QuantizerInfo qinfo;
        try {
            qinfo = create_quantizer(
                    opts.algorithm, d, faiss::METRIC_L2, ps.params);
        } catch (const std::exception& e) {
            std::cerr << "Error creating quantizer: " << e.what() << std::endl;
            continue;
        }

        std::unique_ptr<faiss::Index> quant_idx(qinfo.index);

        // Try loading pre-trained quantizer from disk
        std::string quant_save_path = ds_cfg.base_path + "/index/quant_" +
                                      opts.algorithm + "_" +
                                      qinfo.params_string + ".faiss";

        bool loaded = false;
        if (file_exists(quant_save_path)) {
            std::cout << "[Loading quantizer from " << quant_save_path << "...]" << std::endl;
            try {
                faiss::Index* loaded_idx = faiss::read_index(quant_save_path.c_str());
                quant_idx.reset(loaded_idx);
                std::cout << "  Loaded, ntotal=" << loaded_idx->ntotal << std::endl;
                loaded = true;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load quantizer: " << e.what()
                          << ". Training from scratch." << std::endl;
                quant_idx.reset(qinfo.index);
            }
        }

        if (!loaded) {
            // Train and add with timeout
            std::cout << "[Training " << qinfo.name << " " << qinfo.params_string
                      << "...]" << std::endl;

            std::atomic<bool> train_done{false};
            faiss::Index* quant_ptr = quant_idx.get();
            int num_threads = ds_cfg.threads;

            std::thread train_thread(
                    [quant_ptr, nb, xb, num_threads, &train_done]() {
                        omp_set_num_threads(num_threads);
                        quant_ptr->train(nb, xb);
                        // quant_ptr->add(nb, xb);
                        train_done.store(true, std::memory_order_release);
                    });

            Timer train_timer;
            while (!train_done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                if (train_timer.elapsed_sec() >= opts.timeout_sec)
                    break;
            }

            if (!train_done.load(std::memory_order_acquire)) {
                std::cout << "WARNING: Train+add exceeded timeout ("
                          << std::fixed << std::setprecision(0)
                          << train_timer.elapsed_sec() << "s). Skipping."
                          << std::endl;
                train_thread.detach();
                continue;
            }

            train_thread.join();
            double train_time = train_timer.elapsed_sec();
            std::cout << "Train+add time: " << std::fixed << std::setprecision(1)
                      << train_time << " seconds" << std::endl;
        }

        // Ensure SDC tables are computed (required for PQ symmetric_dis)
        prepare_sdc(quant_idx.get());

        // For each construction mode
        for (ConstructionMode mode : modes) {
            std::cout
                    << "\n--------------------------------------------------"
                    << std::endl;
            std::cout << "Construction mode: " << mode_to_string(mode) << " ("
                      << mode_description(mode) << ")" << std::endl;
            std::cout << "--------------------------------------------------"
                      << std::endl;

            // Check if index already exists (skip rebuild)
            std::string index_filename =
                    "hnsw_construction_" + opts.algorithm + "_" +
                    qinfo.params_string + "_" + mode_to_string(mode) + "_M" +
                    std::to_string(ds_cfg.hnsw_M) + "_efc" +
                    std::to_string(ds_cfg.hnsw_efConstruction) + ".faissindex";
            std::string index_path =
                    ds_cfg.base_path + "/index/" + index_filename;

            faiss::IndexHNSW* hnsw_index = nullptr;
            double build_time = 0;
            bool loaded_from_file = false;

            if (file_exists(index_path)) {
                std::cout << "[Loading existing index from " << index_path
                          << "...]" << std::endl;
                Timer load_timer;
                try {
                    std::unique_ptr<faiss::Index> loaded(
                            faiss::read_index(index_path.c_str()));
                    hnsw_index =
                            dynamic_cast<faiss::IndexHNSW*>(loaded.get());
                    if (hnsw_index) {
                        loaded.release();
                        loaded_from_file = true;
                        std::cout << "Loaded in " << std::fixed
                                  << std::setprecision(1)
                                  << load_timer.elapsed_sec() << "s"
                                  << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to load index: " << e.what()
                              << ". Rebuilding." << std::endl;
                    hnsw_index = nullptr;
                }
            }

            if (!hnsw_index) {
                // Build HNSW with this construction mode
                Timer build_timer;
                hnsw_index = build_hnsw_with_mode(
                        mode,
                        quant_idx.get(),
                        d,
                        ds_cfg.hnsw_M,
                        ds_cfg.hnsw_efConstruction,
                        nb,
                        xb);
                build_time = build_timer.elapsed_sec();

                if (!hnsw_index) {
                    std::cerr << "Error building HNSW index" << std::endl;
                    continue;
                }

                // Save by copying graph to a standard IndexHNSWFlat
                // std::cout << "[Saving index...]" << std::endl;
                // save_hnsw_index(
                //         hnsw_index, index_path, d, ds_cfg.hnsw_M, nb, xb);
            }

            // Search with exact distances
            // If loaded from file, storage is already IndexFlat.
            // If just built, ConstructionStorageIndex is in flat mode.
            std::cout << "\n[Searching with exact distances...]" << std::endl;
            std::cout << std::setw(8) << "ef" << std::setw(12) << "QPS"
                      << std::setw(12) << "Recall@" << k << std::setw(12)
                      << "Latency(ms)" << std::setw(12) << "ndis_mean"
                      << std::setw(12) << "nhops_mean" << std::endl;
            std::cout << std::string(68, '-') << std::endl;

            std::vector<BenchmarkResult> results;

            for (size_t ef : ds_cfg.ef_search_values) {
                std::vector<faiss::idx_t> result_ids(nq * k);
                std::vector<float> result_dists(nq * k);

                faiss::SearchParametersHNSW params;
                params.efSearch = ef;

                // Warmup
                hnsw_index->search(
                        std::min((faiss::idx_t)100, (faiss::idx_t)nq),
                        xq,
                        k,
                        result_dists.data(),
                        result_ids.data(),
                        &params);

                // Timed search
                faiss::hnsw_stats.reset();
                Timer search_timer;
                hnsw_index->search(
                        nq,
                        xq,
                        k,
                        result_dists.data(),
                        result_ids.data(),
                        &params);

                double search_time = search_timer.elapsed_ms();
                double qps = nq * 1000.0 / search_time;
                double latency = search_time / nq;
                float recall = compute_recall_at_k(
                        nq, k, result_ids.data(), gt, gt_k);

                // ndis/nhop from global HNSW stats (mean per query)
                SearchStats ss;
                double mean_ndis = (double)faiss::hnsw_stats.ndis / nq;
                double mean_nhops = (double)faiss::hnsw_stats.nhops / nq;
                ss.ndis_stats.mean = mean_ndis;
                ss.ndis_stats.p50 = mean_ndis;
                ss.nhops_stats.mean = mean_nhops;
                ss.nhops_stats.p50 = mean_nhops;

                results.push_back(
                        {ef, qps, recall, latency, search_time, ss});

                std::cout << std::setw(8) << ef << std::setw(12) << std::fixed
                          << std::setprecision(0) << qps << std::setw(12)
                          << std::setprecision(4) << recall << std::setw(12)
                          << std::setprecision(3) << latency << std::setw(12)
                          << std::setprecision(1) << mean_ndis << std::setw(12)
                          << std::setprecision(1) << mean_nhops << std::endl;
            }

            // Save results
            std::string result_filename =
                    opts.algorithm + "_" + qinfo.params_string + "_" +
                    mode_to_string(mode) + "_M" +
                    std::to_string(ds_cfg.hnsw_M) + "_efc" +
                    std::to_string(ds_cfg.hnsw_efConstruction) + ".txt";
            std::string result_dir = "experiments/hnsw/results/" +
                    opts.dataset + "/construction/" + opts.algorithm;
            std::string result_path = result_dir + "/" + result_filename;

            save_results_to_file(
                    result_path,
                    opts.dataset,
                    opts.algorithm,
                    qinfo.params_string,
                    mode,
                    ds_cfg.hnsw_M,
                    ds_cfg.hnsw_efConstruction,
                    build_time,
                    nq,
                    k,
                    results);

            delete hnsw_index;
        }
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}
