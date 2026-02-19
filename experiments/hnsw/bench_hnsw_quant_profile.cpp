/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * HNSW + Quantization Cache Profiling Tool
 *
 * Two modes:
 *   train:   Train quantizer and save to disk.
 *   profile: Load quantizer + HNSW from disk, run bare search loop.
 *            (No reranking, no stats, no file I/O during search.)
 *
 * The profile mode is designed for cachegrind / perf:
 *   valgrind --tool=cachegrind ./bench_hnsw_quant_profile \
 *       --mode profile --dataset audio --algorithm pq --ef 10 --nq 1000
 *
 * Usage:
 *   # Step 1: train and save quantizer
 *   ./bench_hnsw_quant_profile --mode train --dataset audio --algorithm pq
 *
 *   # Step 2: profile with cachegrind
 *   valgrind --tool=cachegrind ./bench_hnsw_quant_profile \
 *       --mode profile --dataset audio --algorithm pq --ef 10
 */

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

/*****************************************************
 * Path helpers
 *****************************************************/

std::string make_params_string(
        const std::string& algo,
        const std::map<std::string, std::string>& params) {
    if (algo == "pq" || algo == "opq") {
        std::string M = params.count("M") ? params.at("M") : "16";
        std::string nbits = params.count("nbits") ? params.at("nbits") : "8";
        return "M" + M + "_nbits" + nbits;
    } else if (algo == "sq") {
        return params.count("qtype") ? params.at("qtype") : "QT_8bit";
    }
    // generic
    std::string s;
    for (const auto& [k, v] : params) {
        if (!s.empty()) s += "_";
        s += k + v;
    }
    return s;
}

std::string get_quant_path(
        const std::string& base_path,
        const std::string& algo,
        const std::string& params_str) {
    return base_path + "/index/quant_" + algo + "_" + params_str + ".faiss";
}

/*****************************************************
 * Train mode: create index, train, save
 *****************************************************/

faiss::Index* create_quant_index(
        const std::string& algo,
        size_t d,
        const std::map<std::string, std::string>& params) {
    if (algo == "pq") {
        size_t M = params.count("M") ? std::stoul(params.at("M")) : 16;
        size_t nbits = params.count("nbits") ? std::stoul(params.at("nbits")) : 8;
        return new faiss::IndexPQ(d, M, nbits);
    } else if (algo == "sq") {
        std::string qtype_str = params.count("qtype") ? params.at("qtype") : "QT_8bit";
        faiss::ScalarQuantizer::QuantizerType qtype = faiss::ScalarQuantizer::QT_8bit;
        if (qtype_str == "QT_4bit") qtype = faiss::ScalarQuantizer::QT_4bit;
        else if (qtype_str == "QT_6bit") qtype = faiss::ScalarQuantizer::QT_6bit;
        else if (qtype_str == "QT_8bit") qtype = faiss::ScalarQuantizer::QT_8bit;
        else if (qtype_str == "QT_fp16") qtype = faiss::ScalarQuantizer::QT_fp16;
        return new faiss::IndexScalarQuantizer(d, qtype);
    }
    throw std::runtime_error("Unsupported algorithm: " + algo);
}

/*****************************************************
 * Command line
 *****************************************************/

struct Options {
    std::string mode = "profile";  // "train" or "profile"
    std::string dataset = "sift1m";
    std::string algorithm = "pq";
    std::string config_dir = "./config";
    std::string data_path;
    int threads = -1;
    size_t ef = 10;
    size_t nq_limit = 0;  // 0 = all queries
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --mode <train|profile> [options]\n\n"
              << "Modes:\n"
              << "  train    Train quantizer and save to disk\n"
              << "  profile  Load quantizer + HNSW, run bare search loop\n\n"
              << "Options:\n"
              << "  --mode <train|profile>  Mode of operation\n"
              << "  --dataset <name>        Dataset name (e.g., sift1m, audio)\n"
              << "  --algorithm <name>      Algorithm name (pq, sq)\n"
              << "  --config-dir <path>     Config directory (default: ./config)\n"
              << "  --data-path <path>      Override dataset base path\n"
              << "  --threads <n>           Threads for training\n"
              << "  --ef <n>                efSearch for profile mode (default: 10)\n"
              << "  --nq <n>               Limit number of queries (default: all)\n"
              << "  --help                  Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --mode train --dataset audio --algorithm pq\n"
              << "  valgrind --tool=cachegrind " << prog
              << " --mode profile --dataset audio --algorithm pq --ef 10 --nq 1000\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
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
            opts.ef = std::stoul(argv[++i]);
        } else if (arg == "--nq" && i + 1 < argc) {
            opts.nq_limit = std::stoul(argv[++i]);
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
    }

    if (!opts.data_path.empty()) ds_cfg.base_path = opts.data_path;
    if (opts.threads > 0) ds_cfg.threads = opts.threads;

    // Load algorithm config (first param set only)
    std::string algo_config = opts.config_dir + "/" + opts.algorithm + ".conf";
    auto algo_params = ConfigParser::parse_algorithm(algo_config);

    std::map<std::string, std::string> params;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end() && !ait->second.empty()) {
        params = ait->second[0].params;
    } else {
        if (opts.algorithm == "pq")
            params = {{"M", "16"}, {"nbits", "8"}};
        else if (opts.algorithm == "sq")
            params = {{"qtype", "QT_8bit"}};
        else {
            std::cerr << "No config for " << opts.algorithm << "/" << opts.dataset << std::endl;
            return 1;
        }
    }

    std::string params_str = make_params_string(opts.algorithm, params);
    std::string quant_path = get_quant_path(ds_cfg.base_path, opts.algorithm, params_str);
    std::string hnsw_path = ds_cfg.get_hnsw_index_path();

    // ======================= TRAIN MODE =======================
    if (opts.mode == "train") {
        std::cout << "=== Train Mode ===" << std::endl;
        std::cout << "Dataset:   " << opts.dataset << std::endl;
        std::cout << "Algorithm: " << opts.algorithm << " (" << params_str << ")" << std::endl;
        std::cout << "Save to:   " << quant_path << std::endl;

        omp_set_num_threads(ds_cfg.threads);

        // Load base vectors
        size_t d, nb;
        float* xb = fvecs_read(ds_cfg.get_path(ds_cfg.base_file).c_str(), &d, &nb);
        if (!xb) {
            std::cerr << "Error: Failed to load base vectors" << std::endl;
            return 1;
        }
        std::cout << "Database: " << nb << " vectors, dim " << d << std::endl;

        // Create, train, save
        Timer timer;
        faiss::Index* idx = create_quant_index(opts.algorithm, d, params);
        idx->train(nb, xb);
        idx->add(nb, xb);
        std::cout << "Train+add: "
                  << timer.elapsed_ms() << " ms" << std::endl;

        // Ensure directory exists
        size_t last_slash = quant_path.rfind('/');
        if (last_slash != std::string::npos) {
            create_directory(quant_path.substr(0, last_slash));
        }

        faiss::write_index(idx, quant_path.c_str());
        std::cout << "Saved: " << quant_path << std::endl;

        delete idx;
        delete[] xb;
        return 0;
    }

    // ====================== PROFILE MODE ======================
    if (opts.mode == "profile") {
        std::cout << "=== Profile Mode ===" << std::endl;
        std::cout << "Dataset:   " << opts.dataset << std::endl;
        std::cout << "Algorithm: " << opts.algorithm << " (" << params_str << ")" << std::endl;
        std::cout << "ef:        " << opts.ef << std::endl;

        // Load queries only (no base vectors, no groundtruth)
        size_t d, nq;
        float* xq = fvecs_read(ds_cfg.get_path(ds_cfg.query_file).c_str(), &d, &nq);
        if (!xq) {
            std::cerr << "Error: Failed to load queries" << std::endl;
            return 1;
        }

        size_t nq_run = (opts.nq_limit > 0 && opts.nq_limit < nq) ? opts.nq_limit : nq;
        std::cout << "Queries:   " << nq_run << " / " << nq << std::endl;

        // Load HNSW index
        std::cout << "Loading HNSW: " << hnsw_path << std::endl;
        faiss::Index* hnsw_raw = faiss::read_index(hnsw_path.c_str());
        faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(hnsw_raw);
        if (!hnsw_index) {
            std::cerr << "Error: Not an IndexHNSW" << std::endl;
            return 1;
        }
        std::cout << "HNSW: ntotal=" << hnsw_index->ntotal << " d=" << hnsw_index->d << std::endl;

        // Load quantizer
        std::cout << "Loading quant: " << quant_path << std::endl;
        faiss::Index* quant_raw = faiss::read_index(quant_path.c_str());
        faiss::IndexFlatCodes* quant_index = dynamic_cast<faiss::IndexFlatCodes*>(quant_raw);
        if (!quant_index) {
            std::cerr << "Error: Loaded index is not IndexFlatCodes" << std::endl;
            return 1;
        }
        std::cout << "Quant: ntotal=" << quant_index->ntotal << std::endl;

        // Get distance computer
        auto quant_dc = std::unique_ptr<faiss::DistanceComputer>(
            quant_index->get_FlatCodesDistanceComputer());

        // Setup search
        const faiss::HNSW& hnsw = hnsw_index->hnsw;
        const size_t ef = opts.ef;

        faiss::SearchParametersHNSW search_params;
        search_params.efSearch = ef;

        std::vector<float> ef_distances(ef);
        std::vector<faiss::idx_t> ef_labels(ef);
        faiss::VisitedTable vt(hnsw_index->ntotal);

        using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

        std::cout << "Running " << nq_run << " queries, ef=" << ef << " ..." << std::endl;

        // === Bare search loop (this is what cachegrind measures) ===
        size_t total_ndis = 0;
        for (size_t i = 0; i < nq_run; i++) {
            quant_dc->set_query(xq + i * d);

            std::fill(ef_distances.begin(), ef_distances.end(),
                      std::numeric_limits<float>::max());
            std::fill(ef_labels.begin(), ef_labels.end(), (faiss::idx_t)-1);

            RH bres(1, ef_distances.data(), ef_labels.data(), ef);
            RH::SingleResultHandler res(bres);
            vt.advance();

            res.begin(0);
            faiss::HNSWStats stats = hnsw.search(
                *quant_dc, hnsw_index, res, vt, &search_params);
            res.end();

            total_ndis += stats.ndis;
        }

        std::cout << "Done. queries=" << nq_run
                  << " total_ndis=" << total_ndis
                  << " avg_ndis="
                  << (double)total_ndis / nq_run << std::endl;

        delete[] xq;
        delete hnsw_index;
        delete quant_raw;
        return 0;
    }

    std::cerr << "Unknown mode: " << opts.mode << " (use train or profile)" << std::endl;
    return 1;
}
