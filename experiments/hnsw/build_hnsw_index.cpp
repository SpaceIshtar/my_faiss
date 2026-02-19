/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Build HNSW index on a dataset, reading parameters from datasets.conf.
 *
 * Usage:
 *   ./build_hnsw_index --dataset <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1m, gist1m)
 *   --config-dir <path>    Config directory (default: ./config)
 *   --data-path <path>     Override dataset base path
 *   --M <n>                Override HNSW M parameter
 *   --efConstruction <n>   Override HNSW efConstruction parameter
 *   --help                 Show this help
 *
 * Output files stored in:
 *   <base_path>/index/hnsw_M{M}_ef{efConstruction}.faissindex
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include <faiss/IndexHNSW.h>
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
    std::string data_path;  // override
    int M = -1;             // -1 = use config
    int efConstruction = -1;
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>       Dataset name (e.g., sift1m, gist1m)\n"
              << "  --config-dir <path>    Config directory (default: ./config)\n"
              << "  --data-path <path>     Override dataset base path\n"
              << "  --M <n>               Override HNSW M parameter\n"
              << "  --efConstruction <n>   Override HNSW efConstruction parameter\n"
              << "  --help                 Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1m\n"
              << "  " << prog << " --dataset gist1m --M 48 --efConstruction 300\n";
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
        } else if (arg == "--M" && i + 1 < argc) {
            opts.M = std::stoi(argv[++i]);
        } else if (arg == "--efConstruction" && i + 1 < argc) {
            opts.efConstruction = std::stoi(argv[++i]);
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
    if (opts.M > 0) {
        ds_cfg.hnsw_M = opts.M;
    }
    if (opts.efConstruction > 0) {
        ds_cfg.hnsw_efConstruction = opts.efConstruction;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "HNSW Index Builder (Exact Distances)" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "M: " << ds_cfg.hnsw_M << std::endl;
    std::cout << "efConstruction: " << ds_cfg.hnsw_efConstruction << std::endl;
    std::cout << "==================================================" << std::endl;

    // Load data
    std::cout << "\n[Loading data...]" << std::endl;
    size_t d, nb;
    float* xb = fvecs_read(ds_cfg.get_path(ds_cfg.base_file).c_str(), &d, &nb);
    if (!xb) {
        std::cerr << "Error: Failed to load data from "
                  << ds_cfg.get_path(ds_cfg.base_file) << std::endl;
        return 1;
    }
    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;

    // Create index directory
    std::string index_dir = ds_cfg.base_path + "/index";
    create_directory(index_dir);

    // Build HNSW index
    std::cout << "\n[Building HNSW index...]" << std::endl;
    omp_set_num_threads(ds_cfg.threads);
    Timer timer;

    faiss::IndexHNSWFlat index(d, ds_cfg.hnsw_M, faiss::METRIC_L2);
    index.hnsw.efConstruction = ds_cfg.hnsw_efConstruction;

    index.add(nb, xb);

    double build_time = timer.elapsed_ms();
    std::cout << "Build time: " << build_time / 1000.0 << " seconds" << std::endl;

    // Save index
    std::string index_path = ds_cfg.get_hnsw_index_path();

    // Create subdirectory if needed
    size_t last_slash = index_path.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(index_path.substr(0, last_slash));
    }

    std::cout << "\n[Saving index to " << index_path << "...]" << std::endl;
    timer.reset();
    faiss::write_index(&index, index_path.c_str());
    std::cout << "Save time: " << timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "HNSW index built successfully!" << std::endl;
    std::cout << "Index file: " << index_path << std::endl;
    std::cout << "==================================================" << std::endl;

    delete[] xb;
    return 0;
}