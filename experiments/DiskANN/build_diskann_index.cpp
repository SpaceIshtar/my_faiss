/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Build DiskANN disk index, reading parameters from datasets.conf.
 *
 * Usage:
 *   ./build_diskann_index --dataset <name> [options]
 *
 * Options:
 *   --dataset <name>         Dataset name (e.g., sift1M, gist1M)
 *   --config-dir <path>      Config directory (default: ./config)
 *   --data-path <path>       Override dataset base path
 *   --R <n>                  Override max degree
 *   --L <n>                  Override build complexity
 *   --search-budget <gb>     Override search DRAM budget (GB)
 *   --build-budget <gb>      Override build DRAM budget (GB)
 *   --threads <n>            Override number of threads
 *   --help                   Show this help
 *
 * Output files stored in:
 *   <base_path>/index/diskann_R{R}_L{L}/
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "disk_utils.h"
#include "utils.h"

#include "include/diskann_bench_config.h"

using namespace diskann_bench;

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1M";
    std::string config_dir = "./config";
    std::string data_path;         // override
    int R = -1;                    // -1 = use config
    int L = -1;
    float search_budget = -1.0f;   // -1 = use config
    float build_budget = -1.0f;
    int threads = -1;
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>         Dataset name (e.g., sift1M, gist1M)\n"
              << "  --config-dir <path>      Config directory (default: ./config)\n"
              << "  --data-path <path>       Override dataset base path\n"
              << "  --R <n>                  Override max degree\n"
              << "  --L <n>                  Override build complexity\n"
              << "  --search-budget <gb>     Override search DRAM budget (GB)\n"
              << "  --build-budget <gb>      Override build DRAM budget (GB)\n"
              << "  --threads <n>            Override number of threads\n"
              << "  --help                   Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1M\n"
              << "  " << prog << " --dataset gist1M --R 64 --L 100 --build-budget 64.0\n";
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
        } else if (arg == "--R" && i + 1 < argc) {
            opts.R = std::stoi(argv[++i]);
        } else if (arg == "--L" && i + 1 < argc) {
            opts.L = std::stoi(argv[++i]);
        } else if (arg == "--search-budget" && i + 1 < argc) {
            opts.search_budget = std::stof(argv[++i]);
        } else if (arg == "--build-budget" && i + 1 < argc) {
            opts.build_budget = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        }
    }

    return opts;
}

/*****************************************************
 * Main
 *****************************************************/

void create_directory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    int ret = system(cmd.c_str());
    (void)ret;
}

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
    if (opts.R > 0) {
        ds_cfg.diskann_R = opts.R;
    }
    if (opts.L > 0) {
        ds_cfg.diskann_L_build = opts.L;
    }
    if (opts.search_budget > 0) {
        ds_cfg.search_dram_budget_gb = opts.search_budget;
    }
    if (opts.build_budget > 0) {
        ds_cfg.build_dram_budget_gb = opts.build_budget;
    }
    if (opts.threads > 0) {
        ds_cfg.threads = opts.threads;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "DiskANN Disk Index Builder" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "R (max degree): " << ds_cfg.diskann_R << std::endl;
    std::cout << "L (build complexity): " << ds_cfg.diskann_L_build << std::endl;
    std::cout << "Search DRAM budget: " << ds_cfg.search_dram_budget_gb << " GB" << std::endl;
    std::cout << "Build DRAM budget: " << ds_cfg.build_dram_budget_gb << " GB" << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;
    std::cout << "==================================================" << std::endl;

    // Paths
    std::string bin_path = ds_cfg.get_path(ds_cfg.bin_file);
    std::string index_prefix = ds_cfg.get_diskann_index_prefix();

    // Create index directory
    size_t last_slash = index_prefix.rfind('/');
    if (last_slash != std::string::npos) {
        create_directory(index_prefix.substr(0, last_slash));
    }

    // Build parameters: "R L B M num_threads disk_PQ append_reorder_data build_PQ QD"
    uint32_t disk_PQ = 0;
    uint32_t append_reorder_data = 0;
    uint32_t build_PQ = 0;
    uint32_t QD = 0;

    std::ostringstream params_oss;
    params_oss << ds_cfg.diskann_R << " " << ds_cfg.diskann_L_build << " "
               << ds_cfg.search_dram_budget_gb << " " << ds_cfg.build_dram_budget_gb << " "
               << ds_cfg.threads << " " << disk_PQ << " "
               << append_reorder_data << " " << build_PQ << " " << QD;
    std::string params = params_oss.str();

    std::cout << "\n[Building DiskANN disk index...]" << std::endl;
    std::cout << "Data file: " << bin_path << std::endl;
    std::cout << "Parameters: " << params << std::endl;
    std::cout << "Index prefix: " << index_prefix << std::endl;

    int result = diskann::build_disk_index<float>(
        bin_path.c_str(),
        index_prefix.c_str(),
        params.c_str(),
        diskann::Metric::L2,
        false,  // use_opq
        "",     // codebook_prefix
        false,  // use_filters
        "",     // label_file
        "",     // universal_label
        0,      // filter_threshold
        0       // Lf
    );

    if (result == 0) {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "DiskANN index built successfully!" << std::endl;
        std::cout << "Index files at: " << index_prefix << "*" << std::endl;
        std::cout << "==================================================" << std::endl;
    } else {
        std::cerr << "Error building DiskANN index, code: " << result << std::endl;
        return result;
    }

    return 0;
}