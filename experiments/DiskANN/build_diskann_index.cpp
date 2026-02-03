/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Build DiskANN disk index on SIFT1M dataset.
 *
 * Usage:
 *   ./build_diskann_index [data_path] [R] [L]
 *
 * Default:
 *   data_path: /data/local/embedding_dataset/sift1M
 *   R (max degree): 64
 *   L (build complexity): 100
 *
 * Output files stored in:
 *   /data/local/embedding_dataset/sift1M/index/diskann_R{R}_L{L}/
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include "disk_utils.h"
#include "utils.h"

void create_directory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    int ret = system(cmd.c_str());
    (void)ret;  // suppress unused result warning
}

int main(int argc, char* argv[]) {
    std::string data_path = "/data/local/embedding_dataset/sift1M";
    uint32_t R = 64;   // max degree
    uint32_t L = 100;  // build complexity

    if (argc > 1) {
        data_path = argv[1];
    }
    if (argc > 2) {
        R = std::atoi(argv[2]);
    }
    if (argc > 3) {
        L = std::atoi(argv[3]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "DiskANN Disk Index Builder" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "R (max degree): " << R << std::endl;
    std::cout << "L (build complexity): " << L << std::endl;
    std::cout << "==================================================" << std::endl;

    // Paths
    std::string bin_path = data_path + "/sift_base.bin";

    std::string index_dir = data_path + "/index/diskann_R" + std::to_string(R) + "_L" + std::to_string(L);
    std::string index_prefix = index_dir + "/diskann_R" + std::to_string(R) + "_L" + std::to_string(L);

    // Create index directory
    create_directory(index_dir);

    // Build parameters: "R L B M num_threads disk_PQ append_reorder_data build_PQ QD"
    // B = search DRAM budget (GB)
    // M = build DRAM budget (GB)
    float B = 4.0f;   // 4GB search DRAM budget
    float M = 32.0f;  // 32GB build DRAM budget
    uint32_t num_threads = 16;
    uint32_t disk_PQ = 0;           // no disk PQ compression
    uint32_t append_reorder_data = 0;
    uint32_t build_PQ = 0;          // no build-time PQ
    uint32_t QD = 0;                // no quantized dimension

    std::string params = std::to_string(R) + " " + std::to_string(L) + " " +
                         std::to_string(B) + " " + std::to_string(M) + " " +
                         std::to_string(num_threads) + " " + std::to_string(disk_PQ) + " " +
                         std::to_string(append_reorder_data) + " " + std::to_string(build_PQ) + " " +
                         std::to_string(QD);

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
