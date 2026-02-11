/*
 * DiskANN benchmark configuration.
 *
 * DiskANN-specific dataset config and parser.
 * Reuses AlgorithmParamSet and ConfigParser::parse_algorithm() from the
 * HNSW framework (algorithm configs are algorithm-agnostic).
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Reuse algorithm config parsing from HNSW framework
#include "../../hnsw/include/bench_config.h"

namespace diskann_bench {

// Re-export from hnsw_bench for convenience
using hnsw_bench::AlgorithmParamSet;
using hnsw_bench::ConfigParser;

struct DiskANNDatasetConfig {
    std::string name;
    std::string base_path;
    std::string base_file = "sift_base.fvecs";
    std::string query_file = "sift_query.fvecs";
    std::string groundtruth_file = "sift_groundtruth.ivecs";

    // DiskANN graph parameters
    int diskann_R = 64;
    int diskann_L_build = 100;
    std::string diskann_index_prefix;  // relative to base_path/index/

    // Build parameters (for build_disk_index)
    std::string bin_file = "sift_base.bin";    // DiskANN binary format file
    float search_dram_budget_gb = 4.0f;        // B: search DRAM budget in GB
    float build_dram_budget_gb = 32.0f;        // M: build DRAM budget in GB

    // Search parameters
    std::vector<uint32_t> L_search_values;
    uint32_t beam_width = 4;
    uint64_t num_nodes_to_cache = 10000;
    int threads = 16;
    size_t k = 10;

    std::string get_path(const std::string& filename) const {
        return base_path + "/" + filename;
    }

    std::string get_diskann_index_prefix() const {
        if (!diskann_index_prefix.empty()) {
            return base_path + "/index/" + diskann_index_prefix;
        }
        std::ostringstream oss;
        oss << base_path << "/index/diskann_R" << diskann_R
            << "_L" << diskann_L_build << "/diskann_R" << diskann_R
            << "_L" << diskann_L_build;
        return oss.str();
    }
};

inline std::string trim_str(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::map<std::string, DiskANNDatasetConfig> parse_diskann_datasets(
        const std::string& filepath) {
    std::map<std::string, DiskANNDatasetConfig> result;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filepath << std::endl;
        return result;
    }

    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        line = trim_str(line);
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            result[current_section].name = current_section;
        } else if (!current_section.empty()) {
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim_str(line.substr(0, pos));
                std::string value = trim_str(line.substr(pos + 1));

                auto& cfg = result[current_section];
                if (key == "base_path") cfg.base_path = value;
                else if (key == "base_file") cfg.base_file = value;
                else if (key == "query_file") cfg.query_file = value;
                else if (key == "groundtruth_file") cfg.groundtruth_file = value;
                else if (key == "diskann_R") cfg.diskann_R = std::stoi(value);
                else if (key == "diskann_L_build") cfg.diskann_L_build = std::stoi(value);
                else if (key == "diskann_index_prefix") cfg.diskann_index_prefix = value;
                else if (key == "bin_file") cfg.bin_file = value;
                else if (key == "search_dram_budget_gb") cfg.search_dram_budget_gb = std::stof(value);
                else if (key == "build_dram_budget_gb") cfg.build_dram_budget_gb = std::stof(value);
                else if (key == "threads") cfg.threads = std::stoi(value);
                else if (key == "k") cfg.k = std::stoul(value);
                else if (key == "beam_width") cfg.beam_width = std::stoul(value);
                else if (key == "num_nodes_to_cache") cfg.num_nodes_to_cache = std::stoull(value);
                else if (key == "L_search") {
                    std::istringstream iss(value);
                    std::string token;
                    while (std::getline(iss, token, ',')) {
                        token = trim_str(token);
                        if (!token.empty())
                            cfg.L_search_values.push_back(std::stoul(token));
                    }
                }
            }
        }
    }
    return result;
}

inline std::vector<uint32_t> get_default_L_search() {
    return {10, 20, 30, 50, 70, 100, 150, 200};
}

} // namespace diskann_bench
