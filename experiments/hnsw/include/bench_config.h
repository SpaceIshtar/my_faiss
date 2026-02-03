/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace hnsw_bench {

/**
 * Configuration for a dataset.
 */
struct DatasetConfig {
    std::string name;
    std::string base_path;
    std::string base_file = "sift_base.fvecs";
    std::string query_file = "sift_query.fvecs";
    std::string groundtruth_file = "sift_groundtruth.ivecs";

    // HNSW parameters
    int hnsw_M = 32;
    int hnsw_efConstruction = 200;
    std::string hnsw_index_file;  // relative to base_path/index/

    // Search parameters
    std::vector<size_t> ef_search_values;
    int threads = 16;
    size_t k = 10;  // recall@k

    // Get full path to a file
    std::string get_path(const std::string& filename) const {
        return base_path + "/" + filename;
    }

    std::string get_hnsw_index_path() const {
        if (hnsw_index_file.empty()) {
            std::ostringstream oss;
            oss << base_path << "/index/hnsw_M" << hnsw_M
                << "_ef" << hnsw_efConstruction << ".faissindex";
            return oss.str();
        }
        return base_path + "/index/" + hnsw_index_file;
    }
};

/**
 * Configuration for a quantization algorithm parameter set.
 */
struct AlgorithmParamSet {
    std::map<std::string, std::string> params;

    std::string to_string() const {
        std::ostringstream oss;
        bool first = true;
        for (const auto& kv : params) {
            if (!first) oss << "_";
            oss << kv.first << kv.second;
            first = false;
        }
        return oss.str();
    }
};

/**
 * Configuration for a quantization algorithm.
 */
struct AlgorithmConfig {
    std::string name;  // e.g., "pq", "sq"
    std::vector<AlgorithmParamSet> param_sets;
};

/**
 * Simple configuration file parser.
 * Format is INI-like:
 *
 * [section_name]
 * key = value
 * key2 = value2
 *
 * For algorithm configs, each line after section header is a param set:
 * [dataset_name]
 * key1=val1,key2=val2
 * key1=val3,key2=val4
 */
class ConfigParser {
public:
    /**
     * Parse a dataset config file.
     * Returns a map from dataset name to config.
     */
    static std::map<std::string, DatasetConfig> parse_datasets(const std::string& filepath) {
        std::map<std::string, DatasetConfig> result;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file: " << filepath << std::endl;
            return result;
        }

        std::string line;
        std::string current_section;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
                result[current_section].name = current_section;
            } else if (!current_section.empty()) {
                auto pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = trim(line.substr(0, pos));
                    std::string value = trim(line.substr(pos + 1));

                    auto& cfg = result[current_section];
                    if (key == "base_path") cfg.base_path = value;
                    else if (key == "base_file") cfg.base_file = value;
                    else if (key == "query_file") cfg.query_file = value;
                    else if (key == "groundtruth_file") cfg.groundtruth_file = value;
                    else if (key == "hnsw_M") cfg.hnsw_M = std::stoi(value);
                    else if (key == "hnsw_efConstruction") cfg.hnsw_efConstruction = std::stoi(value);
                    else if (key == "hnsw_index_file") cfg.hnsw_index_file = value;
                    else if (key == "threads") cfg.threads = std::stoi(value);
                    else if (key == "k") cfg.k = std::stoul(value);
                    else if (key == "ef_search") {
                        cfg.ef_search_values = parse_size_list(value);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Parse an algorithm config file.
     * Returns a map from dataset name to list of parameter sets.
     */
    static std::map<std::string, std::vector<AlgorithmParamSet>> parse_algorithm(
            const std::string& filepath) {
        std::map<std::string, std::vector<AlgorithmParamSet>> result;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file: " << filepath << std::endl;
            return result;
        }

        std::string line;
        std::string current_section;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.size() - 2);
            } else if (!current_section.empty()) {
                // Parse param set: key1=val1,key2=val2
                AlgorithmParamSet ps;
                std::istringstream iss(line);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    token = trim(token);
                    auto pos = token.find('=');
                    if (pos != std::string::npos) {
                        std::string key = trim(token.substr(0, pos));
                        std::string value = trim(token.substr(pos + 1));
                        ps.params[key] = value;
                    }
                }
                if (!ps.params.empty()) {
                    result[current_section].push_back(ps);
                }
            }
        }

        return result;
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static std::vector<size_t> parse_size_list(const std::string& s) {
        std::vector<size_t> result;
        std::istringstream iss(s);
        std::string token;
        while (std::getline(iss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                result.push_back(std::stoul(token));
            }
        }
        return result;
    }
};

/**
 * Get default ef_search values if not specified.
 */
inline std::vector<size_t> get_default_ef_search() {
    return {10, 20, 30, 40, 50, 75, 100, 150, 200};
}

} // namespace hnsw_bench
