#include "OSQIndex.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string trim(const std::string& s) {
  const size_t l = s.find_first_not_of(" \t\r\n");
  if (l == std::string::npos) return "";
  const size_t r = s.find_last_not_of(" \t\r\n");
  return s.substr(l, r - l + 1);
}

struct DatasetConfig {
  std::string base_path;
  std::string base_file;
  std::string query_file;
  std::string groundtruth_file;
  int threads = 1;
  int k = 10;
};

std::unordered_map<std::string, DatasetConfig> load_dataset_configs(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open config file: " + path);

  std::unordered_map<std::string, DatasetConfig> out;
  std::string line;
  std::string current;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line.front() == '[' && line.back() == ']') {
      current = line.substr(1, line.size() - 2);
      out[current] = DatasetConfig{};
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos || current.empty()) continue;
    const std::string key = trim(line.substr(0, eq));
    const std::string val = trim(line.substr(eq + 1));
    DatasetConfig& cfg = out[current];
    if (key == "base_path") cfg.base_path = val;
    if (key == "base_file") cfg.base_file = val;
    if (key == "query_file") cfg.query_file = val;
    if (key == "groundtruth_file") cfg.groundtruth_file = val;
    if (key == "threads") cfg.threads = std::stoi(val);
    if (key == "k") cfg.k = std::stoi(val);
  }
  return out;
}

std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::vector<float> load_fvecs(const std::string& path, size_t& n, size_t& d) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open fvecs file: " + path);
  n = 0;
  d = 0;
  std::vector<float> out;
  while (true) {
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in) break;
    if (dim <= 0) throw std::runtime_error("invalid fvecs dim in " + path);
    if (d == 0) d = static_cast<size_t>(dim);
    if (d != static_cast<size_t>(dim)) {
      throw std::runtime_error("inconsistent fvecs dimensions in " + path);
    }
    const size_t old = out.size();
    out.resize(old + d);
    in.read(reinterpret_cast<char*>(out.data() + old), static_cast<std::streamsize>(d * sizeof(float)));
    if (!in) throw std::runtime_error("truncated fvecs file: " + path);
    ++n;
  }
  return out;
}

std::vector<int32_t> load_ivecs(const std::string& path, size_t& n, size_t& d) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open ivecs file: " + path);
  n = 0;
  d = 0;
  std::vector<int32_t> out;
  while (true) {
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in) break;
    if (dim <= 0) throw std::runtime_error("invalid ivecs dim in " + path);
    if (d == 0) d = static_cast<size_t>(dim);
    if (d != static_cast<size_t>(dim)) {
      throw std::runtime_error("inconsistent ivecs dimensions in " + path);
    }
    const size_t old = out.size();
    out.resize(old + d);
    in.read(reinterpret_cast<char*>(out.data() + old), static_cast<std::streamsize>(d * sizeof(int32_t)));
    if (!in) throw std::runtime_error("truncated ivecs file: " + path);
    ++n;
  }
  return out;
}

float recall_at_k(
    const std::vector<osq::idx_t>& labels,
    size_t nq,
    size_t k,
    const std::vector<int32_t>& gt,
    size_t gt_k) {
  if (gt_k == 0) return 0.0f;
  size_t hit = 0;
  for (size_t i = 0; i < nq; ++i) {
    const int32_t* g = gt.data() + i * gt_k;
    for (size_t j = 0; j < k; ++j) {
      const osq::idx_t id = labels[i * k + j];
      for (size_t t = 0; t < std::min(k, gt_k); ++t) {
        if (id == static_cast<osq::idx_t>(g[t])) {
          ++hit;
          break;
        }
      }
    }
  }
  return static_cast<float>(hit) / static_cast<float>(nq * k);
}

osq::Similarity parse_similarity(const std::string& s) {
  if (s == "euclidean" || s == "l2") return osq::Similarity::EUCLIDEAN;
  if (s == "cosine") return osq::Similarity::COSINE;
  if (s == "dot" || s == "dot_product") return osq::Similarity::DOT_PRODUCT;
  if (s == "mip" || s == "max_inner_product") return osq::Similarity::MAX_INNER_PRODUCT;
  throw std::runtime_error("unknown similarity: " + s);
}

osq::ScalarEncoding parse_encoding(const std::string& s) {
  if (s == "uint8") return osq::ScalarEncoding::UNSIGNED_BYTE;
  if (s == "int4") return osq::ScalarEncoding::PACKED_NIBBLE;
  if (s == "int7") return osq::ScalarEncoding::SEVEN_BIT;
  if (s == "binary") return osq::ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE;
  if (s == "dibit") return osq::ScalarEncoding::DIBIT_QUERY_NIBBLE;
  throw std::runtime_error("unknown encoding: " + s);
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " --dataset <name> [--config datasets.conf] "
            << "[--similarity euclidean|cosine|dot|mip] [--encoding uint8|int7|int4|binary|dibit] "
            << "[--threads T] [--k K]\n";
}

} // namespace

int main(int argc, char** argv) {
  std::string dataset;
  std::string config_path = "datasets.conf";
  std::string similarity = "euclidean";
  std::string encoding = "dibit";
  int override_threads = -1;
  int override_k = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_val = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (arg == "--dataset") dataset = require_val(arg);
    else if (arg == "--config") config_path = require_val(arg);
    else if (arg == "--similarity") similarity = require_val(arg);
    else if (arg == "--encoding") encoding = require_val(arg);
    else if (arg == "--threads") override_threads = std::stoi(require_val(arg));
    else if (arg == "--k") override_k = std::stoi(require_val(arg));
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (dataset.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  const auto cfgs = load_dataset_configs(config_path);
  auto it = cfgs.find(dataset);
  if (it == cfgs.end()) {
    throw std::runtime_error("dataset not found in config: " + dataset);
  }
  const DatasetConfig& cfg = it->second;
  const int threads = (override_threads > 0) ? override_threads : cfg.threads;
  const int k = (override_k > 0) ? override_k : cfg.k;

  const std::string base_path = join_path(cfg.base_path, cfg.base_file);
  const std::string query_path = join_path(cfg.base_path, cfg.query_file);
  const std::string gt_path = join_path(cfg.base_path, cfg.groundtruth_file);

  size_t nb = 0, dbase = 0;
  size_t nq = 0, dquery = 0;
  size_t ngt = 0, dgt = 0;
  std::vector<float> xb = load_fvecs(base_path, nb, dbase);
  std::vector<float> xq = load_fvecs(query_path, nq, dquery);
  std::vector<int32_t> gt = load_ivecs(gt_path, ngt, dgt);

  if (dbase != dquery) throw std::runtime_error("base/query dims mismatch");
  if (ngt != nq) throw std::runtime_error("query/groundtruth size mismatch");

  std::cout << "dataset=" << dataset << " nb=" << nb << " nq=" << nq << " d=" << dbase
            << " k=" << k << " threads=" << threads
            << " similarity=" << similarity << " encoding=" << encoding << "\n";

  osq::OSQIndex index(dbase, parse_similarity(similarity), parse_encoding(encoding));
  index.set_num_threads(threads);

  const auto t0 = std::chrono::steady_clock::now();
  index.train(nb, xb.data());
  index.add(nb, xb.data());
  const auto t1 = std::chrono::steady_clock::now();

  std::vector<float> distances(nq * static_cast<size_t>(k));
  std::vector<osq::idx_t> labels(nq * static_cast<size_t>(k));
  const auto t2 = std::chrono::steady_clock::now();
  index.search(nq, xq.data(), static_cast<size_t>(k), distances.data(), labels.data());
  const auto t3 = std::chrono::steady_clock::now();

  const double build_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
  const double search_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t3 - t2).count();
  const double qps = (search_ms > 0.0) ? (1000.0 * static_cast<double>(nq) / search_ms) : 0.0;
  const float recall = recall_at_k(labels, nq, static_cast<size_t>(k), gt, dgt);

  std::cout << "build_ms=" << build_ms << "\n";
  std::cout << "search_ms=" << search_ms << "\n";
  std::cout << "qps=" << qps << "\n";
  std::cout << "recall@" << k << "=" << recall << "\n";

  return 0;
}
