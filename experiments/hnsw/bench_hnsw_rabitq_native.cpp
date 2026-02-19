/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * HNSW + RaBitQ Native Search Benchmark
 *
 * This benchmark uses FAISS HNSW graph structure but implements the authentic
 * RaBitQ search logic with low_dist optimization:
 * - First compute 1-bit distance estimate and low_dist bound
 * - Only compute full estimate (with extra bits) if low_dist < current bound
 *
 * Usage:
 *   ./bench_hnsw_rabitq_native --dataset <name> [options]
 *
 * Options:
 *   --dataset <name>       Dataset name (e.g., sift1m)
 *   --bits <n>             Total bits for RaBitQ (1-9, default: 4)
 *   --rerank               Enable exact distance reranking
 *   --data-path <path>     Override dataset base path
 *   --threads <n>          Number of threads
 *   --ef <list>            Comma-separated ef values to test
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

// RaBitQ library
#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

/*****************************************************
 * RaBitQ Data Storage
 *****************************************************/

class RaBitQIndex {
public:
    using IpFunc = float (*)(const float*, const uint8_t*, size_t);

    RaBitQIndex(size_t d, size_t total_bits, size_t num_clusters = 16)
        : d_(d), total_bits_(total_bits), ex_bits_(total_bits - 1),
          num_clusters_(num_clusters), ntotal_(0), is_trained_(false) {

        // Create rotator
        // IMPORTANT: Set deterministic seed so rotator is reproducible across save/load
        padded_dim_ = rabitqlib::round_up_to_multiple(d_, 64);
        std::srand(d_ + padded_dim_);  // Deterministic seed based on dimensions
        rotator_.reset(rabitqlib::choose_rotator<float>(
            d_, rabitqlib::RotatorType::FhtKacRotator, padded_dim_));

        // Calculate storage sizes
        bin_data_size_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
        ex_data_size_ = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        code_size_ = bin_data_size_ + ex_data_size_;

        // Initialize config
        config_ = rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);

        // Select IP function for extra bits
        ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);

        // Allocate rotated centroids storage
        rotated_centroids_.resize(num_clusters_ * padded_dim_);
    }

    void train(size_t n, const float* x) {
        // Use FAISS Clustering for KMeans
        faiss::ClusteringParameters cp;
        cp.niter = 25;
        cp.seed = 1234;
        cp.verbose = false;

        faiss::Clustering clus(d_, num_clusters_, cp);
        faiss::IndexFlatL2 index(d_);

        // Run clustering
        clus.train(n, x, index);

        // Store and rotate centroids
        centroids_.resize(num_clusters_ * d_);
        std::memcpy(centroids_.data(), clus.centroids.data(), num_clusters_ * d_ * sizeof(float));

        for (size_t i = 0; i < num_clusters_; i++) {
            rotator_->rotate(
                centroids_.data() + i * d_,
                rotated_centroids_.data() + i * padded_dim_
            );
        }

        // Build quantizer for cluster assignment
        quantizer_.reset(new faiss::IndexFlatL2(d_));
        quantizer_->add(num_clusters_, centroids_.data());

        is_trained_ = true;
    }

    void add(size_t n, const float* x) {
        if (!is_trained_) {
            train(n, x);
        }

        size_t old_ntotal = ntotal_;
        ntotal_ += n;
        codes_.resize(ntotal_ * code_size_);
        cluster_ids_.resize(ntotal_);

        // Assign vectors to clusters
        std::vector<float> dists(n);
        std::vector<faiss::idx_t> labels(n);
        quantizer_->search(n, x, 1, dists.data(), labels.data());

        std::vector<float> rotated(padded_dim_);

        for (size_t i = 0; i < n; i++) {
            size_t cid = labels[i];
            cluster_ids_[old_ntotal + i] = static_cast<uint32_t>(cid);

            rotator_->rotate(x + i * d_, rotated.data());

            char* vec_code = codes_.data() + (old_ntotal + i) * code_size_;
            char* bin_data = vec_code;
            char* ex_data = vec_code + bin_data_size_;

            // Quantize relative to cluster centroid
            rabitqlib::quant::quantize_split_single(
                rotated.data(),
                rotated_centroids_.data() + cid * padded_dim_,
                padded_dim_, ex_bits_,
                bin_data, ex_data, rabitqlib::METRIC_L2, config_);
        }
    }

    // Accessors
    size_t get_ntotal() const { return ntotal_; }
    size_t padded_dim() const { return padded_dim_; }
    size_t ex_bits() const { return ex_bits_; }
    size_t code_size() const { return code_size_; }
    size_t bin_data_size() const { return bin_data_size_; }
    size_t num_clusters() const { return num_clusters_; }
    const char* get_code(size_t i) const { return codes_.data() + i * code_size_; }
    uint32_t get_cluster_id(size_t i) const { return cluster_ids_[i]; }
    const float* get_rotated_centroids() const { return rotated_centroids_.data(); }
    rabitqlib::Rotator<float>* rotator() const { return rotator_.get(); }
    const rabitqlib::quant::RabitqConfig& config() const { return config_; }
    IpFunc ip_func() const { return ip_func_; }

    /*****************************************************
     * Save / Load (same binary format as RaBitQWrapper)
     *****************************************************/

    bool save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;

        uint64_t magic = 0x524142495451ULL;
        uint64_t version = 1;
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

        auto write_u64 = [&](uint64_t v) {
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        write_u64(d_);
        write_u64(padded_dim_);
        write_u64(total_bits_);
        write_u64(ex_bits_);
        write_u64(num_clusters_);
        write_u64(ntotal_);
        write_u64(bin_data_size_);
        write_u64(ex_data_size_);
        write_u64(code_size_);

        ofs.write(codes_.data(), ntotal_ * code_size_);
        ofs.write(reinterpret_cast<const char*>(cluster_ids_.data()),
                  ntotal_ * sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(centroids_.data()),
                  num_clusters_ * d_ * sizeof(float));
        ofs.write(reinterpret_cast<const char*>(rotated_centroids_.data()),
                  num_clusters_ * padded_dim_ * sizeof(float));

        ofs.close();
        return ofs.good();
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return false;

        uint64_t magic, version;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != 0x524142495451ULL || version != 1) return false;

        auto read_u64 = [&]() -> uint64_t {
            uint64_t v;
            ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };

        uint64_t d = read_u64();
        uint64_t padded_dim = read_u64();
        uint64_t total_bits = read_u64();
        uint64_t ex_bits = read_u64();
        uint64_t num_clusters = read_u64();
        uint64_t ntotal = read_u64();
        uint64_t bin_data_size = read_u64();
        uint64_t ex_data_size = read_u64();
        uint64_t code_size = read_u64();

        if (d != d_ || padded_dim != padded_dim_ ||
            total_bits != total_bits_ || ex_bits != ex_bits_ ||
            num_clusters != num_clusters_ ||
            bin_data_size != bin_data_size_ ||
            ex_data_size != ex_data_size_ ||
            code_size != code_size_) {
            std::cerr << "Error: RaBitQ index parameters mismatch" << std::endl;
            return false;
        }

        ntotal_ = ntotal;
        is_trained_ = true;

        codes_.resize(ntotal_ * code_size_);
        ifs.read(codes_.data(), ntotal_ * code_size_);

        cluster_ids_.resize(ntotal_);
        ifs.read(reinterpret_cast<char*>(cluster_ids_.data()),
                 ntotal_ * sizeof(uint32_t));

        centroids_.resize(num_clusters_ * d_);
        ifs.read(reinterpret_cast<char*>(centroids_.data()),
                 num_clusters_ * d_ * sizeof(float));

        rotated_centroids_.resize(num_clusters_ * padded_dim_);
        ifs.read(reinterpret_cast<char*>(rotated_centroids_.data()),
                 num_clusters_ * padded_dim_ * sizeof(float));

        quantizer_.reset(new faiss::IndexFlatL2(d_));
        quantizer_->add(num_clusters_, centroids_.data());

        if (!ifs.good()) {
            std::cerr << "Error: failed reading RaBitQ index" << std::endl;
            return false;
        }
        return true;
    }

    static std::string get_index_path(
            const std::string& base_path,
            size_t total_bits,
            size_t num_clusters) {
        return base_path + "/index/rabitq_bits" +
               std::to_string(total_bits) + "_c" +
               std::to_string(num_clusters) + ".bin";
    }

private:
    size_t d_;
    size_t padded_dim_;
    size_t total_bits_;
    size_t ex_bits_;
    size_t num_clusters_;
    size_t ntotal_;
    bool is_trained_;
    size_t bin_data_size_;
    size_t ex_data_size_;
    size_t code_size_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    rabitqlib::quant::RabitqConfig config_;
    std::vector<char> codes_;
    std::vector<uint32_t> cluster_ids_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    IpFunc ip_func_;
};

/*****************************************************
 * Native RaBitQ Search (replicates original logic)
 *****************************************************/

struct SearchResult {
    float est_dist;
    float low_dist;
    faiss::idx_t id;

    bool operator<(const SearchResult& other) const {
        return est_dist < other.est_dist;
    }
    bool operator>(const SearchResult& other) const {
        return est_dist > other.est_dist;
    }
};

struct QueryStats1{
    size_t nhops = 0;
    size_t ndis1 = 0;
    size_t ndis2 = 0;
};

using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<SearchResult>>;
using MaxHeap = std::priority_queue<SearchResult>;

QueryStats1 rabitq_native_search(
        const faiss::HNSW& hnsw,
        const faiss::IndexHNSW* hnsw_index,
        faiss::VisitedTable& vt,
        const RaBitQIndex* rabitq,
        const float* query,
        size_t k,
        size_t ef,
        float* distances,
        faiss::idx_t* labels,
        bool do_rerank,
        faiss::DistanceComputer* exact_dc) {

    QueryStats1 stats;
    size_t padded_dim = rabitq->padded_dim();
    size_t ex_bits = rabitq->ex_bits();
    size_t num_clusters = rabitq->num_clusters();

    // Rotate query
    std::vector<float> rotated_query(padded_dim);
    rabitq->rotator()->rotate(query, rotated_query.data());

    // Precompute query-to-centroid distances (like original RaBitQ library)
    std::vector<float> q_to_centroids(num_clusters);
    const float* rotated_centroids = rabitq->get_rotated_centroids();
    for (size_t i = 0; i < num_clusters; i++) {
        float dist_sq = rabitqlib::euclidean_sqr(
            rotated_query.data(),
            rotated_centroids + i * padded_dim,
            padded_dim
        );
        q_to_centroids[i] = std::sqrt(dist_sq);  // store norm
    }

    // Create query wrapper
    rabitqlib::SplitSingleQuery<float> query_wrapper(
        rotated_query.data(), padded_dim, ex_bits,
        rabitq->config(), rabitqlib::METRIC_L2);

    // Entry point
    faiss::HNSW::storage_idx_t ep = hnsw.entry_point;
    if (ep < 0) {
        for (size_t i = 0; i < k; i++) {
            distances[i] = std::numeric_limits<float>::max();
            labels[i] = -1;
        }
        return stats;
    }

    // Get 1-bit estimate with proper g_add/g_error
    auto get_estimate = [&](faiss::idx_t id, float& est_dist, float& low_dist, float& ip_x0_qr) {
        const char* code = rabitq->get_code(id);
        const char* bin_data = code;
        uint32_t cid = rabitq->get_cluster_id(id);
        float norm = q_to_centroids[cid];
        float g_add = norm * norm;
        float g_error = norm;

        rabitqlib::split_single_estdist(
            bin_data, query_wrapper, padded_dim,
            ip_x0_qr, est_dist, low_dist, g_add, g_error);
        stats.ndis1++;
    };

    // Get full estimate with extra bits
    auto get_full_estimate = [&](faiss::idx_t id, float& ip_x0_qr, float& est_dist, float& low_dist) {
        const char* code = rabitq->get_code(id);
        const char* bin_data = code;
        const char* ex_data = code + rabitq->bin_data_size();
        uint32_t cid = rabitq->get_cluster_id(id);
        float norm = q_to_centroids[cid];
        float g_add = norm * norm;
        float g_error = norm;

        rabitqlib::split_single_fulldist(
            bin_data, ex_data, rabitq->ip_func(), query_wrapper, padded_dim,
            ex_bits, est_dist, low_dist, ip_x0_qr, g_add, g_error);
        stats.ndis2++;
    };

    float ep_est, ep_low, ep_ip;
    get_estimate(ep, ep_est, ep_low, ep_ip);
    // std::cout << "ep = " << ep << ", ep_est = " << ep_est << ", ep_low = " << ep_low << std::endl;
    if (ex_bits > 0) {
        get_full_estimate(ep, ep_ip, ep_est, ep_low);
    }
    // std::cout << "ep = " << ep << ", ep_est = " << ep_est << ", ep_low = " << ep_low << std::endl;

    // Search upper levels (greedy)
    for (int level = hnsw.max_level; level > 0; level--) {
        // std::cout << level << std::endl;
        bool changed = true;
        while (changed) {
            changed = false;
            size_t begin, end;
            hnsw.neighbor_range(ep, level, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                faiss::HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
                if (neighbor < 0) break;

                float n_est, n_low, n_ip;
                get_estimate(neighbor, n_est, n_low, n_ip);
                // std::cout << "neighbor = " << neighbor << ", est = " << n_est << ", low = " << n_low << std::endl;
                if (n_est < ep_est) {
                    // if (ex_bits > 0) {
                    //     get_full_estimate(neighbor, n_ip, n_est, n_low);
                    // }
                    if (n_est < ep_est) {
                        ep = neighbor;
                        ep_est = n_est;
                        ep_low = n_low;
                        ep_ip = n_ip;
                        changed = true;
                    }
                }
            }
            stats.nhops++;
        }
    }

    // Search level 0 with bounded queue (native RaBitQ logic)
    MinHeap candidates;
    MaxHeap topk_heap;

    candidates.push({ep_est, ep_low, ep});
    topk_heap.push({ep_est, ep_low, ep});
    vt.set(ep);

    float dist_bound = ep_est;

    while (!candidates.empty()) {
        SearchResult curr = candidates.top();
        candidates.pop();

        // Pruning: if current candidate is worse than bound, stop
        if (curr.est_dist > dist_bound && topk_heap.size() >= ef) {
            break;
        }

        // Explore neighbors
        size_t begin, end;
        hnsw.neighbor_range(curr.id, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            faiss::HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor < 0) break;
            if (vt.get(neighbor)) continue;
            vt.set(neighbor);

            stats.nhops++;

            // First, get 1-bit estimate
            float n_est, n_low, n_ip;
            get_estimate(neighbor, n_est, n_low, n_ip);

            // Key optimization: only compute full estimate if low_dist < bound
            bool should_add_to_topk = (topk_heap.size() < ef) || (n_low < dist_bound);

            if (should_add_to_topk && ex_bits > 0) {
                // Compute full estimate only when promising
                get_full_estimate(neighbor, n_ip, n_est, n_low);
            }

            // Add to candidates if promising
            if (candidates.size() < ef || n_est < dist_bound) {
                candidates.push({n_est, n_low, neighbor});
            }

            // Update top-k
            if (should_add_to_topk) {
                topk_heap.push({n_est, n_low, neighbor});
                if (topk_heap.size() > ef) {
                    topk_heap.pop();
                }
                if (topk_heap.size() >= ef) {
                    dist_bound = topk_heap.top().est_dist;
                }
            }
        }
    }

    vt.advance();

    // Collect results
    std::vector<SearchResult> results;
    while (!topk_heap.empty()) {
        results.push_back(topk_heap.top());
        topk_heap.pop();
    }
    std::reverse(results.begin(), results.end());

    // Optional rerank with exact distances
    if (do_rerank && exact_dc) {
        exact_dc->set_query(query);
        std::vector<std::pair<float, faiss::idx_t>> reranked;
        for (const auto& r : results) {
            float exact_dist = (*exact_dc)(r.id);
            reranked.push_back({exact_dist, r.id});
        }
        std::sort(reranked.begin(), reranked.end());

        for (size_t i = 0; i < k && i < reranked.size(); i++) {
            distances[i] = reranked[i].first;
            labels[i] = reranked[i].second;
        }
    } else {
        for (size_t i = 0; i < k && i < results.size(); i++) {
            distances[i] = results[i].est_dist;
            labels[i] = results[i].id;
        }
    }

    // Fill remaining
    for (size_t i = results.size(); i < k; i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }

    return stats;
}

/*****************************************************
 * Benchmark runner
 *****************************************************/

struct SearchStats1 {
    PercentileStats ndis1_stats;
    PercentileStats ndis2_stats;
    PercentileStats nhops_stats;

    static SearchStats1 compute(const std::vector<QueryStats1>& query_stats) {
        SearchStats1 result;
        if (query_stats.empty()) return result;

        std::vector<size_t> ndis1_values, ndis2_values, nhops_values;
        ndis1_values.reserve(query_stats.size());
        ndis2_values.reserve(query_stats.size());
        nhops_values.reserve(query_stats.size());

        for (const auto& qs : query_stats) {
            ndis1_values.push_back(qs.ndis1);
            ndis2_values.push_back(qs.ndis2);
            nhops_values.push_back(qs.nhops);
        }

        result.ndis1_stats = compute_percentile_stats(ndis1_values);
        result.ndis2_stats = compute_percentile_stats(ndis2_values);
        result.nhops_stats = compute_percentile_stats(nhops_values);

        return result;
    }
};

struct BenchmarkResult {
    size_t ef;
    double qps;
    float recall;
    double latency_ms;
    SearchStats1 stats;
};

std::vector<BenchmarkResult> run_benchmark(
        const faiss::IndexHNSW* hnsw_index,
        const faiss::IndexFlat* flat_index,
        const RaBitQIndex* rabitq,
        const float* xq,
        size_t nq,
        size_t d,
        const int* gt,
        size_t gt_k,
        size_t k,
        const std::vector<size_t>& ef_values,
        int num_threads,
        bool do_rerank) {

    std::vector<BenchmarkResult> results;
    Timer timer;
    const faiss::HNSW& hnsw = hnsw_index->hnsw;

    for (size_t ef : ef_values) {
        std::vector<faiss::idx_t> result_ids(nq * k, -1);
        std::vector<float> result_dists(nq * k, std::numeric_limits<float>::max());
        std::vector<QueryStats1> query_stats(nq);

        // Timed search
        timer.reset();

        #pragma omp parallel num_threads(num_threads)
        {
            std::unique_ptr<faiss::DistanceComputer> exact_dc;
            if (do_rerank) {
                exact_dc.reset(flat_index->get_distance_computer());
            }
            faiss::VisitedTable vt(hnsw_index->ntotal);

            #pragma omp for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)nq; i++) {
                query_stats[i] = rabitq_native_search(
                    hnsw, hnsw_index, vt, rabitq,
                    xq + i * d, k, ef,
                    result_dists.data() + i * k,
                    result_ids.data() + i * k,
                    do_rerank, exact_dc.get());
            }
        }

        double search_time = timer.elapsed_ms();
        double qps = nq * 1000.0 / search_time;
        double latency = search_time / nq;
        float recall = compute_recall_at_k(nq, k, result_ids.data(), gt, gt_k);
        SearchStats1 stats = SearchStats1::compute(query_stats);

        results.push_back({ef, qps, recall, latency, stats});

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << std::setw(12) << std::setprecision(4) << recall
                  << std::setw(12) << std::setprecision(3) << latency
                  << std::setw(12) << std::setprecision(1) << stats.ndis1_stats.mean
                  << std::setw(12) << std::setprecision(1) << stats.ndis2_stats.mean
                  << std::setw(12) << std::setprecision(1) << stats.nhops_stats.mean
                  << std::endl;
    }

    return results;
}

/*****************************************************
 * Command line parsing
 *****************************************************/

struct Options {
    std::string dataset = "sift1m";
    std::string config_dir = "./config";
    std::string data_path;
    size_t total_bits = 4;
    int threads = 16;
    std::vector<size_t> ef_values;
    bool rerank = false;
    bool help = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --dataset <name> [options]\n\n"
              << "Options:\n"
              << "  --dataset <name>       Dataset name (e.g., sift1m)\n"
              << "  --bits <n>             Total bits for RaBitQ (1-9, default: 4)\n"
              << "  --rerank               Enable exact distance reranking\n"
              << "  --config-dir <path>    Config directory (default: ./config)\n"
              << "  --data-path <path>     Override dataset base path\n"
              << "  --threads <n>          Number of threads\n"
              << "  --ef <list>            Comma-separated ef values to test\n"
              << "  --help                 Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " --dataset sift1m --bits 4\n"
              << "  " << prog << " --dataset sift1m --bits 4 --rerank\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--dataset" && i + 1 < argc) {
            opts.dataset = argv[++i];
        } else if (arg == "--bits" && i + 1 < argc) {
            opts.total_bits = std::stoul(argv[++i]);
        } else if (arg == "--rerank") {
            opts.rerank = true;
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
        const std::string& quant_params,
        int hnsw_M,
        int hnsw_efConstruction,
        size_t nq,
        size_t k,
        const std::vector<BenchmarkResult>& results) {


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
        << "  Algorithm:         " << "RaBitQ-Native" << "\n"
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
        << std::setw(12) << "ndis1_mean"
        << std::setw(12) << "ndis2_mean"
        << std::setw(12) << "nhops_mean"
        << "\n"
        << std::string(68, '-') << "\n";

    for (const auto& r : results) {
        ofs << std::setw(8) << r.ef
            << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
            << std::setw(12) << std::setprecision(4) << r.recall
            << std::setw(12) << std::setprecision(3) << r.latency_ms
            << std::setw(12) << std::setprecision(1) << r.stats.ndis1_stats.mean
            << std::setw(12) << std::setprecision(1) << r.stats.ndis2_stats.mean
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
            << "  Total Time:   " << std::setprecision(2) << -1 << " ms\n"
            << "  Avg Latency:  " << std::setprecision(4) << r.latency_ms << " ms\n\n";

        print_percentile_stats(ofs, "ndis (distance computations)", r.stats.ndis1_stats);
        ofs << "\n";
        print_percentile_stats(ofs, "ndis (distance computations)", r.stats.ndis2_stats);
        ofs << "\n";
        print_percentile_stats(ofs, "nhops (graph edges traversed)", r.stats.nhops_stats);
    }

    ofs << "\n========================================\n"
        << "End of Results\n"
        << "========================================\n";

    ofs.close();
    std::cout << "Results saved to: " << filepath << std::endl;
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
    std::cout << "HNSW + RaBitQ Native Search Benchmark" << std::endl;
    std::cout << "Dataset: " << opts.dataset << std::endl;
    std::cout << "Total bits: " << opts.total_bits << std::endl;
    std::cout << "Rerank: " << (opts.rerank ? "yes" : "no") << std::endl;
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

    std::string algo_config = opts.config_dir + "/rabitq.conf";
    auto algo_params = ConfigParser::parse_algorithm(algo_config);

    std::vector<AlgorithmParamSet> param_sets;
    auto ait = algo_params.find(opts.dataset);
    if (ait != algo_params.end()) {
        param_sets = ait->second;
    }
    else{
        std::cerr << "cannot find corresponding dataset " << std::endl;
        exit(-1);
    }


    omp_set_num_threads(ds_cfg.threads);
    std::cout << "Data path: " << ds_cfg.base_path << std::endl;
    std::cout << "Threads: " << ds_cfg.threads << std::endl;

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

    // Load HNSW index
    std::string hnsw_path = ds_cfg.get_hnsw_index_path();
    faiss::IndexHNSW* hnsw_index = load_or_build_hnsw(
        hnsw_path, d, ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction, nb, xb);

    if (!hnsw_index) {
        std::cerr << "Error: Failed to load/build HNSW index" << std::endl;
        return 1;
    }

    faiss::IndexFlat* flat_index = dynamic_cast<faiss::IndexFlat*>(hnsw_index->storage);
    if (!flat_index) {
        std::cerr << "Error: HNSW storage is not IndexFlat" << std::endl;
        return 1;
    }

    for (const auto& parameters: param_sets){
        size_t total_bits = 4;  // default: 1-bit + 3 extra bits
        size_t num_clusters = 16;  // default as in RaBitQ paper
        
        // Also accept "bits" as parameter name
        auto it = parameters.params.find("bits");
        if (it != parameters.params.end()) {
            total_bits = std::stoul(it->second);
        }
        else{
            std::cerr << "bits needed to be set in conf" << std::endl;
            exit(-1);
        }

        it = parameters.params.find("clusters");
        if (it != parameters.params.end()) {
            num_clusters = std::stoul(it->second);
        }
        else{
            std::cerr << "clusters needed to be set in conf" << std::endl;
            exit(-1);
        }
        // Build or load RaBitQ index
        RaBitQIndex rabitq(d, total_bits, num_clusters);
        std::string rabitq_path = RaBitQIndex::get_index_path(
            ds_cfg.base_path, total_bits, num_clusters);

        if (rabitq.load(rabitq_path)) {
            std::cout << "\n[Loaded RaBitQ index from " << rabitq_path << "]" << std::endl;
            std::cout << "  ntotal=" << rabitq.get_ntotal() << std::endl;
        } else {
            std::cout << "\n[Building RaBitQ index (bits=" << total_bits
                      << ", ncluster=" << num_clusters << ")...]" << std::endl;
            Timer timer;
            rabitq.add(nb, xb);
            std::cout << "RaBitQ build time: " << timer.elapsed_ms() << " ms" << std::endl;

            // Save for future runs
            size_t last_slash = rabitq_path.rfind('/');
            if (last_slash != std::string::npos) {
                create_directory(rabitq_path.substr(0, last_slash));
            }
            if (rabitq.save(rabitq_path)) {
                std::cout << "Saved RaBitQ index to: " << rabitq_path << std::endl;
            }
        }

        // Run benchmark
        std::cout << "\n" << std::setw(8) << "ef"
                << std::setw(12) << "QPS"
                << std::setw(12) << "Recall@" << ds_cfg.k
                << std::setw(12) << "Latency(ms)"
                << std::setw(12) << "ndis1_mean"
                << std::setw(12) << "ndis2_mean"
                << std::setw(12) << "nhops_mean" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        auto results = run_benchmark(
            hnsw_index, flat_index, &rabitq,
            xq, nq, d, gt, gt_k, ds_cfg.k,
            ds_cfg.ef_search_values, ds_cfg.threads, opts.rerank);

        // Save results
        std::string result_dir = "experiments/hnsw/results/" + opts.dataset + "/rabitq_native";
        create_directory(result_dir);
        std::string suffix = opts.rerank ? "_rerank" : "";
        std::string result_file = result_dir + "/rabitq_native_bits" + std::to_string(total_bits) +
            "_c" + std::to_string(num_clusters) + suffix +
            "_M" + std::to_string(ds_cfg.hnsw_M) +
            "_efc" + std::to_string(ds_cfg.hnsw_efConstruction) + ".txt";

        save_results_to_file(
            result_file, opts.dataset, parameters.to_string(),
            ds_cfg.hnsw_M, ds_cfg.hnsw_efConstruction,
            nq, ds_cfg.k, results);

        // std::ofstream ofs(result_file);
        // if (ofs.is_open()) {
        //     ofs << "# RaBitQ Native Search Benchmark\n"
        //         << "# Dataset: " << opts.dataset << "\n"
        //         << "# Total bits: " << opts.total_bits << "\n"
        //         << "# Rerank: " << (opts.rerank ? "yes" : "no") << "\n"
        //         << "# HNSW M: " << ds_cfg.hnsw_M << ", efConstruction: " << ds_cfg.hnsw_efConstruction << "\n\n"
        //         << std::setw(8) << "ef"
        //         << std::setw(12) << "QPS"
        //         << std::setw(12) << "Recall"
        //         << std::setw(12) << "Latency(ms)"
        //         << std::setw(12) << "ndis1_mean"
        //         << std::setw(12) << "ndis2_mean"
        //         << std::setw(12) << "nhops_mean" << "\n";

        //     for (const auto& r : results) {
        //         ofs << std::setw(8) << r.ef
        //             << std::setw(12) << std::fixed << std::setprecision(0) << r.qps
        //             << std::setw(12) << std::setprecision(4) << r.recall
        //             << std::setw(12) << std::setprecision(3) << r.latency_ms
        //             << std::setw(12) << std::setprecision(1) << r.stats.ndis1_stats.mean
        //             << std::setw(12) << std::setprecision(1) << r.stats.ndis2_stats.mean
        //             << std::setw(12) << std::setprecision(1) << r.stats.nhops_stats.mean << "\n";
        //     }

        //     ofs.close();
        //     std::cout << "\nResults saved to: " << result_file << std::endl;
        // }
    }

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(1)
              << global_timer.elapsed_sec() << " seconds" << std::endl;

    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete hnsw_index;

    return 0;
}
