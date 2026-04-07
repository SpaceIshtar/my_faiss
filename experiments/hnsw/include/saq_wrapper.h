/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>

#include "defines.hpp"
#include "quantization/cluster_data.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hnsw_bench {

class SAQWrapper;

struct SAQProfileStats {
    std::atomic<uint64_t> set_query_calls{0};
    std::atomic<uint64_t> set_query_ns{0};
    std::atomic<uint64_t> prototype_build_calls{0};
    std::atomic<uint64_t> prototype_build_ns{0};
    std::atomic<uint64_t> cluster_prepare_calls{0};
    std::atomic<uint64_t> cluster_prepare_ns{0};
    std::atomic<uint64_t> fast_block_calls{0};
    std::atomic<uint64_t> fast_block_ns{0};
    std::atomic<uint64_t> accurate_calls{0};
    std::atomic<uint64_t> accurate_ns{0};

    static bool enabled() {
        static const bool kEnabled = []() {
            const char* env = std::getenv("HNSW_SAQ_PROFILE");
            return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        }();
        return kEnabled;
    }

    void report() const {
        const auto sq_calls = set_query_calls.load(std::memory_order_relaxed);
        if (!enabled() || sq_calls == 0) {
            return;
        }

        const auto proto_calls = prototype_build_calls.load(std::memory_order_relaxed);
        const auto prep_calls = cluster_prepare_calls.load(std::memory_order_relaxed);
        const auto fast_calls = fast_block_calls.load(std::memory_order_relaxed);
        const auto acc_calls = accurate_calls.load(std::memory_order_relaxed);

        auto to_ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
        auto avg_us = [](uint64_t ns, uint64_t calls) {
            return calls ? static_cast<double>(ns) / calls / 1e3 : 0.0;
        };

        std::cerr << "\n[SAQ HNSW Profile]\n"
                  << "  set_query:        calls=" << sq_calls
                  << " total_ms=" << to_ms(set_query_ns.load(std::memory_order_relaxed))
                  << " avg_us=" << avg_us(set_query_ns.load(std::memory_order_relaxed), sq_calls) << "\n"
                  << "  prototype_build:  calls=" << proto_calls
                  << " total_ms=" << to_ms(prototype_build_ns.load(std::memory_order_relaxed))
                  << " avg_us=" << avg_us(prototype_build_ns.load(std::memory_order_relaxed), proto_calls) << "\n"
                  << "  cluster_prepare:  calls=" << prep_calls
                  << " total_ms=" << to_ms(cluster_prepare_ns.load(std::memory_order_relaxed))
                  << " avg_us=" << avg_us(cluster_prepare_ns.load(std::memory_order_relaxed), prep_calls)
                  << " avg_per_query=" << (sq_calls ? static_cast<double>(prep_calls) / sq_calls : 0.0) << "\n"
                  << "  fast_block:       calls=" << fast_calls
                  << " total_ms=" << to_ms(fast_block_ns.load(std::memory_order_relaxed))
                  << " avg_us=" << avg_us(fast_block_ns.load(std::memory_order_relaxed), fast_calls)
                  << " avg_per_query=" << (sq_calls ? static_cast<double>(fast_calls) / sq_calls : 0.0) << "\n"
                  << "  accurate:         calls=" << acc_calls
                  << " total_ms=" << to_ms(accurate_ns.load(std::memory_order_relaxed))
                  << " avg_us=" << avg_us(accurate_ns.load(std::memory_order_relaxed), acc_calls)
                  << " avg_per_query=" << (sq_calls ? static_cast<double>(acc_calls) / sq_calls : 0.0) << "\n";
    }

    ~SAQProfileStats() {
        report();
    }
};

inline SAQProfileStats& saq_profile_stats() {
    static SAQProfileStats stats;
    return stats;
}

template <typename F>
inline auto saq_profile_scope(std::atomic<uint64_t>& total_ns, F&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    auto result = fn();
    const auto t1 = std::chrono::steady_clock::now();
    total_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    return result;
}

class SAQDistanceComputer : public faiss::DistanceComputer {
   public:
    using FastEstimator = saqlib::SaqCluEstimator<saqlib::DistType::L2Sqr>;
    using SingleEstimator = saqlib::SaqCluEstimatorSingle<saqlib::DistType::L2Sqr>;

    explicit SAQDistanceComputer(const SAQWrapper* parent)
            : parent_(parent) {
        searcher_cfg_.dist_type = saqlib::DistType::L2Sqr;
        searcher_cfg_.searcher_vars_bound_m = 4.0f;
    }

    void set_query(const float* x) override;

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t /*i*/, faiss::idx_t /*j*/) override {
        throw std::runtime_error("SAQDistanceComputer::symmetric_dis not implemented");
    }

   private:
    struct PreparedClusterState {
        std::unique_ptr<FastEstimator> fast_estimator;
        std::unique_ptr<SingleEstimator> single_estimator;
        uint32_t generation = 0;
        uint32_t prepared_block = std::numeric_limits<uint32_t>::max();
    };

    const SAQWrapper* parent_;
    saqlib::SearcherConfig searcher_cfg_;
    saqlib::FloatVec query_;
    bool use_cluster_cache_ = false;
    bool use_fastscan_path_ = true;
    std::unique_ptr<FastEstimator> fast_estimator_;
    std::unique_ptr<SingleEstimator> single_estimator_;
    uint32_t prepared_cluster_ = std::numeric_limits<uint32_t>::max();
    uint32_t prepared_block_ = std::numeric_limits<uint32_t>::max();
    std::vector<PreparedClusterState> cluster_cache_;
    uint32_t query_generation_ = 0;
};

class SAQWrapper : public QuantWrapper {
   public:
    SAQWrapper(
            size_t d,
            float avg_bits = 4.0f,
            size_t num_clusters = 4096,
            bool enable_segmentation = true,
            int seg_eqseg = 0,
            bool use_compact_layout = false,
            bool random_rotation = true,
            bool use_fastscan = false,
            bool cluster_cache_enabled = true,
            faiss::MetricType metric = faiss::METRIC_L2)
            : d_(d),
              avg_bits_(avg_bits),
              num_clusters_(num_clusters),
              enable_segmentation_(enable_segmentation),
              seg_eqseg_(seg_eqseg),
              use_compact_layout_(use_compact_layout),
              random_rotation_(random_rotation),
              use_fastscan_(use_fastscan),
              cluster_cache_enabled_(cluster_cache_enabled),
              metric_(metric) {
        if (d_ == 0) {
            throw std::runtime_error("SAQWrapper: dimension must be > 0");
        }
        if (avg_bits_ <= 0.0f) {
            throw std::runtime_error("SAQWrapper: avg_bits must be > 0");
        }
        if (num_clusters_ == 0) {
            throw std::runtime_error("SAQWrapper: num_clusters must be > 0");
        }
        if (metric_ != faiss::METRIC_L2) {
            throw std::runtime_error("SAQWrapper currently supports METRIC_L2 only");
        }
    }

    void train(size_t n, const float* x) override {
        if (n == 0 || x == nullptr) {
            throw std::runtime_error("SAQWrapper::train requires non-empty input");
        }

        // 1) Run k-means to get coarse centroids and cluster assignments.
        faiss::ClusteringParameters cp;
        cp.niter = 25;
        cp.seed = 1234;
        cp.verbose = false;

        faiss::Clustering clus(d_, num_clusters_, cp);
        faiss::IndexFlatL2 kmeans_index(d_);
        clus.train(n, x, kmeans_index);

        centroids_.assign(clus.centroids.begin(), clus.centroids.end());
        quantizer_ = std::make_unique<faiss::IndexFlatL2>(d_);
        quantizer_->add(num_clusters_, centroids_.data());

        // 2) Copy input data to Eigen row-major matrix for SAQ quantization.
        saqlib::FloatRowMat data(n, d_);
#pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            std::memcpy(data.row(i).data(), x + i * d_, sizeof(float) * d_);
        }

        // 3) Build SAQ quantization plan from data variance.
        saqlib::QuantizeConfig cfg;
        cfg.avg_bits = avg_bits_;
        cfg.enable_segmentation = enable_segmentation_;
        cfg.seg_eqseg = seg_eqseg_;
        cfg.use_compact_layout = use_compact_layout_;
        cfg.single.random_rotation = random_rotation_;
        cfg.single.use_fastscan = use_fastscan_;

        saqlib::SaqDataMaker data_maker(cfg, d_);
        saqlib::FloatRowMat padded_data(n, data_maker.getPaddedDim());
        padded_data.setZero();
        padded_data.leftCols(d_) = data;
        data_maker.compute_variance(padded_data);
        saq_data_ = data_maker.return_data();

        // 4) Reset encoded data. add() will populate searchable clusters.
        const bool use_fastscan =
                saq_data_ && !saq_data_->base_datas.empty()
                ? saq_data_->base_datas.front().cfg.use_fastscan
                : true;
        clusters_.clear();
        clusters_.resize(num_clusters_);
        for (size_t c = 0; c < num_clusters_; ++c) {
            clusters_[c] = std::make_unique<saqlib::SaqCluData>(
                    0,
                    saq_data_->quant_plan,
                    use_compact_layout_,
                    use_fastscan);
        }
        vector_cluster_ids_.clear();
        vector_offsets_.clear();
        ntotal_ = 0;
        is_trained_ = true;
    }

    void add(size_t n, const float* x) override {
        if (!is_trained_) {
            throw std::runtime_error("SAQWrapper::add called before train");
        }
        if (n == 0 || x == nullptr) {
            return;
        }

        const size_t old_ntotal = ntotal_;
        ntotal_ += n;
        vector_cluster_ids_.resize(ntotal_);
        vector_offsets_.resize(ntotal_);

        std::vector<float> coarse_dists(n);
        std::vector<faiss::idx_t> coarse_ids(n);
        quantizer_->search(n, x, 1, coarse_dists.data(), coarse_ids.data());

        saqlib::FloatRowMat data(n, d_);
#pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            std::memcpy(data.row(i).data(), x + i * d_, sizeof(float) * d_);
        }

        std::vector<std::vector<saqlib::PID>> local_id_lists(num_clusters_);
        for (size_t i = 0; i < n; ++i) {
            const auto cid = coarse_ids[i];
            if (cid < 0 || static_cast<size_t>(cid) >= num_clusters_) {
                throw std::runtime_error("SAQWrapper::add got invalid coarse cluster id");
            }
            local_id_lists[cid].push_back(static_cast<saqlib::PID>(i));
        }

        const bool use_fastscan =
                saq_data_ && !saq_data_->base_datas.empty()
                ? saq_data_->base_datas.front().cfg.use_fastscan
                : true;
        std::vector<uint32_t> base_offsets(num_clusters_, 0);
        for (size_t c = 0; c < num_clusters_; ++c) {
            if (clusters_[c]) {
                base_offsets[c] = static_cast<uint32_t>(clusters_[c]->num_vec_);
            }
        }

#pragma omp parallel
        {
            saqlib::SAQuantizer saq_quantizer(saq_data_.get());
            saqlib::FloatVec centroid(d_);

#pragma omp for schedule(dynamic, 1)
            for (int64_t c = 0; c < static_cast<int64_t>(num_clusters_); ++c) {
                const auto& ids_in_cluster = local_id_lists[c];
                if (ids_in_cluster.empty()) {
                    continue;
                }

                auto cluster = std::make_unique<saqlib::SaqCluData>(
                        ids_in_cluster.size(),
                        saq_data_->quant_plan,
                        use_compact_layout_,
                        use_fastscan);
                std::memcpy(
                        centroid.data(),
                        centroids_.data() + static_cast<size_t>(c) * d_,
                        sizeof(float) * d_);

                saq_quantizer.quantize_cluster(
                        data, centroid, ids_in_cluster, *cluster);

                auto* ids = cluster->ids();
                const uint32_t base_off = base_offsets[c];
                for (size_t off = 0; off < ids_in_cluster.size(); ++off) {
                    const auto gid =
                            static_cast<uint32_t>(old_ntotal + static_cast<size_t>(ids_in_cluster[off]));
                    ids[off] = static_cast<saqlib::PID>(gid);
                    vector_cluster_ids_[gid] = static_cast<uint32_t>(c);
                    vector_offsets_[gid] = base_off + static_cast<uint32_t>(off);
                }

                clusters_[c]->append(*cluster);
            }
        }
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        if (!is_trained_) {
            throw std::runtime_error(
                    "SAQWrapper::get_distance_computer called before train/load");
        }
        return std::make_unique<SAQDistanceComputer>(this);
    }

    std::string get_name() const override {
        return "SAQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "bits" << avg_bits_ << "_c" << num_clusters_;
        if (!enable_segmentation_) {
            oss << "_noseg";
        }
        if (seg_eqseg_ > 0) {
            oss << "_eqseg" << seg_eqseg_;
        }
        if (use_compact_layout_) {
            oss << "_compact";
        }
        if (!random_rotation_) {
            oss << "_norot";
        }
        if (!use_fastscan_) {
            oss << "_nofast";
        }
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return ntotal_;
    }

    bool save(const std::string& path) override {
        if (!is_trained_ || !saq_data_) {
            return false;
        }

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            return false;
        }

        const uint64_t magic = 0x5341515752415050ULL; // SAQWRAPP
        const uint64_t version = 1;
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

        auto write_u64 = [&ofs](uint64_t v) {
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        auto write_u8 = [&ofs](uint8_t v) {
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };

        write_u64(d_);
        write_u64(num_clusters_);
        ofs.write(reinterpret_cast<const char*>(&avg_bits_), sizeof(avg_bits_));
        write_u8(enable_segmentation_ ? 1 : 0);
        write_u64(static_cast<uint64_t>(seg_eqseg_));
        write_u8(use_compact_layout_ ? 1 : 0);
        write_u8(random_rotation_ ? 1 : 0);
        write_u64(ntotal_);

        const uint64_t centroids_size = centroids_.size();
        write_u64(centroids_size);
        ofs.write(
                reinterpret_cast<const char*>(centroids_.data()),
                centroids_size * sizeof(float));

        const uint64_t map_size = vector_cluster_ids_.size();
        write_u64(map_size);
        ofs.write(
                reinterpret_cast<const char*>(vector_cluster_ids_.data()),
                map_size * sizeof(uint32_t));
        ofs.write(
                reinterpret_cast<const char*>(vector_offsets_.data()),
                map_size * sizeof(uint32_t));

        saq_data_->save(ofs);

        write_u64(clusters_.size());
        for (const auto& c : clusters_) {
            write_u64(c->num_vec_);
            c->save(ofs);
        }

        ofs.close();
        return ofs.good();
    }

    bool load(const std::string& path) override {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }

        uint64_t magic = 0;
        uint64_t version = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != 0x5341515752415050ULL || version != 1) {
            return false;
        }

        auto read_u64 = [&ifs]() {
            uint64_t v = 0;
            ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };
        auto read_u8 = [&ifs]() {
            uint8_t v = 0;
            ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };

        const uint64_t d = read_u64();
        const uint64_t num_clusters = read_u64();
        float avg_bits = 0;
        ifs.read(reinterpret_cast<char*>(&avg_bits), sizeof(avg_bits));
        const bool enable_seg = read_u8() != 0;
        const int seg_eqseg = static_cast<int>(read_u64());
        const bool compact = read_u8() != 0;
        const bool random_rot = read_u8() != 0;
        const uint64_t ntotal = read_u64();

        // Require wrapper construction params to match persisted index params.
        if (d != d_ || num_clusters != num_clusters_ ||
            std::fabs(avg_bits - avg_bits_) > 1e-6 ||
            enable_seg != enable_segmentation_ ||
            seg_eqseg != seg_eqseg_ ||
            compact != use_compact_layout_ ||
            random_rot != random_rotation_) {
            return false;
        }

        const uint64_t centroids_size = read_u64();
        centroids_.resize(centroids_size);
        ifs.read(
                reinterpret_cast<char*>(centroids_.data()),
                centroids_size * sizeof(float));

        const uint64_t map_size = read_u64();
        vector_cluster_ids_.resize(map_size);
        vector_offsets_.resize(map_size);
        ifs.read(
                reinterpret_cast<char*>(vector_cluster_ids_.data()),
                map_size * sizeof(uint32_t));
        ifs.read(
                reinterpret_cast<char*>(vector_offsets_.data()),
                map_size * sizeof(uint32_t));

        saq_data_ = std::make_unique<saqlib::SaqData>();
        saq_data_->load(ifs);
        const bool loaded_use_fastscan =
                !saq_data_->base_datas.empty()
                ? saq_data_->base_datas.front().cfg.use_fastscan
                : true;
        if (loaded_use_fastscan != use_fastscan_) {
            return false;
        }

        const uint64_t num_clusters_stored = read_u64();
        const bool use_fastscan =
                !saq_data_->base_datas.empty()
                ? saq_data_->base_datas.front().cfg.use_fastscan
                : true;
        clusters_.clear();
        clusters_.resize(num_clusters_stored);
        for (size_t c = 0; c < num_clusters_stored; ++c) {
            const uint64_t num_vec = read_u64();
            clusters_[c] = std::make_unique<saqlib::SaqCluData>(
                        num_vec,
                        saq_data_->quant_plan,
                        use_compact_layout_,
                        use_fastscan);
            clusters_[c]->load(ifs);
        }

        if (!ifs.good()) {
            return false;
        }

        quantizer_ = std::make_unique<faiss::IndexFlatL2>(d_);
        quantizer_->add(num_clusters_, centroids_.data());

        ntotal_ = ntotal;
        is_trained_ = true;
        return true;
    }

    size_t get_num_clusters() const {
        return num_clusters_;
    }

    float get_avg_bits() const {
        return avg_bits_;
    }

   private:
    friend class SAQDistanceComputer;

    size_t d_;
    float avg_bits_;
    size_t num_clusters_;
    bool enable_segmentation_;
    int seg_eqseg_;
    bool use_compact_layout_;
    bool random_rotation_;
    bool use_fastscan_;
    bool cluster_cache_enabled_;
    faiss::MetricType metric_;

    size_t ntotal_ = 0;
    bool is_trained_ = false;

    std::vector<float> centroids_;
    std::vector<uint32_t> vector_cluster_ids_;
    std::vector<uint32_t> vector_offsets_;

    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::unique_ptr<saqlib::SaqData> saq_data_;
    std::vector<std::unique_ptr<saqlib::SaqCluData>> clusters_;
};

inline void SAQDistanceComputer::set_query(const float* x) {
    const bool profile_enabled = SAQProfileStats::enabled();
    const auto set_query_begin =
            profile_enabled ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
    query_.resize(parent_->d_);
    std::memcpy(query_.data(), x, sizeof(float) * parent_->d_);
    use_cluster_cache_ = parent_->cluster_cache_enabled_;
    use_fastscan_path_ = parent_->use_fastscan_;
    if (use_cluster_cache_) {
        if (cluster_cache_.size() != parent_->clusters_.size()) {
            cluster_cache_.clear();
            cluster_cache_.resize(parent_->clusters_.size());
            query_generation_ = 0;
        }
        query_generation_++;
        if (query_generation_ == 0) {
            query_generation_ = 1;
            for (auto& state : cluster_cache_) {
                state.generation = 0;
            }
        }
        if (profile_enabled) {
            saq_profile_stats().prototype_build_calls.fetch_add(1, std::memory_order_relaxed);
            if (use_fastscan_path_) {
                fast_estimator_ = saq_profile_scope(
                        saq_profile_stats().prototype_build_ns,
                        [&]() {
                            return std::make_unique<FastEstimator>(
                                    *parent_->saq_data_,
                                    searcher_cfg_,
                                    query_);
                        });
                single_estimator_.reset();
            } else {
                single_estimator_ = saq_profile_scope(
                        saq_profile_stats().prototype_build_ns,
                        [&]() {
                            return std::make_unique<SingleEstimator>(
                                    *parent_->saq_data_,
                                    searcher_cfg_,
                                    query_);
                        });
                fast_estimator_.reset();
            }
        } else {
            if (use_fastscan_path_) {
                fast_estimator_ = std::make_unique<FastEstimator>(
                        *parent_->saq_data_,
                        searcher_cfg_,
                        query_);
                single_estimator_.reset();
            } else {
                single_estimator_ = std::make_unique<SingleEstimator>(
                        *parent_->saq_data_,
                        searcher_cfg_,
                        query_);
                fast_estimator_.reset();
            }
        }
    } else {
        cluster_cache_.clear();
        if (profile_enabled) {
            saq_profile_stats().prototype_build_calls.fetch_add(1, std::memory_order_relaxed);
            if (use_fastscan_path_) {
                fast_estimator_ = saq_profile_scope(
                        saq_profile_stats().prototype_build_ns,
                        [&]() {
                            return std::make_unique<FastEstimator>(
                                    *parent_->saq_data_,
                                    searcher_cfg_,
                                    query_);
                        });
                single_estimator_.reset();
            } else {
                single_estimator_ = saq_profile_scope(
                        saq_profile_stats().prototype_build_ns,
                        [&]() {
                            return std::make_unique<SingleEstimator>(
                                    *parent_->saq_data_,
                                    searcher_cfg_,
                                    query_);
                        });
                fast_estimator_.reset();
            }
        } else {
            if (use_fastscan_path_) {
                fast_estimator_ = std::make_unique<FastEstimator>(
                        *parent_->saq_data_,
                        searcher_cfg_,
                        query_);
                single_estimator_.reset();
            } else {
                single_estimator_ = std::make_unique<SingleEstimator>(
                        *parent_->saq_data_,
                        searcher_cfg_,
                        query_);
                fast_estimator_.reset();
            }
        }
        prepared_cluster_ = std::numeric_limits<uint32_t>::max();
        prepared_block_ = std::numeric_limits<uint32_t>::max();
    }
    if (profile_enabled) {
        saq_profile_stats().set_query_calls.fetch_add(1, std::memory_order_relaxed);
        const auto set_query_end = std::chrono::steady_clock::now();
        saq_profile_stats().set_query_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                        set_query_end - set_query_begin)
                        .count(),
                std::memory_order_relaxed);
    }
}

inline float SAQDistanceComputer::operator()(faiss::idx_t i) {
    if (i < 0 || static_cast<size_t>(i) >= parent_->ntotal_) {
        return std::numeric_limits<float>::infinity();
    }

    const uint32_t cid = parent_->vector_cluster_ids_[i];
    const uint32_t off = parent_->vector_offsets_[i];
    if (cid >= parent_->clusters_.size()) {
        return std::numeric_limits<float>::infinity();
    }
    if (off >= parent_->clusters_[cid]->num_vec_) {
        return std::numeric_limits<float>::infinity();
    }

    float dist = std::numeric_limits<float>::infinity();
    const uint32_t block = off / saqlib::KFastScanSize;
    const bool profile_enabled = SAQProfileStats::enabled();
    if (use_cluster_cache_) {
        auto& state = cluster_cache_[cid];
        if (state.generation != query_generation_) {
            if (use_fastscan_path_) {
                state.fast_estimator = std::make_unique<FastEstimator>(*fast_estimator_);
                state.single_estimator.reset();
            } else {
                state.single_estimator = std::make_unique<SingleEstimator>(*single_estimator_);
                state.fast_estimator.reset();
            }
            if (profile_enabled) {
                saq_profile_stats().cluster_prepare_calls.fetch_add(1, std::memory_order_relaxed);
                saq_profile_scope(
                        saq_profile_stats().cluster_prepare_ns,
                        [&]() {
                            if (use_fastscan_path_) {
                                state.fast_estimator->prepare(parent_->clusters_[cid].get());
                            } else {
                                state.single_estimator->prepare(parent_->clusters_[cid].get());
                            }
                            return 0;
                        });
            } else {
                if (use_fastscan_path_) {
                    state.fast_estimator->prepare(parent_->clusters_[cid].get());
                } else {
                    state.single_estimator->prepare(parent_->clusters_[cid].get());
                }
            }
            state.generation = query_generation_;
            state.prepared_block = std::numeric_limits<uint32_t>::max();
        }

        if (use_fastscan_path_ && block != state.prepared_block) {
            if (profile_enabled) {
                saq_profile_stats().fast_block_calls.fetch_add(1, std::memory_order_relaxed);
                saq_profile_scope(
                        saq_profile_stats().fast_block_ns,
                        [&]() {
                            state.fast_estimator->compFastDist(block, nullptr);
                            return 0;
                        });
            } else {
                state.fast_estimator->compFastDist(block, nullptr);
            }
            state.prepared_block = block;
        }
        if (profile_enabled) {
            saq_profile_stats().accurate_calls.fetch_add(1, std::memory_order_relaxed);
            dist = saq_profile_scope(
                    saq_profile_stats().accurate_ns,
                    [&]() {
                        return use_fastscan_path_
                                ? state.fast_estimator->compAccurateDist(off)
                                : state.single_estimator->compAccurateDist(off);
                    });
        } else {
            dist = use_fastscan_path_
                    ? state.fast_estimator->compAccurateDist(off)
                    : state.single_estimator->compAccurateDist(off);
        }
    } else {
        if (cid != prepared_cluster_) {
            if (profile_enabled) {
                saq_profile_stats().cluster_prepare_calls.fetch_add(1, std::memory_order_relaxed);
                saq_profile_scope(
                        saq_profile_stats().cluster_prepare_ns,
                        [&]() {
                            if (use_fastscan_path_) {
                                fast_estimator_->prepare(parent_->clusters_[cid].get());
                            } else {
                                single_estimator_->prepare(parent_->clusters_[cid].get());
                            }
                            return 0;
                        });
            } else {
                if (use_fastscan_path_) {
                    fast_estimator_->prepare(parent_->clusters_[cid].get());
                } else {
                    single_estimator_->prepare(parent_->clusters_[cid].get());
                }
            }
            prepared_cluster_ = cid;
            prepared_block_ = std::numeric_limits<uint32_t>::max();
        }

        if (use_fastscan_path_ && block != prepared_block_) {
            if (profile_enabled) {
                saq_profile_stats().fast_block_calls.fetch_add(1, std::memory_order_relaxed);
                saq_profile_scope(
                        saq_profile_stats().fast_block_ns,
                        [&]() {
                            fast_estimator_->compFastDist(block, nullptr);
                            return 0;
                        });
            } else {
                fast_estimator_->compFastDist(block, nullptr);
            }
            prepared_block_ = block;
        }
        if (profile_enabled) {
            saq_profile_stats().accurate_calls.fetch_add(1, std::memory_order_relaxed);
            dist = saq_profile_scope(
                    saq_profile_stats().accurate_ns,
                    [&]() {
                        return use_fastscan_path_
                                ? fast_estimator_->compAccurateDist(off)
                                : single_estimator_->compAccurateDist(off);
                    });
        } else {
            dist = use_fastscan_path_
                    ? fast_estimator_->compAccurateDist(off)
                    : single_estimator_->compAccurateDist(off);
        }
    }
    if (!std::isfinite(dist)) {
        return std::numeric_limits<float>::infinity();
    }
    return dist;
}

inline bool parse_saq_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "on" ||
            v == "yes";
}

inline std::unique_ptr<QuantWrapper> create_saq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    float avg_bits = 4.0f;
    size_t clusters = 4096;
    bool enable_segmentation = true;
    int seg_eqseg = 0;
    bool use_compact_layout = false;
    bool random_rotation = true;
    bool use_fastscan = false;
    bool cluster_cache_enabled = true;

    if (auto it = params.find("avg_bits"); it != params.end()) {
        avg_bits = std::stof(it->second);
    }
    if (auto it = params.find("bits"); it != params.end()) {
        avg_bits = std::stof(it->second);
    }
    if (auto it = params.find("B"); it != params.end()) {
        avg_bits = std::stof(it->second);
    }
    if (auto it = params.find("clusters"); it != params.end()) {
        clusters = std::stoul(it->second);
    }
    if (auto it = params.find("enable_segmentation"); it != params.end()) {
        enable_segmentation = parse_saq_bool(it->second);
    }
    if (auto it = params.find("seg_eqseg"); it != params.end()) {
        seg_eqseg = std::stoi(it->second);
    }
    if (auto it = params.find("use_compact_layout"); it != params.end()) {
        use_compact_layout = parse_saq_bool(it->second);
    }
    if (auto it = params.find("rand_rotate"); it != params.end()) {
        random_rotation = parse_saq_bool(it->second);
    }
    if (auto it = params.find("use_fastscan"); it != params.end()) {
        use_fastscan = parse_saq_bool(it->second);
    }
    if (auto it = params.find("cluster_cache"); it != params.end()) {
        cluster_cache_enabled = parse_saq_bool(it->second);
    }

    return std::make_unique<SAQWrapper>(
            d,
            avg_bits,
            clusters,
            enable_segmentation,
            seg_eqseg,
            use_compact_layout,
            random_rotation,
            use_fastscan,
            cluster_cache_enabled,
            metric);
}

} // namespace hnsw_bench
