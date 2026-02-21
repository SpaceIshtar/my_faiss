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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnsw_bench {

class SAQWrapper;

class SAQDistanceComputer : public faiss::DistanceComputer {
   public:
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
    const SAQWrapper* parent_;
    saqlib::SearcherConfig searcher_cfg_;
    saqlib::FloatVec query_;
    std::unique_ptr<saqlib::SaqCluEstimator<saqlib::DistType::L2Sqr>> estimator_;
    uint32_t prepared_cluster_ = std::numeric_limits<uint32_t>::max();
    uint32_t prepared_block_ = std::numeric_limits<uint32_t>::max();
    alignas(64) __m512 fast_block_cache_[2];
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
            faiss::MetricType metric = faiss::METRIC_L2)
            : d_(d),
              avg_bits_(avg_bits),
              num_clusters_(num_clusters),
              enable_segmentation_(enable_segmentation),
              seg_eqseg_(seg_eqseg),
              use_compact_layout_(use_compact_layout),
              random_rotation_(random_rotation),
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

        std::vector<float> coarse_dists(n);
        std::vector<faiss::idx_t> coarse_ids(n);
        quantizer_->search(n, x, 1, coarse_dists.data(), coarse_ids.data());

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
        cfg.single.use_fastscan = true;

        saqlib::SaqDataMaker data_maker(cfg, d_);
        saqlib::FloatRowMat padded_data(n, data_maker.getPaddedDim());
        padded_data.setZero();
        padded_data.leftCols(d_) = data;
        data_maker.compute_variance(padded_data);
        saq_data_ = data_maker.return_data();

        // 4) Group ids by cluster and quantize each cluster.
        std::vector<std::vector<saqlib::PID>> id_lists(num_clusters_);
        for (size_t i = 0; i < n; ++i) {
            const auto cid = coarse_ids[i];
            if (cid >= 0) {
                id_lists[cid].push_back(static_cast<saqlib::PID>(i));
            }
        }

        clusters_.clear();
        clusters_.reserve(num_clusters_);
        for (size_t c = 0; c < num_clusters_; ++c) {
            clusters_.emplace_back(std::make_unique<saqlib::SaqCluData>(
                    id_lists[c].size(),
                    saq_data_->quant_plan,
                    use_compact_layout_));
        }

        saqlib::SAQuantizer saq_quantizer(saq_data_.get());
        for (size_t c = 0; c < num_clusters_; ++c) {
            if (id_lists[c].empty()) {
                continue;
            }
            saqlib::FloatVec centroid(d_);
            std::memcpy(
                    centroid.data(),
                    centroids_.data() + c * d_,
                    sizeof(float) * d_);
            saq_quantizer.quantize_cluster(
                    data, centroid, id_lists[c], *clusters_[c]);
        }

        // 5) Build global id -> (cluster, offset) map for O(1) random access.
        vector_cluster_ids_.assign(n, 0);
        vector_offsets_.assign(n, 0);
        for (size_t c = 0; c < num_clusters_; ++c) {
            auto* ids = clusters_[c]->ids();
            for (size_t off = 0; off < clusters_[c]->num_vec_; ++off) {
                const uint32_t gid = ids[off];
                vector_cluster_ids_[gid] = static_cast<uint32_t>(c);
                vector_offsets_[gid] = static_cast<uint32_t>(off);
            }
        }

        ntotal_ = n;
        is_trained_ = true;
    }

    void add(size_t /*n*/, const float* /*x*/) override {
        if (!is_trained_) {
            throw std::runtime_error("SAQWrapper::add called before train");
        }
        // Quantization is already completed in train().
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

        const uint64_t num_clusters_stored = read_u64();
        clusters_.clear();
        clusters_.reserve(num_clusters_stored);
        for (size_t c = 0; c < num_clusters_stored; ++c) {
            const uint64_t num_vec = read_u64();
            clusters_.emplace_back(std::make_unique<saqlib::SaqCluData>(
                    num_vec,
                    saq_data_->quant_plan,
                    use_compact_layout_));
            clusters_.back()->load(ifs);
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
    query_.resize(parent_->d_);
    std::memcpy(query_.data(), x, sizeof(float) * parent_->d_);
    estimator_ = std::make_unique<saqlib::SaqCluEstimator<saqlib::DistType::L2Sqr>>(
            *parent_->saq_data_,
            searcher_cfg_,
            query_);
    prepared_cluster_ = std::numeric_limits<uint32_t>::max();
    prepared_block_ = std::numeric_limits<uint32_t>::max();
}

inline float SAQDistanceComputer::operator()(faiss::idx_t i) {
    if (!estimator_ || i < 0 || static_cast<size_t>(i) >= parent_->ntotal_) {
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

    if (cid != prepared_cluster_) {
        estimator_->prepare(parent_->clusters_[cid].get());
        prepared_cluster_ = cid;
        prepared_block_ = std::numeric_limits<uint32_t>::max();
    }

    // Upstream SAQ estimator requires compFastDist(block) before compAccurateDist(vec).
    const uint32_t block = off / saqlib::KFastScanSize;
    if (block != prepared_block_) {
        estimator_->compFastDist(block, fast_block_cache_);
        prepared_block_ = block;
    }
    const float dist = estimator_->compAccurateDist(off);
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

    return std::make_unique<SAQWrapper>(
            d,
            avg_bits,
            clusters,
            enable_segmentation,
            seg_eqseg,
            use_compact_layout,
            random_rotation,
            metric);
}

} // namespace hnsw_bench
