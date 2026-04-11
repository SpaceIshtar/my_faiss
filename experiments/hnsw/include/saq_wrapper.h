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

class SAQDistanceComputer : public faiss::DistanceComputer {
   public:
    using FastEstimator = saqlib::SaqCluEstimator<saqlib::DistType::L2Sqr>;

    explicit SAQDistanceComputer(const SAQWrapper* parent)
            : parent_(parent) {
        searcher_cfg_.dist_type = saqlib::DistType::L2Sqr;
        searcher_cfg_.searcher_vars_bound_m = 4.0f;
    }

    void set_query(const float* x) override;

    float operator()(faiss::idx_t i) override;

    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override;

    float symmetric_dis(faiss::idx_t /*i*/, faiss::idx_t /*j*/) override {
        throw std::runtime_error("SAQDistanceComputer::symmetric_dis not implemented");
    }

   private:
    const SAQWrapper* parent_;
    saqlib::SearcherConfig searcher_cfg_;
    saqlib::FloatVec query_;
    std::unique_ptr<FastEstimator> fast_estimator_;
    uint32_t prepared_cluster_ = std::numeric_limits<uint32_t>::max();
    uint32_t prepared_block_ = std::numeric_limits<uint32_t>::max();
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

        // 4) Reset encoded data. add() will populate searchable clusters.
        clusters_.clear();
        clusters_.resize(num_clusters_);
        for (size_t c = 0; c < num_clusters_; ++c) {
            clusters_[c] = std::make_unique<saqlib::SaqCluData>(
                    0,
                    saq_data_->quant_plan,
                    use_compact_layout_,
                    /*use_fastscan=*/true);
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
                        /*use_fastscan=*/true);
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
        // Only fastscan-trained files are supported now — reject anything
        // persisted with use_fastscan=false so we fail fast instead of
        // silently mis-decoding.
        if (!saq_data_->base_datas.empty() &&
            !saq_data_->base_datas.front().cfg.use_fastscan) {
            return false;
        }

        const uint64_t num_clusters_stored = read_u64();
        clusters_.clear();
        clusters_.resize(num_clusters_stored);
        for (size_t c = 0; c < num_clusters_stored; ++c) {
            const uint64_t num_vec = read_u64();
            clusters_[c] = std::make_unique<saqlib::SaqCluData>(
                        num_vec,
                        saq_data_->quant_plan,
                        use_compact_layout_,
                        /*use_fastscan=*/true);
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
    fast_estimator_ = std::make_unique<FastEstimator>(
            *parent_->saq_data_, searcher_cfg_, query_);
    prepared_cluster_ = std::numeric_limits<uint32_t>::max();
    prepared_block_   = std::numeric_limits<uint32_t>::max();
}

inline float SAQDistanceComputer::operator()(faiss::idx_t i) {
    if (i < 0 || static_cast<size_t>(i) >= parent_->ntotal_) {
        return std::numeric_limits<float>::infinity();
    }

    const uint32_t cid = parent_->vector_cluster_ids_[i];
    const uint32_t off = parent_->vector_offsets_[i];
    if (cid >= parent_->clusters_.size() ||
        off >= parent_->clusters_[cid]->num_vec_) {
        return std::numeric_limits<float>::infinity();
    }

    const uint32_t block = off / saqlib::KFastScanSize;
    if (cid != prepared_cluster_) {
        fast_estimator_->prepare(parent_->clusters_[cid].get());
        prepared_cluster_ = cid;
        prepared_block_   = std::numeric_limits<uint32_t>::max();
    }
    if (block != prepared_block_) {
        fast_estimator_->compFastDist(block, nullptr);
        prepared_block_ = block;
    }
    const float dist = fast_estimator_->compAccurateDist(off);
    if (!std::isfinite(dist)) {
        return std::numeric_limits<float>::infinity();
    }
    return dist;
}

inline void SAQDistanceComputer::distances_batch_4(
        const faiss::idx_t idx0,
        const faiss::idx_t idx1,
        const faiss::idx_t idx2,
        const faiss::idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    constexpr int N = 4;
    const faiss::idx_t idxs[N] = {idx0, idx1, idx2, idx3};

    uint32_t cids[N], offs[N], blks[N];
    for (int i = 0; i < N; i++) {
        cids[i] = parent_->vector_cluster_ids_[idxs[i]];
        offs[i] = parent_->vector_offsets_[idxs[i]];
        blks[i] = offs[i] / saqlib::KFastScanSize;
        if (cids[i] >= parent_->clusters_.size() ||
            offs[i] >= parent_->clusters_[cids[i]]->num_vec_) {
            dis0 = (*this)(idx0); dis1 = (*this)(idx1);
            dis2 = (*this)(idx2); dis3 = (*this)(idx3);
            return;
        }
    }

    const size_t n_segs = fast_estimator_->getEstimators().size();
    // Stack storage (supports up to 8 segments; fall back if more)
    constexpr size_t kMaxSegs = 8;
    if (n_segs > kMaxSegs) {
        dis0 = (*this)(idx0); dis1 = (*this)(idx1);
        dis2 = (*this)(idx2); dis3 = (*this)(idx3);
        return;
    }

    // Phase 1: Prepare each cluster+block sequentially, save per-segment data.
    // The Lut is shared across clusters (built once from q in the constructor),
    // so we only need to save the cluster-specific values before switching to
    // the next cluster.
    float saved_ip_xbq[N][kMaxSegs] = {};
    float saved_q_l2sqr[N][kMaxSegs] = {};
    float saved_cent_corr[N][kMaxSegs] = {};

    for (int i = 0; i < N; i++) {
        if (cids[i] != prepared_cluster_) {
            fast_estimator_->prepare(parent_->clusters_[cids[i]].get());
            prepared_cluster_ = cids[i];
            prepared_block_ = std::numeric_limits<uint32_t>::max();
        }
        if (blks[i] != prepared_block_) {
            fast_estimator_->compFastDist(blks[i], nullptr);
            prepared_block_ = blks[i];
        }
        // Save before this cluster's state is overwritten by next iteration
        for (size_t s = 0; s < n_segs; s++) {
            auto& est = fast_estimator_->getEstimators()[s];
            saved_ip_xbq[i][s] =
                    est.getLut().getIPXBQPrime(offs[i] % saqlib::KFastScanSize);
            saved_q_l2sqr[i][s] = est.getQl2sqr();
            saved_cent_corr[i][s] = est.getCentCorrection(offs[i]);
        }
    }

    // Phase 2: Per-segment batch IP + combine. The Lut is shared across all
    // clusters (built once from q in the constructor), so computeRawIPs_4
    // can pass in long_codes from four different clusters. The per-cluster
    // terms are saved_q_l2sqr (= ||q - c_i||²) and saved_cent_corr (=
    // 2 * rescale * <c_i, nominal_r>).
    float dis[N] = {};
    for (size_t s = 0; s < n_segs; s++) {
        const uint8_t* long_codes[N];
        float rescales[N], o_l2norms[N];
        for (int i = 0; i < N; i++) {
            const auto& seg = parent_->clusters_[cids[i]]->get_segment(s);
            long_codes[i] = seg.long_code(offs[i]);
            rescales[i] = seg.long_factor(offs[i]).rescale;
            o_l2norms[i] = seg.factor_o_l2norm(blks[i])[offs[i] % saqlib::KFastScanSize];
        }

        float raw[N];
        fast_estimator_->getEstimators()[s].getLut().computeRawIPs_4(
                long_codes[0], long_codes[1], long_codes[2], long_codes[3],
                raw[0], raw[1], raw[2], raw[3]);

        const float sq_delta = fast_estimator_->getEstimators()[s].getSqDelta();
        const float sum_q =
                fast_estimator_->getEstimators()[s].getLut().getSumQ();

        for (int i = 0; i < N; i++) {
            // ext_ip ≈ <q, nominal_r_i> (LUT built from q, sum_q = sum(q)).
            float ext_ip =
                    saved_ip_xbq[i][s] + raw[i] * sq_delta +
                    (-1.0f + sq_delta * 0.5f) * sum_q;
            float ip_o_q = rescales[i] * ext_ip;
            float o_l2sqr = o_l2norms[i] * o_l2norms[i];
            dis[i] += o_l2sqr + saved_q_l2sqr[i][s] - 2.0f * ip_o_q +
                    saved_cent_corr[i][s];
        }
    }

    dis0 = std::isfinite(dis[0]) ? std::max(0.0f, dis[0])
                                 : std::numeric_limits<float>::infinity();
    dis1 = std::isfinite(dis[1]) ? std::max(0.0f, dis[1])
                                 : std::numeric_limits<float>::infinity();
    dis2 = std::isfinite(dis[2]) ? std::max(0.0f, dis[2])
                                 : std::numeric_limits<float>::infinity();
    dis3 = std::isfinite(dis[3]) ? std::max(0.0f, dis[3])
                                 : std::numeric_limits<float>::infinity();
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
