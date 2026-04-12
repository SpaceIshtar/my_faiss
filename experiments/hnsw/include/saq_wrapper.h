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
#include "quantization/caq/caq_encoder.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "utils/code_helper.hpp"
#include "utils/memory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// =====================================================================
// Per-vector flat blob layout
//
//   [  0, 4)            cluster_id (u32)
//   [  4, 8)            padding
//   [  8, 8 + 12*n_segs)  per-segment factors {o_l2norm, ip_cent_oa, rescale}
//   [ ... ]             per-segment canonical N-bit codes
//                       (concatenated, each segment 16-byte aligned)
//   padded up to 64-byte stride
//
// A single distance computation reads exactly one contiguous slot;
// per-segment factors and code bytes live side-by-side and share the
// same stride.
// =====================================================================

class SAQWrapper;

class SAQDistanceComputer : public faiss::DistanceComputer {
  public:
    explicit SAQDistanceComputer(const SAQWrapper* parent) : parent_(parent) {}

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

    // Per-segment rotated query (q_rot[s]) and its running sum. Contiguous
    // storage is kept per-segment because each segment's padded_dim differs.
    std::vector<saqlib::FloatVec> q_rot_per_seg_;
    std::vector<float> sum_q_per_seg_;

    // Per-cluster total ||q - c||² = Σ_s ||q_s - c_s||², computed lazily
    // on first visit for a given query. cluster_gen_[cid] records which
    // query generation the cached value belongs to, so the cache stays
    // valid across queries without ever being explicitly cleared.
    uint32_t query_gen_ = 0;
    std::vector<uint32_t> cluster_gen_;
    std::vector<float> q_minus_c_total_;  // size = num_clusters

    inline float get_q_minus_c_total_(uint32_t cid);
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

        // 1) Run k-means to get coarse centroids.
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

        // 2) Copy input data to Eigen row-major matrix for SAQ variance.
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

        // 4) Build per-segment layout and vec_stride, then pre-rotate
        //    centroids per segment for use by set_query.
        build_seg_layout_();
        rotate_centroids_();

        // 5) Reset blob. add() will grow it.
        blob_.clear();
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
        blob_.resize(ntotal_ * vec_stride_, 0);

        // Assign to clusters.
        std::vector<float> coarse_dists(n);
        std::vector<faiss::idx_t> coarse_ids(n);
        quantizer_->search(n, x, 1, coarse_dists.data(), coarse_ids.data());

#pragma omp parallel
        {
            // Per-thread scratch for encoding: CAQEncoders (one per segment),
            // a uint16 buffer for compacted_code16 packing, and per-segment
            // rotated-residual / rotated-centroid buffers.
            std::vector<std::unique_ptr<saqlib::CAQEncoder>> encoders;
            encoders.reserve(n_segs_);
            for (size_t s = 0; s < n_segs_; ++s) {
                const auto& bd = saq_data_->base_datas[s];
                encoders.emplace_back(std::make_unique<saqlib::CAQEncoder>(
                        bd.num_dim_pad, bd.num_bits, bd.cfg));
            }
            std::vector<uint16_t> code_u16_buf(total_padded_dim_);
            saqlib::FloatVec rot_vec;
            saqlib::FloatVec raw_seg;

#pragma omp for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
                const auto cid = coarse_ids[i];
                if (cid < 0 || static_cast<size_t>(cid) >= num_clusters_) {
                    throw std::runtime_error("SAQWrapper::add got invalid coarse id");
                }
                const uint32_t gid = static_cast<uint32_t>(old_ntotal + i);
                uint8_t* slot = blob_.data() + gid * vec_stride_;
                std::memset(slot, 0, vec_stride_);
                *reinterpret_cast<uint32_t*>(slot) = static_cast<uint32_t>(cid);

                for (size_t s = 0; s < n_segs_; ++s) {
                    const auto& seg = segs_[s];
                    const auto& bd = saq_data_->base_datas[s];

                    // Extract raw slice (zero-padded to seg.num_dim_pad).
                    raw_seg.setZero(seg.num_dim_pad);
                    raw_seg.head(seg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                            x + i * d_ + seg.src_offset, seg.num_dim_copy);

                    // Rotate raw slice via segment's rotator (if any).
                    if (bd.rotator) {
                        rot_vec = raw_seg * bd.rotator->get_P();
                    } else {
                        rot_vec = raw_seg;
                    }

                    // Residual = rotated_vec - rotated_centroid.
                    const float* rot_cent_ptr = rotated_centroids_.data() +
                            (static_cast<size_t>(cid) * total_padded_dim_ +
                             seg.padded_offset);
                    Eigen::Map<const saqlib::FloatVec> rot_cent_vec(
                            rot_cent_ptr, seg.num_dim_pad);
                    saqlib::FloatVec residual = rot_vec - rot_cent_vec;

                    // Encode residual → full N-bit canonical code + factors.
                    saqlib::QuantBaseCode base_code;
                    saqlib::FloatVec rot_cent_copy = rot_cent_vec;
                    encoders[s]->encode_and_fac(residual, base_code, &rot_cent_copy);

                    // Write factors.
                    float* fac = reinterpret_cast<float*>(slot + seg.blob_factor_offset);
                    fac[0] = base_code.o_l2norm;
                    fac[1] = base_code.ip_cent_oa;
                    fac[2] = base_code.fac_rescale;

                    // Pack canonical N-bit code via CodeHelper<N>::compacted_code16.
                    if (seg.num_bits > 0 && base_code.code.size() > 0) {
                        for (size_t j = 0; j < seg.num_dim_pad; ++j) {
                            code_u16_buf[j] = static_cast<uint16_t>(base_code.code[j]);
                        }
                        auto pack_fn = saqlib::utils::get_compacted_code16_func(
                                static_cast<int>(seg.num_bits));
                        pack_fn(slot + seg.blob_code_offset,
                                code_u16_buf.data(),
                                seg.num_dim_pad);
                    }
                }
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

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return ntotal_; }
    size_t get_num_clusters() const { return num_clusters_; }
    float get_avg_bits() const { return avg_bits_; }

    bool save(const std::string& path) override {
        if (!is_trained_ || !saq_data_) {
            return false;
        }

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            return false;
        }

        const uint64_t magic = 0x5341515752415050ULL;  // SAQWRAPP
        const uint64_t version = 2;
        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

        auto write_u64 = [&](uint64_t v) {
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        auto write_u8 = [&](uint8_t v) {
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

        saq_data_->save(ofs);

        write_u64(vec_stride_);
        write_u64(blob_.size());
        ofs.write(reinterpret_cast<const char*>(blob_.data()), blob_.size());

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
        if (magic != 0x5341515752415050ULL || version != 2) {
            return false;
        }

        auto read_u64 = [&]() {
            uint64_t v = 0;
            ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
            return v;
        };
        auto read_u8 = [&]() {
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

        saq_data_ = std::make_unique<saqlib::SaqData>();
        saq_data_->load(ifs);

        build_seg_layout_();
        rotate_centroids_();

        const uint64_t vec_stride_on_disk = read_u64();
        if (vec_stride_on_disk != vec_stride_) {
            return false;
        }
        const uint64_t blob_size = read_u64();
        if (blob_size != ntotal * vec_stride_) {
            return false;
        }
        blob_.resize(blob_size);
        ifs.read(reinterpret_cast<char*>(blob_.data()), blob_size);

        if (!ifs.good()) {
            return false;
        }

        quantizer_ = std::make_unique<faiss::IndexFlatL2>(d_);
        quantizer_->add(num_clusters_, centroids_.data());

        ntotal_ = ntotal;
        is_trained_ = true;
        return true;
    }

  private:
    friend class SAQDistanceComputer;

    // Per-segment packed layout (computed from saq_data_->base_datas[s]).
    struct SegLayout {
        size_t num_dim_pad;       // D_s
        size_t num_dim_copy;      // how many of num_dim_pad are from raw input
        size_t src_offset;        // offset into the raw d_-dim vector
        size_t padded_offset;     // offset into concat padded space
        size_t num_bits;          // N_s
        size_t code_bytes;        // D_s * N_s / 8
        size_t blob_factor_offset; // byte offset into vec_stride_ for {xn, icn, rs}
        size_t blob_code_offset;  // byte offset into vec_stride_ for code bytes
        float caq_delta;          // 2 / 2^N_s
        float v_nom;              // -1 + caq_delta/2
        float (*ip_func)(const float*, const uint8_t*, size_t);
        saqlib::utils::IP_FUNC_4_t ip_func_4;
    };

    // Per-segment partial distance contribution. Excludes the per-
    // cluster ||q - c||² term, which is added once per vector outside
    // the segment loop.
    //
    //   ext_s  = caq_delta_s * <q_rot_s, code> + v_nom_s * sum_q_s
    //          ≈ <q_rot_s, nominal_r_s>
    //   part_s = o_l2sqr_s - 2 * rescale_s * (ext_s - icn_s)
    //
    // Full distance: ||q - c||² + Σ_s part_s.
    static inline float seg_partial_from_ip(
            const SegLayout& seg,
            const uint8_t* slot,
            float ip_q_code,
            float sum_q_s) {
        const float* fac = reinterpret_cast<const float*>(slot + seg.blob_factor_offset);
        const float xn = fac[0];
        const float icn = fac[1];
        const float rs = fac[2];
        const float ext = seg.caq_delta * ip_q_code + seg.v_nom * sum_q_s;
        return xn * xn - 2.0f * rs * (ext - icn);
    }

    void build_seg_layout_() {
        segs_.clear();
        n_segs_ = saq_data_->base_datas.size();
        segs_.reserve(n_segs_);

        size_t src_offset_acc = 0;
        size_t padded_offset_acc = 0;
        // Factors region: 12 bytes per seg, starts at byte 8.
        constexpr size_t kHeaderBytes = 8;
        constexpr size_t kFactorBytesPerSeg = 12;
        size_t code_region_begin = kHeaderBytes + kFactorBytesPerSeg * n_segs_;
        // Round code region start up to 16-byte boundary so per-seg codes
        // line up with SIMD loads.
        code_region_begin = (code_region_begin + 15) & ~size_t(15);

        size_t code_cursor = code_region_begin;
        for (size_t s = 0; s < n_segs_; ++s) {
            const auto& bd = saq_data_->base_datas[s];
            SegLayout seg{};
            seg.num_dim_pad = bd.num_dim_pad;
            seg.num_bits = bd.num_bits;
            seg.src_offset = src_offset_acc;
            seg.padded_offset = padded_offset_acc;
            seg.num_dim_copy = std::min(seg.num_dim_pad,
                                        d_ > src_offset_acc ? d_ - src_offset_acc : size_t(0));
            seg.code_bytes = seg.num_dim_pad * seg.num_bits / 8;
            seg.blob_factor_offset = kHeaderBytes + kFactorBytesPerSeg * s;
            seg.blob_code_offset = code_cursor;
            seg.caq_delta = seg.num_bits > 0
                    ? 2.0f / static_cast<float>(1 << seg.num_bits)
                    : 0.0f;
            seg.v_nom = seg.num_bits > 0
                    ? -1.0f + seg.caq_delta * 0.5f
                    : 0.0f;
            seg.ip_func = saqlib::utils::get_IP_FUNC(static_cast<int>(seg.num_bits));
            seg.ip_func_4 = saqlib::utils::get_IP_FUNC_4(static_cast<int>(seg.num_bits));

            // Round per-seg code region up to 16 bytes so the next segment's
            // code starts aligned.
            const size_t code_padded = (seg.code_bytes + 15) & ~size_t(15);
            code_cursor += code_padded;

            src_offset_acc += seg.num_dim_pad;
            padded_offset_acc += seg.num_dim_pad;
            segs_.push_back(seg);
        }

        total_padded_dim_ = padded_offset_acc;
        vec_stride_ = (code_cursor + 63) & ~size_t(63);  // round to cache line
    }

    void rotate_centroids_() {
        rotated_centroids_.assign(num_clusters_ * total_padded_dim_, 0.0f);
        saqlib::FloatVec raw_seg;
        saqlib::FloatVec rot_vec;
        for (size_t c = 0; c < num_clusters_; ++c) {
            for (size_t s = 0; s < n_segs_; ++s) {
                const auto& seg = segs_[s];
                const auto& bd = saq_data_->base_datas[s];

                raw_seg.setZero(seg.num_dim_pad);
                raw_seg.head(seg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                        centroids_.data() + c * d_ + seg.src_offset,
                        seg.num_dim_copy);

                if (bd.rotator) {
                    rot_vec = raw_seg * bd.rotator->get_P();
                } else {
                    rot_vec = raw_seg;
                }

                float* dst = rotated_centroids_.data() +
                        c * total_padded_dim_ + seg.padded_offset;
                std::memcpy(dst, rot_vec.data(), sizeof(float) * seg.num_dim_pad);
            }
        }
    }

    // Configuration
    size_t d_;
    float avg_bits_;
    size_t num_clusters_;
    bool enable_segmentation_;
    int seg_eqseg_;
    bool use_compact_layout_;
    bool random_rotation_;
    faiss::MetricType metric_;

    // Runtime state
    size_t ntotal_ = 0;
    bool is_trained_ = false;
    size_t n_segs_ = 0;
    size_t total_padded_dim_ = 0;
    size_t vec_stride_ = 0;

    std::vector<SegLayout> segs_;

    // Training artifacts
    std::vector<float> centroids_;                  // num_clusters * d_, raw
    std::vector<float, saqlib::memory::AlignedAllocator<float, 64>>
            rotated_centroids_;                     // num_clusters * total_padded_dim_
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::unique_ptr<saqlib::SaqData> saq_data_;

    // Flat per-vec blob
    std::vector<uint8_t, saqlib::memory::AlignedAllocator<uint8_t, 64>> blob_;
};

// ---------------------------------------------------------------------
// SAQDistanceComputer implementation
// ---------------------------------------------------------------------

inline void SAQDistanceComputer::set_query(const float* x) {
    const size_t n_segs = parent_->n_segs_;
    const size_t num_clusters = parent_->num_clusters_;

    // Rotate query per segment, cache sum_q per segment.
    q_rot_per_seg_.resize(n_segs);
    sum_q_per_seg_.resize(n_segs);
    for (size_t s = 0; s < n_segs; ++s) {
        const auto& seg = parent_->segs_[s];
        const auto& bd = parent_->saq_data_->base_datas[s];

        saqlib::FloatVec raw_seg = saqlib::FloatVec::Zero(seg.num_dim_pad);
        raw_seg.head(seg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                x + seg.src_offset, seg.num_dim_copy);
        if (bd.rotator) {
            q_rot_per_seg_[s] = raw_seg * bd.rotator->get_P();
        } else {
            q_rot_per_seg_[s] = raw_seg;
        }
        sum_q_per_seg_[s] = q_rot_per_seg_[s].sum();
    }

    // Bump query generation. get_q_minus_c_total_ uses cluster_gen_[cid]
    // to decide whether the cached value is fresh for this query, which
    // avoids re-zeroing num_clusters entries on every set_query.
    if (cluster_gen_.size() != num_clusters) {
        cluster_gen_.assign(num_clusters, 0);
        q_minus_c_total_.assign(num_clusters, 0.0f);
    }
    ++query_gen_;
    if (query_gen_ == 0) {
        // Counter wrap-around: reset all generations and restart.
        std::fill(cluster_gen_.begin(), cluster_gen_.end(), 0);
        query_gen_ = 1;
    }
}

inline float SAQDistanceComputer::get_q_minus_c_total_(uint32_t cid) {
    if (cluster_gen_[cid] == query_gen_) {
        return q_minus_c_total_[cid];
    }
    const size_t n_segs = parent_->n_segs_;
    float acc = 0.0f;
    for (size_t s = 0; s < n_segs; ++s) {
        const auto& seg = parent_->segs_[s];
        const float* cent = parent_->rotated_centroids_.data() +
                static_cast<size_t>(cid) * parent_->total_padded_dim_ +
                seg.padded_offset;
        const float* q = q_rot_per_seg_[s].data();
        const size_t D = seg.num_dim_pad;

        // num_dim_pad is always a multiple of saqlib::kDimPaddingSize (= 64),
        // enforced in SaqDataMaker, so no scalar tail is needed.
        // FP non-associativity blocks auto-vectorization of a scalar
        // sum-of-squares reduction, so the 16-way FMA chain is issued
        // explicitly.
        __m512 acc_vec = _mm512_setzero_ps();
        for (size_t j = 0; j < D; j += 16) {
            __m512 qv = _mm512_loadu_ps(q + j);
            __m512 cv = _mm512_loadu_ps(cent + j);
            __m512 diff = _mm512_sub_ps(qv, cv);
            acc_vec = _mm512_fmadd_ps(diff, diff, acc_vec);
        }
        acc += _mm512_reduce_add_ps(acc_vec);
    }
    q_minus_c_total_[cid] = acc;
    cluster_gen_[cid] = query_gen_;
    return acc;
}

inline float SAQDistanceComputer::operator()(faiss::idx_t i) {
    if (i < 0 || static_cast<size_t>(i) >= parent_->ntotal_) {
        return std::numeric_limits<float>::infinity();
    }
    const uint8_t* slot = parent_->blob_.data() + static_cast<size_t>(i) * parent_->vec_stride_;
    const uint32_t cid = *reinterpret_cast<const uint32_t*>(slot);
    const size_t n_segs = parent_->n_segs_;

    float dist = get_q_minus_c_total_(cid);
    for (size_t s = 0; s < n_segs; ++s) {
        const auto& seg = parent_->segs_[s];
        if (seg.num_bits == 0) {
            const float* fac = reinterpret_cast<const float*>(slot + seg.blob_factor_offset);
            const float xn = fac[0];
            dist += xn * xn;
            continue;
        }
        const uint8_t* code = slot + seg.blob_code_offset;
        const float ip_q_code = seg.ip_func(
                q_rot_per_seg_[s].data(), code, seg.num_dim_pad);
        dist += SAQWrapper::seg_partial_from_ip(
                seg, slot, ip_q_code, sum_q_per_seg_[s]);
    }
    // Clamp to a non-negative L2² value; compiles to a branchless maxss.
    return std::max(0.0f, dist);
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
    const faiss::idx_t idxs[4] = {idx0, idx1, idx2, idx3};
    for (int k = 0; k < 4; ++k) {
        if (idxs[k] < 0 || static_cast<size_t>(idxs[k]) >= parent_->ntotal_) {
            dis0 = (*this)(idx0); dis1 = (*this)(idx1);
            dis2 = (*this)(idx2); dis3 = (*this)(idx3);
            return;
        }
    }

    const size_t n_segs = parent_->n_segs_;
    const uint8_t* slots[4];
    uint32_t cids[4];
    for (int k = 0; k < 4; ++k) {
        slots[k] = parent_->blob_.data() + static_cast<size_t>(idxs[k]) * parent_->vec_stride_;
        cids[k] = *reinterpret_cast<const uint32_t*>(slots[k]);
    }

    // Per-cluster ||q - c||² once per vector (cached across vectors in
    // the same cluster via cluster_gen_).
    float dis[4];
    for (int k = 0; k < 4; ++k) {
        dis[k] = get_q_minus_c_total_(cids[k]);
    }

    for (size_t s = 0; s < n_segs; ++s) {
        const auto& seg = parent_->segs_[s];
        if (seg.num_bits == 0) {
            for (int k = 0; k < 4; ++k) {
                const float* fac = reinterpret_cast<const float*>(
                        slots[k] + seg.blob_factor_offset);
                const float xn = fac[0];
                dis[k] += xn * xn;
            }
            continue;
        }

        // Batch 4-way IP sharing the query.
        const uint8_t* c0 = slots[0] + seg.blob_code_offset;
        const uint8_t* c1 = slots[1] + seg.blob_code_offset;
        const uint8_t* c2 = slots[2] + seg.blob_code_offset;
        const uint8_t* c3 = slots[3] + seg.blob_code_offset;
        float ip[4];
        seg.ip_func_4(
                q_rot_per_seg_[s].data(),
                c0, c1, c2, c3,
                seg.num_dim_pad,
                ip[0], ip[1], ip[2], ip[3]);
        const float sum_q_s = sum_q_per_seg_[s];
        for (int k = 0; k < 4; ++k) {
            dis[k] += SAQWrapper::seg_partial_from_ip(
                    seg, slots[k], ip[k], sum_q_s);
        }
    }

    dis0 = std::max(0.0f, dis[0]);
    dis1 = std::max(0.0f, dis[1]);
    dis2 = std::max(0.0f, dis[2]);
    dis3 = std::max(0.0f, dis[3]);
}

inline bool parse_saq_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "yes";
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
