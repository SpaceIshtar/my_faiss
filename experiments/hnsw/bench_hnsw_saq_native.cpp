/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * HNSW + SAQ Native Search Benchmark
 *
 * Self-contained SAQ benchmark. Uses saqlib for encoding, FAISS HNSW for
 * the graph, and implements a custom two-stage search loop:
 *   1. Cheap 1-bit sign(residual) estimate via popcount kernel
 *   2. Full N-bit distance via CodeHelper<N> only if low_dist < bound
 *
 * Per-vector blob layout:
 *   [full factors + N-bit code]  (for accurate distance)
 *   [1-bit shadow factors + 1-bit sign code]  (for cheap screening)
 *
 * Usage:
 *   ./bench_hnsw_saq_native --dataset <name> [--data-path <p>] [--threads <n>]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include "defines.hpp"
#include "quantization/caq/caq_encoder.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "utils/code_helper.hpp"
#include "utils/memory.hpp"

#include "include/bench_config.h"
#include "include/bench_utils.h"

using namespace hnsw_bench;

// Error-bound multiplier for the 1-bit low_dist. Baked into f_error at
// encode time would prevent tuning; stored separately so it's applied at
// search time only. 1.9 is the standard value (same as the original paper).
static constexpr float kEpsilon = 1.9f;

// Popcount-based 1-bit inner product. Data is LSB-first packed
// (CodeHelper<1>::compacted_code8 layout). Query bit-planes are also
// LSB-first: query_bin[block*4 + j] bit d = bit j of quant_query[64*block+d].
//
// Returns delta * <q_int, bin_code> + vl * sum(bin_code) ≈ <q_float, bin_code>.
[[gnu::always_inline]]
static inline float popcount_ip_1bit(
        const uint8_t* __restrict__ data,
        const uint64_t* __restrict__ query_bin,
        float delta, float vl, size_t padded_dim) {
    const auto* d64 = reinterpret_cast<const uint64_t*>(data);
    const size_t nblk = padded_dim / 64;
    uint64_t ip_sum = 0, ppc_sum = 0;
    for (size_t i = 0; i < nblk; i++) {
        uint64_t x = d64[i];
        ppc_sum += __builtin_popcountll(x);
        const uint64_t* qb = query_bin + i * 4;
        ip_sum += (uint64_t)__builtin_popcountll(x & qb[0]);
        ip_sum += (uint64_t)__builtin_popcountll(x & qb[1]) << 1;
        ip_sum += (uint64_t)__builtin_popcountll(x & qb[2]) << 2;
        ip_sum += (uint64_t)__builtin_popcountll(x & qb[3]) << 3;
    }
    return delta * (float)ip_sum + vl * (float)ppc_sum;
}

// Batch-4 popcount IP: interleaves memory loads from 4 data streams so the
// CPU can issue 4 independent load chains in parallel.
[[gnu::always_inline]]
static inline void popcount_ip_1bit_4(
        const uint8_t* __restrict__ d0, const uint8_t* __restrict__ d1,
        const uint8_t* __restrict__ d2, const uint8_t* __restrict__ d3,
        const uint64_t* __restrict__ query_bin,
        float delta, float vl, size_t padded_dim,
        float& r0, float& r1, float& r2, float& r3) {
    const auto* p0 = reinterpret_cast<const uint64_t*>(d0);
    const auto* p1 = reinterpret_cast<const uint64_t*>(d1);
    const auto* p2 = reinterpret_cast<const uint64_t*>(d2);
    const auto* p3 = reinterpret_cast<const uint64_t*>(d3);
    const size_t nblk = padded_dim / 64;
    uint64_t ip0 = 0, ip1 = 0, ip2 = 0, ip3 = 0;
    uint64_t pc0 = 0, pc1 = 0, pc2 = 0, pc3 = 0;
    for (size_t i = 0; i < nblk; i++) {
        uint64_t x0 = p0[i], x1 = p1[i], x2 = p2[i], x3 = p3[i];
        pc0 += __builtin_popcountll(x0);
        pc1 += __builtin_popcountll(x1);
        pc2 += __builtin_popcountll(x2);
        pc3 += __builtin_popcountll(x3);
        const uint64_t* qb = query_bin + i * 4;
        uint64_t q0 = qb[0], q1 = qb[1], q2 = qb[2], q3 = qb[3];
        ip0 += (uint64_t)__builtin_popcountll(x0 & q0);
        ip0 += (uint64_t)__builtin_popcountll(x0 & q1) << 1;
        ip0 += (uint64_t)__builtin_popcountll(x0 & q2) << 2;
        ip0 += (uint64_t)__builtin_popcountll(x0 & q3) << 3;
        ip1 += (uint64_t)__builtin_popcountll(x1 & q0);
        ip1 += (uint64_t)__builtin_popcountll(x1 & q1) << 1;
        ip1 += (uint64_t)__builtin_popcountll(x1 & q2) << 2;
        ip1 += (uint64_t)__builtin_popcountll(x1 & q3) << 3;
        ip2 += (uint64_t)__builtin_popcountll(x2 & q0);
        ip2 += (uint64_t)__builtin_popcountll(x2 & q1) << 1;
        ip2 += (uint64_t)__builtin_popcountll(x2 & q2) << 2;
        ip2 += (uint64_t)__builtin_popcountll(x2 & q3) << 3;
        ip3 += (uint64_t)__builtin_popcountll(x3 & q0);
        ip3 += (uint64_t)__builtin_popcountll(x3 & q1) << 1;
        ip3 += (uint64_t)__builtin_popcountll(x3 & q2) << 2;
        ip3 += (uint64_t)__builtin_popcountll(x3 & q3) << 3;
    }
    r0 = delta * (float)ip0 + vl * (float)pc0;
    r1 = delta * (float)ip1 + vl * (float)pc1;
    r2 = delta * (float)ip2 + vl * (float)pc2;
    r3 = delta * (float)ip3 + vl * (float)pc3;
}

// =====================================================================
// SAQNativeIndex — per-vec blob with full N-bit code + 1-bit shadow
// =====================================================================

class SAQNativeIndex {
public:
    struct SegLayout {
        size_t num_dim_pad, num_dim_copy, src_offset, padded_offset;
        size_t num_bits, code_bytes;
        // Full N-bit offsets
        size_t full_factor_offset;  // {o_l2norm, ip_cent_oa, rescale} 12 bytes
        size_t full_code_offset;
        // 1-bit shadow offsets
        size_t shadow_factor_offset; // {f_add, f_rescale, f_error} 12 bytes
        size_t shadow_code_offset;   // D/8 bytes
        size_t shadow_code_bytes;    // D/8
        // Cached constants
        float caq_delta, v_nom;
        float (*ip_func)(const float*, const uint8_t*, size_t);
        saqlib::utils::IP_FUNC_4_t ip_func_4;
    };

    SAQNativeIndex(size_t d, float avg_bits, size_t num_clusters,
                   bool enable_seg = true, int seg_eqseg = 0,
                   bool random_rotation = true)
        : d_(d), avg_bits_(avg_bits), num_clusters_(num_clusters),
          enable_seg_(enable_seg), seg_eqseg_(seg_eqseg),
          random_rotation_(random_rotation) {}

    void train_and_add(size_t n, const float* x);

    // Accessors
    size_t ntotal() const { return ntotal_; }
    size_t n_segs() const { return n_segs_; }
    size_t vec_stride() const { return vec_stride_; }
    size_t total_padded_dim() const { return total_padded_dim_; }
    size_t num_clusters() const { return num_clusters_; }
    const std::vector<SegLayout>& segs() const { return segs_; }
    const uint8_t* blob() const { return blob_.data(); }
    const float* rotated_centroids() const { return rotated_centroids_.data(); }
    const saqlib::SaqData* saq_data() const { return saq_data_.get(); }

    std::string params_string() const {
        std::ostringstream o;
        o << "bits" << avg_bits_ << "_c" << num_clusters_;
        return o.str();
    }

    // Full distance helper (N-bit path).
    static float seg_full_partial(const SegLayout& sg, const uint8_t* slot,
                                  float ip_q_code, float sum_q) {
        const float* f = (const float*)(slot + sg.full_factor_offset);
        float ext = sg.caq_delta * ip_q_code + sg.v_nom * sum_q;
        return f[0] * f[0] - 2.0f * f[2] * (ext - f[1]);
    }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    size_t d_, num_clusters_, ntotal_ = 0;
    float avg_bits_;
    bool enable_seg_, random_rotation_;
    int seg_eqseg_ = 0;
    size_t n_segs_ = 0, total_padded_dim_ = 0, vec_stride_ = 0;

    std::vector<SegLayout> segs_;
    std::vector<float> centroids_, rotated_centroids_;
    std::unique_ptr<saqlib::SaqData> saq_data_;
    std::vector<uint8_t> blob_;

    void build_seg_layout_();
    void rotate_centroids_();
};

// --- Layout: header (8) + full_factors + full_codes + shadow_factors + shadow_codes + pad ---

void SAQNativeIndex::build_seg_layout_() {
    segs_.clear();
    n_segs_ = saq_data_->base_datas.size();
    size_t src = 0, pad = 0;
    constexpr size_t kH = 8, kFF = 12, kSF = 12;

    // Full factors region
    size_t full_code_start = (kH + kFF * n_segs_ + 15) & ~size_t(15);
    size_t cursor = full_code_start;

    // First pass: full codes
    std::vector<size_t> code_cursors(n_segs_);
    for (size_t s = 0; s < n_segs_; s++) {
        auto& bd = saq_data_->base_datas[s];
        code_cursors[s] = cursor;
        size_t cb = bd.num_dim_pad * bd.num_bits / 8;
        cursor += (cb + 15) & ~size_t(15);
    }

    // Shadow factors region (right after full codes)
    size_t shadow_factor_start = cursor;

    // Shadow codes region (right after shadow factors)
    size_t shadow_code_start = (shadow_factor_start + kSF * n_segs_ + 7) & ~size_t(7);
    size_t shadow_cursor = shadow_code_start;

    for (size_t s = 0; s < n_segs_; s++) {
        auto& bd = saq_data_->base_datas[s];
        SegLayout sg{};
        sg.num_dim_pad = bd.num_dim_pad;
        sg.num_bits = bd.num_bits;
        sg.src_offset = src;
        sg.padded_offset = pad;
        sg.num_dim_copy = std::min(sg.num_dim_pad, d_ > src ? d_ - src : size_t(0));
        sg.code_bytes = sg.num_dim_pad * sg.num_bits / 8;
        sg.shadow_code_bytes = sg.num_dim_pad / 8;

        sg.full_factor_offset = kH + kFF * s;
        sg.full_code_offset = code_cursors[s];

        sg.shadow_factor_offset = shadow_factor_start + kSF * s;
        sg.shadow_code_offset = shadow_cursor;
        shadow_cursor += (sg.shadow_code_bytes + 7) & ~size_t(7);

        sg.caq_delta = sg.num_bits > 0 ? 2.0f / float(1 << sg.num_bits) : 0;
        sg.v_nom = sg.num_bits > 0 ? -1.0f + sg.caq_delta * 0.5f : 0;
        sg.ip_func = saqlib::utils::get_IP_FUNC((int)sg.num_bits);
        sg.ip_func_4 = saqlib::utils::get_IP_FUNC_4((int)sg.num_bits);

        src += sg.num_dim_pad; pad += sg.num_dim_pad;
        segs_.push_back(sg);
    }
    total_padded_dim_ = pad;
    vec_stride_ = (shadow_cursor + 63) & ~size_t(63);
}

void SAQNativeIndex::rotate_centroids_() {
    rotated_centroids_.assign(num_clusters_ * total_padded_dim_, 0);
    saqlib::FloatVec raw, rot;
    for (size_t c = 0; c < num_clusters_; c++)
        for (size_t s = 0; s < n_segs_; s++) {
            auto& sg = segs_[s]; auto& bd = saq_data_->base_datas[s];
            raw.setZero(sg.num_dim_pad);
            raw.head(sg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                centroids_.data() + c * d_ + sg.src_offset, sg.num_dim_copy);
            if (bd.rotator) rot = raw * bd.rotator->get_P(); else rot = raw;
            std::memcpy(rotated_centroids_.data() + c * total_padded_dim_ + sg.padded_offset,
                        rot.data(), sizeof(float) * sg.num_dim_pad);
        }
}

void SAQNativeIndex::train_and_add(size_t n, const float* x) {
    // k-means
    faiss::ClusteringParameters cp; cp.niter = 25; cp.seed = 1234; cp.verbose = false;
    faiss::Clustering clus(d_, num_clusters_, cp);
    faiss::IndexFlatL2 idx(d_);
    clus.train(n, x, idx);
    centroids_.assign(clus.centroids.begin(), clus.centroids.end());

    // Variance + quant plan
    saqlib::QuantizeConfig cfg;
    cfg.avg_bits = avg_bits_; cfg.enable_segmentation = enable_seg_;
    cfg.seg_eqseg = seg_eqseg_; cfg.single.random_rotation = random_rotation_;
    cfg.single.use_fastscan = true;
    saqlib::SaqDataMaker maker(cfg, d_);
    saqlib::FloatRowMat padded(n, maker.getPaddedDim()); padded.setZero();
    for (size_t i = 0; i < n; i++)
        std::memcpy(padded.row(i).data(), x + i * d_, sizeof(float) * d_);
    maker.compute_variance(padded);
    saq_data_ = maker.return_data();

    build_seg_layout_();
    rotate_centroids_();

    // Assign + encode
    ntotal_ = n;
    blob_.resize(n * vec_stride_, 0);

    faiss::IndexFlatL2 quantizer(d_);
    quantizer.add(num_clusters_, centroids_.data());
    std::vector<float> dists(n); std::vector<faiss::idx_t> labels(n);
    quantizer.search(n, x, 1, dists.data(), labels.data());

#pragma omp parallel
    {
        std::vector<std::unique_ptr<saqlib::CAQEncoder>> encs;
        for (size_t s = 0; s < n_segs_; s++) {
            auto& bd = saq_data_->base_datas[s];
            encs.emplace_back(std::make_unique<saqlib::CAQEncoder>(
                bd.num_dim_pad, bd.num_bits, bd.cfg));
        }
        std::vector<uint16_t> u16buf(total_padded_dim_);
        std::vector<uint8_t> bin_raw;
        saqlib::FloatVec raw_seg, rot_vec;

#pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; i++) {
            auto cid = labels[i];
            uint8_t* slot = blob_.data() + i * vec_stride_;
            *(uint32_t*)slot = (uint32_t)cid;

            for (size_t s = 0; s < n_segs_; s++) {
                auto& sg = segs_[s]; auto& bd = saq_data_->base_datas[s];
                const size_t D = sg.num_dim_pad;

                raw_seg.setZero(D);
                raw_seg.head(sg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                    x + i * d_ + sg.src_offset, sg.num_dim_copy);
                if (bd.rotator) rot_vec = raw_seg * bd.rotator->get_P();
                else rot_vec = raw_seg;

                const float* rc = rotated_centroids_.data() +
                    (size_t)cid * total_padded_dim_ + sg.padded_offset;
                Eigen::Map<const saqlib::FloatVec> rc_vec(rc, D);
                saqlib::FloatVec residual = rot_vec - rc_vec;

                // ---- Full N-bit encoding ----
                saqlib::QuantBaseCode bc;
                saqlib::FloatVec rc_copy = rc_vec;
                encs[s]->encode_and_fac(residual, bc, &rc_copy);

                float* ff = (float*)(slot + sg.full_factor_offset);
                ff[0] = bc.o_l2norm;
                ff[1] = bc.ip_cent_oa;
                ff[2] = bc.fac_rescale;

                if (sg.num_bits > 0 && bc.code.size() > 0) {
                    for (size_t j = 0; j < D; j++)
                        u16buf[j] = (uint16_t)bc.code[j];
                    saqlib::utils::get_compacted_code16_func((int)sg.num_bits)(
                        slot + sg.full_code_offset, u16buf.data(), D);
                }

                // ---- 1-bit shadow: sign(residual) + factors ----
                bin_raw.resize(D);
                float l2_sqr = 0, ip_r_xcb = 0, ip_c_xcb = 0;
                for (size_t j = 0; j < D; j++) {
                    float r = residual[j];
                    l2_sqr += r * r;
                    uint8_t b = (r >= 0) ? 1 : 0;
                    bin_raw[j] = b;
                    float xcb = b - 0.5f; // xu_cb ∈ {-0.5, +0.5}
                    ip_r_xcb += r * xcb;
                    ip_c_xcb += rc[j] * xcb;
                }
                float l2_norm = std::sqrt(l2_sqr);
                float xcb_l2sqr = D * 0.25f; // all entries ±0.5

                if (ip_r_xcb == 0.0f)
                    ip_r_xcb = std::numeric_limits<float>::infinity();

                float f_add = l2_sqr + 2.0f * l2_sqr * ip_c_xcb / ip_r_xcb;
                float f_rescale = -2.0f * l2_sqr / ip_r_xcb;
                float ratio = (l2_sqr * xcb_l2sqr) / (ip_r_xcb * ip_r_xcb);
                // Raw error factor (epsilon is applied at search time).
                float f_error = 2.0f * l2_norm *
                    std::sqrt(std::max(0.0f, ratio - 1.0f) / (D - 1));

                float* sf = (float*)(slot + sg.shadow_factor_offset);
                sf[0] = f_add;
                sf[1] = f_rescale;
                sf[2] = f_error;

                // Pack 1-bit code (LSB-first, compatible with CodeHelper<1>)
                saqlib::utils::CodeHelper<1>::compacted_code8(
                    slot + sg.shadow_code_offset, bin_raw.data(), D);
            }
        }
    }
}

bool SAQNativeIndex::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    uint64_t magic = 0x5341514E41544956ULL, ver = 3; // v3: epsilon separated from f_error
    os.write((char*)&magic, 8); os.write((char*)&ver, 8);
    auto w = [&](uint64_t v){ os.write((char*)&v, 8); };
    w(d_); w(num_clusters_);
    os.write((char*)&avg_bits_, 4);
    uint8_t fl = (enable_seg_?1:0)|(random_rotation_?2:0);
    os.write((char*)&fl, 1);
    w(seg_eqseg_); w(ntotal_); w(vec_stride_);
    w(centroids_.size());
    os.write((char*)centroids_.data(), centroids_.size()*4);
    saq_data_->save(os);
    w(blob_.size());
    os.write((char*)blob_.data(), blob_.size());
    os.close(); return os.good();
}

bool SAQNativeIndex::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    uint64_t magic, ver;
    is.read((char*)&magic, 8); is.read((char*)&ver, 8);
    if (magic != 0x5341514E41544956ULL || ver != 3) return false;
    auto r = [&]{ uint64_t v; is.read((char*)&v, 8); return v; };
    if (r() != d_ || r() != num_clusters_) return false;
    float ab; is.read((char*)&ab, 4);
    if (std::fabs(ab - avg_bits_) > 1e-6) return false;
    uint8_t fl; is.read((char*)&fl, 1);
    seg_eqseg_ = (int)r(); ntotal_ = r();
    size_t vs = r();
    size_t cs = r();
    centroids_.resize(cs); is.read((char*)centroids_.data(), cs*4);
    saq_data_ = std::make_unique<saqlib::SaqData>(); saq_data_->load(is);
    build_seg_layout_(); rotate_centroids_();
    if (vs != vec_stride_) return false;
    size_t bs = r();
    blob_.resize(bs); is.read((char*)blob_.data(), bs);
    return is.good();
}

// =====================================================================
// Two-stage HNSW search
// =====================================================================

struct QStats { size_t nhops = 0, ndis1 = 0, ndis2 = 0; };
struct SR { float est, low; faiss::idx_t id;
    bool operator<(const SR& o) const { return est < o.est; }
    bool operator>(const SR& o) const { return est > o.est; }
};
using MinH = std::priority_queue<SR, std::vector<SR>, std::greater<SR>>;
using MaxH = std::priority_queue<SR>;

// Per-query state.
struct QS {
    const SAQNativeIndex* idx;
    std::vector<saqlib::FloatVec> q_rot;
    std::vector<float> sum_q;
    // Popcount query bit-planes (per seg, 4 planes per 64-dim block, LSB-first)
    std::vector<std::vector<uint64_t>> q_bin;  // [seg][block*4 + plane]
    std::vector<float> q_delta, q_vl;           // per-seg scalar-quant params
    // Per-cluster cache
    uint32_t qgen = 0;
    std::vector<uint32_t> cgen;
    std::vector<float> qc_sqr, qc_nrm;
    std::vector<float> qc_seg_sqr, qc_seg_nrm;

    void set_query(const float* x, const SAQNativeIndex* p) {
        idx = p;
        size_t ns = p->n_segs();
        q_rot.resize(ns); sum_q.resize(ns);
        q_bin.resize(ns); q_delta.resize(ns); q_vl.resize(ns);
        for (size_t s = 0; s < ns; s++) {
            auto& sg = p->segs()[s];
            auto& bd = p->saq_data()->base_datas[s];
            const size_t D = sg.num_dim_pad;

            saqlib::FloatVec raw = saqlib::FloatVec::Zero(D);
            raw.head(sg.num_dim_copy) = Eigen::Map<const saqlib::FloatVec>(
                x + sg.src_offset, sg.num_dim_copy);
            if (bd.rotator) q_rot[s] = raw * bd.rotator->get_P();
            else q_rot[s] = raw;
            sum_q[s] = q_rot[s].sum();

            // Inline 4-bit min/max scalar quantization of rotated query.
            const float* qr = q_rot[s].data();
            float qmin = qr[0], qmax = qr[0];
            for (size_t j = 1; j < D; j++) {
                if (qr[j] < qmin) qmin = qr[j];
                if (qr[j] > qmax) qmax = qr[j];
            }
            float range = qmax - qmin;
            float delta = range > 0 ? range / 15.0f : 1.0f;
            float inv_d = range > 0 ? 15.0f / range : 0.0f;
            q_delta[s] = delta;
            q_vl[s] = qmin;

            // Transpose into 4 bit-planes (LSB-first: bit d of plane j
            // = bit j of quant_query[d]).  Scalar loop, ~4*D ops, amortized.
            const size_t nblk = D / 64;
            q_bin[s].assign(nblk * 4, 0);
            for (size_t b = 0; b < nblk; b++) {
                uint64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                for (size_t d = 0; d < 64; d++) {
                    float v = (qr[b * 64 + d] - qmin) * inv_d + 0.5f;
                    int iv = (int)v;
                    if (iv < 0) iv = 0; if (iv > 15) iv = 15;
                    uint64_t bit = 1ULL << d;
                    if (iv & 1) p0 |= bit;
                    if (iv & 2) p1 |= bit;
                    if (iv & 4) p2 |= bit;
                    if (iv & 8) p3 |= bit;
                }
                q_bin[s][b * 4 + 0] = p0;
                q_bin[s][b * 4 + 1] = p1;
                q_bin[s][b * 4 + 2] = p2;
                q_bin[s][b * 4 + 3] = p3;
            }
        }
        size_t nc = p->num_clusters();
        if (cgen.size() != nc) {
            cgen.assign(nc, 0);
            qc_sqr.assign(nc, 0); qc_nrm.assign(nc, 0);
            qc_seg_sqr.assign(nc * ns, 0); qc_seg_nrm.assign(nc * ns, 0);
        }
        ++qgen; if (qgen == 0) { std::fill(cgen.begin(), cgen.end(), 0); qgen = 1; }
    }

    void ensure_cluster(uint32_t cid) {
        if (cgen[cid] == qgen) return;
        size_t ns = idx->n_segs();
        float total = 0;
        for (size_t s = 0; s < ns; s++) {
            auto& sg = idx->segs()[s];
            const float* c = idx->rotated_centroids() +
                (size_t)cid * idx->total_padded_dim() + sg.padded_offset;
            const float* q = q_rot[s].data();
            __m512 av = _mm512_setzero_ps();
            for (size_t j = 0; j < sg.num_dim_pad; j += 16) {
                __m512 d = _mm512_sub_ps(_mm512_loadu_ps(q+j), _mm512_loadu_ps(c+j));
                av = _mm512_fmadd_ps(d, d, av);
            }
            float seg_sqr = _mm512_reduce_add_ps(av);
            qc_seg_sqr[cid * ns + s] = seg_sqr;
            qc_seg_nrm[cid * ns + s] = std::sqrt(seg_sqr);
            total += seg_sqr;
        }
        qc_sqr[cid] = total; qc_nrm[cid] = std::sqrt(total);
        cgen[cid] = qgen;
    }

    // Cheap 1-bit estimate using popcount IP (much faster than CodeHelper<1>).
    // est = Σ_s [f_add_s + g_add_s + f_rescale_s * (ip_1bit_s + k1xsumq_s)]
    // low = est - Σ_s [f_error_s * g_error_s]
    void cheap_dist(faiss::idx_t id, float& est, float& low) {
        const uint8_t* sl = idx->blob() + (size_t)id * idx->vec_stride();
        uint32_t cid = *(const uint32_t*)sl;
        ensure_cluster(cid);
        size_t ns = idx->n_segs();
        est = 0; float err_sum = 0;
        for (size_t s = 0; s < ns; s++) {
            auto& sg = idx->segs()[s];
            float g_add = qc_seg_sqr[cid * ns + s];
            float g_err = qc_seg_nrm[cid * ns + s];

            const float* sf = (const float*)(sl + sg.shadow_factor_offset);
            float ip_1bit = popcount_ip_1bit(
                sl + sg.shadow_code_offset,
                q_bin[s].data(), q_delta[s], q_vl[s], sg.num_dim_pad);
            float k1xsumq = sum_q[s] * (-0.5f);
            est += sf[0] + g_add + sf[1] * (ip_1bit + k1xsumq);
            err_sum += sf[2] * g_err;  // sf[2] = raw f_error (no epsilon)
        }
        low = est - kEpsilon * err_sum;
    }

    // Batch-4 cheap 1-bit estimates.
    void cheap_dist_4(const faiss::idx_t* ids,
                      float* est, float* low) {
        const uint8_t* base = idx->blob();
        size_t stride = idx->vec_stride();
        const uint8_t* sl0 = base + (size_t)ids[0] * stride;
        const uint8_t* sl1 = base + (size_t)ids[1] * stride;
        const uint8_t* sl2 = base + (size_t)ids[2] * stride;
        const uint8_t* sl3 = base + (size_t)ids[3] * stride;
        uint32_t c0 = *(const uint32_t*)sl0;
        uint32_t c1 = *(const uint32_t*)sl1;
        uint32_t c2 = *(const uint32_t*)sl2;
        uint32_t c3 = *(const uint32_t*)sl3;
        ensure_cluster(c0); ensure_cluster(c1);
        ensure_cluster(c2); ensure_cluster(c3);

        size_t ns = idx->n_segs();
        float e0 = 0, e1 = 0, e2 = 0, e3 = 0;
        float er0 = 0, er1 = 0, er2 = 0, er3 = 0;
        for (size_t s = 0; s < ns; s++) {
            auto& sg = idx->segs()[s];
            float ga0 = qc_seg_sqr[c0*ns+s], ge0 = qc_seg_nrm[c0*ns+s];
            float ga1 = qc_seg_sqr[c1*ns+s], ge1 = qc_seg_nrm[c1*ns+s];
            float ga2 = qc_seg_sqr[c2*ns+s], ge2 = qc_seg_nrm[c2*ns+s];
            float ga3 = qc_seg_sqr[c3*ns+s], ge3 = qc_seg_nrm[c3*ns+s];

            float ip0, ip1, ip2, ip3;
            popcount_ip_1bit_4(
                sl0 + sg.shadow_code_offset, sl1 + sg.shadow_code_offset,
                sl2 + sg.shadow_code_offset, sl3 + sg.shadow_code_offset,
                q_bin[s].data(), q_delta[s], q_vl[s], sg.num_dim_pad,
                ip0, ip1, ip2, ip3);
            float kq = sum_q[s] * (-0.5f);

            const float* sf0 = (const float*)(sl0 + sg.shadow_factor_offset);
            const float* sf1 = (const float*)(sl1 + sg.shadow_factor_offset);
            const float* sf2 = (const float*)(sl2 + sg.shadow_factor_offset);
            const float* sf3 = (const float*)(sl3 + sg.shadow_factor_offset);
            e0 += sf0[0] + ga0 + sf0[1] * (ip0 + kq);
            e1 += sf1[0] + ga1 + sf1[1] * (ip1 + kq);
            e2 += sf2[0] + ga2 + sf2[1] * (ip2 + kq);
            e3 += sf3[0] + ga3 + sf3[1] * (ip3 + kq);
            er0 += sf0[2] * ge0; er1 += sf1[2] * ge1;
            er2 += sf2[2] * ge2; er3 += sf3[2] * ge3;
        }
        est[0] = e0; est[1] = e1; est[2] = e2; est[3] = e3;
        low[0] = e0 - kEpsilon*er0; low[1] = e1 - kEpsilon*er1;
        low[2] = e2 - kEpsilon*er2; low[3] = e3 - kEpsilon*er3;
    }

    // Full N-bit distance.
    float full_dist(faiss::idx_t id) {
        const uint8_t* sl = idx->blob() + (size_t)id * idx->vec_stride();
        uint32_t cid = *(const uint32_t*)sl;
        ensure_cluster(cid);
        float d = qc_sqr[cid];
        for (size_t s = 0; s < idx->n_segs(); s++) {
            auto& sg = idx->segs()[s];
            if (sg.num_bits == 0) {
                float xn = *(const float*)(sl + sg.full_factor_offset);
                d += xn * xn; continue;
            }
            float ip = sg.ip_func(q_rot[s].data(), sl + sg.full_code_offset, sg.num_dim_pad);
            d += SAQNativeIndex::seg_full_partial(sg, sl, ip, sum_q[s]);
        }
        return std::max(0.0f, d);
    }

    // Batch-4 full N-bit distances.
    void full_dist_4(const faiss::idx_t* ids, float* out) {
        const uint8_t* base = idx->blob();
        size_t stride = idx->vec_stride();
        const uint8_t* sl0 = base + (size_t)ids[0] * stride;
        const uint8_t* sl1 = base + (size_t)ids[1] * stride;
        const uint8_t* sl2 = base + (size_t)ids[2] * stride;
        const uint8_t* sl3 = base + (size_t)ids[3] * stride;
        uint32_t c0 = *(const uint32_t*)sl0;
        uint32_t c1 = *(const uint32_t*)sl1;
        uint32_t c2 = *(const uint32_t*)sl2;
        uint32_t c3 = *(const uint32_t*)sl3;
        ensure_cluster(c0); ensure_cluster(c1);
        ensure_cluster(c2); ensure_cluster(c3);

        float d0 = qc_sqr[c0], d1 = qc_sqr[c1];
        float d2 = qc_sqr[c2], d3 = qc_sqr[c3];
        for (size_t s = 0; s < idx->n_segs(); s++) {
            auto& sg = idx->segs()[s];
            if (sg.num_bits == 0) {
                float xn0 = *(const float*)(sl0 + sg.full_factor_offset);
                float xn1 = *(const float*)(sl1 + sg.full_factor_offset);
                float xn2 = *(const float*)(sl2 + sg.full_factor_offset);
                float xn3 = *(const float*)(sl3 + sg.full_factor_offset);
                d0 += xn0*xn0; d1 += xn1*xn1;
                d2 += xn2*xn2; d3 += xn3*xn3;
                continue;
            }
            float ip0, ip1, ip2, ip3;
            sg.ip_func_4(
                q_rot[s].data(),
                sl0 + sg.full_code_offset, sl1 + sg.full_code_offset,
                sl2 + sg.full_code_offset, sl3 + sg.full_code_offset,
                sg.num_dim_pad, ip0, ip1, ip2, ip3);
            d0 += SAQNativeIndex::seg_full_partial(sg, sl0, ip0, sum_q[s]);
            d1 += SAQNativeIndex::seg_full_partial(sg, sl1, ip1, sum_q[s]);
            d2 += SAQNativeIndex::seg_full_partial(sg, sl2, ip2, sum_q[s]);
            d3 += SAQNativeIndex::seg_full_partial(sg, sl3, ip3, sum_q[s]);
        }
        out[0] = std::max(0.0f, d0); out[1] = std::max(0.0f, d1);
        out[2] = std::max(0.0f, d2); out[3] = std::max(0.0f, d3);
    }
};

QStats saq_native_search(
        const faiss::HNSW& hnsw, const faiss::IndexHNSW* hi,
        faiss::VisitedTable& vt, QS& qs,
        const float* query, const SAQNativeIndex* saq,
        size_t k, size_t ef,
        float* out_d, faiss::idx_t* out_l,
        bool rerank, faiss::DistanceComputer* exact_dc) {

    QStats st;
    qs.set_query(query, saq);
    auto ep = hnsw.entry_point;
    if (ep < 0) {
        for (size_t i = 0; i < k; i++) { out_d[i] = 1e30f; out_l[i] = -1; }
        return st;
    }

    // Entry point: full distance.
    float ep_est = qs.full_dist(ep); st.ndis2++;
    float ep_low = ep_est;

    // Upper levels: greedy with full distances (few hops, accuracy matters
    // more than speed here — a wrong entry point at level 0 can't be recovered).
    for (int lev = hnsw.max_level; lev > 0; lev--) {
        bool chg = true;
        while (chg) {
            chg = false;
            size_t b, e; hnsw.neighbor_range(ep, lev, &b, &e);
            for (size_t j = b; j < e; j++) {
                auto nb = hnsw.neighbors[j]; if (nb < 0) break;
                float ne = qs.full_dist(nb); st.ndis2++;
                if (ne < ep_est) { ep = nb; ep_est = ne; ep_low = ne; chg = true; }
            }
            st.nhops++;
        }
    }

    // Level 0: two-stage bounded search.
    MinH cands; MaxH topk;
    cands.push({ep_est, ep_low, ep});
    topk.push({ep_est, ep_low, ep});
    vt.set(ep);
    // Start with infinite bound until topk fills to ef; using ep_est here
    // would prematurely block good candidates near the entry point.
    float bound = std::numeric_limits<float>::max();

    static constexpr size_t kMN = 256;
    faiss::HNSW::storage_idx_t nids[kMN];
    float nest[kMN], nlow[kMN];
    bool nref[kMN]; size_t rpos[kMN];

    while (!cands.empty()) {
        auto cur = cands.top(); cands.pop();
        if (cur.est > bound && topk.size() >= ef) break;

        size_t b, e; hnsw.neighbor_range(cur.id, 0, &b, &e);
        size_t nv = 0;
        for (size_t j = b; j < e; j++) {
            auto nb = hnsw.neighbors[j]; if (nb < 0) break;
            if (vt.get(nb)) continue; vt.set(nb);
            st.nhops++; nids[nv++] = nb;
        }

        // Phase 2: cheap 1-bit estimates (batch-4 + scalar tail).
        {
            size_t i = 0;
            for (; i + 4 <= nv; i += 4) {
                faiss::idx_t b4[4] = {nids[i], nids[i+1], nids[i+2], nids[i+3]};
                qs.cheap_dist_4(b4, nest + i, nlow + i);
                st.ndis1 += 4;
            }
            for (; i < nv; i++) {
                qs.cheap_dist(nids[i], nest[i], nlow[i]);
                st.ndis1++;
            }
        }

        // Phase 3: decide who needs full refinement.
        size_t nr = 0;
        for (size_t i = 0; i < nv; i++) {
            nref[i] = (topk.size() < ef) || (nlow[i] < bound);
            if (nref[i]) rpos[nr++] = i;
        }

        // Phase 4: full N-bit distances for survivors (batch-4 + scalar tail).
        {
            size_t i = 0;
            for (; i + 4 <= nr; i += 4) {
                faiss::idx_t batch_ids[4] = {nids[rpos[i]], nids[rpos[i+1]],
                                              nids[rpos[i+2]], nids[rpos[i+3]]};
                float batch_d[4];
                qs.full_dist_4(batch_ids, batch_d);
                st.ndis2 += 4;
                for (int j = 0; j < 4; j++) {
                    nest[rpos[i+j]] = batch_d[j]; nlow[rpos[i+j]] = batch_d[j];
                }
            }
            for (; i < nr; i++) {
                size_t p = rpos[i];
                float d = qs.full_dist(nids[p]); st.ndis2++;
                nest[p] = d; nlow[p] = d;
            }
        }

        // Phase 5: update candidates and topk.
        // All neighbours with est < bound enter the candidates queue (so
        // unrefined neighbours can still be explored). Only refined
        // neighbours enter topk (with their accurate full distance).
        for (size_t i = 0; i < nv; i++) {
            if (cands.size() < ef || nest[i] < bound) {
                cands.push({nest[i], nlow[i], nids[i]});
            }
            if (nref[i]) {
                topk.push({nest[i], nlow[i], nids[i]});
                if (topk.size() > ef) topk.pop();
                if (topk.size() >= ef) bound = topk.top().est;
            }
        }
    }
    vt.advance();

    // Collect results.
    std::vector<SR> res;
    while (!topk.empty()) { res.push_back(topk.top()); topk.pop(); }
    std::reverse(res.begin(), res.end());

    if (rerank && exact_dc) {
        exact_dc->set_query(query);
        std::vector<std::pair<float, faiss::idx_t>> rr;
        for (auto& r : res) rr.push_back({(*exact_dc)(r.id), r.id});
        std::sort(rr.begin(), rr.end());
        for (size_t i = 0; i < k && i < rr.size(); i++) { out_d[i] = rr[i].first; out_l[i] = rr[i].second; }
    } else {
        for (size_t i = 0; i < k && i < res.size(); i++) { out_d[i] = res[i].est; out_l[i] = res[i].id; }
    }
    for (size_t i = res.size(); i < k; i++) { out_d[i] = 1e30f; out_l[i] = -1; }
    return st;
}

// =====================================================================
// Benchmark runner + main
// =====================================================================

struct SS {
    PercentileStats c, f, h;
    static SS compute(const std::vector<QStats>& v) {
        SS r; if (v.empty()) return r;
        std::vector<size_t> vc, vf, vh;
        for (auto& q : v) { vc.push_back(q.ndis1); vf.push_back(q.ndis2); vh.push_back(q.nhops); }
        r.c = compute_percentile_stats(vc); r.f = compute_percentile_stats(vf);
        r.h = compute_percentile_stats(vh); return r;
    }
};
struct BR { size_t ef; double qps; float recall; double lat; SS ss; };

std::vector<BR> run_bench(
        const faiss::IndexHNSW* hi, const faiss::IndexFlat* fi,
        const SAQNativeIndex* saq,
        const float* xq, size_t nq, size_t d, const int* gt, size_t gtk, size_t k,
        const std::vector<size_t>& efs, int nth, bool rerank) {
    std::vector<BR> out; Timer t;
    for (size_t ef : efs) {
        std::vector<faiss::idx_t> ids(nq*k, -1);
        std::vector<float> dists(nq*k, 1e30f);
        std::vector<QStats> qs(nq);
        t.reset();
#pragma omp parallel num_threads(nth)
        {
            std::unique_ptr<faiss::DistanceComputer> edc;
            if (rerank) edc.reset(fi->get_distance_computer());
            faiss::VisitedTable vt(hi->ntotal); QS state;
#pragma omp for schedule(dynamic,1)
            for (int64_t i = 0; i < (int64_t)nq; i++)
                qs[i] = saq_native_search(hi->hnsw, hi, vt, state, xq+i*d, saq,
                    k, ef, dists.data()+i*k, ids.data()+i*k, rerank, edc.get());
        }
        double ms = t.elapsed_ms();
        float rec = compute_recall_at_k(nq, k, ids.data(), gt, gtk);
        SS ss = SS::compute(qs);
        out.push_back({ef, nq*1000.0/ms, rec, ms/nq, ss});
        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(0) << out.back().qps
                  << std::setw(12) << std::setprecision(4) << rec
                  << std::setw(12) << std::setprecision(3) << out.back().lat
                  << std::setw(14) << std::setprecision(1) << ss.c.mean
                  << std::setw(14) << ss.f.mean
                  << std::setw(12) << ss.h.mean << "\n";
    }
    return out;
}

int main(int argc, char** argv) {
    std::string dataset, config_dir = "./config", data_path;
    int threads = 16; bool rerank = false; std::vector<size_t> efs;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dataset" && i+1<argc) dataset = argv[++i];
        else if (a == "--config-dir" && i+1<argc) config_dir = argv[++i];
        else if (a == "--data-path" && i+1<argc) data_path = argv[++i];
        else if (a == "--threads" && i+1<argc) threads = std::stoi(argv[++i]);
        else if (a == "--rerank") rerank = true;
        else if (a == "--ef" && i+1<argc) {
            std::istringstream is(argv[++i]); std::string t;
            while (std::getline(is,t,',')) efs.push_back(std::stoul(t));
        }
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " --dataset <name> [--data-path <p>] [--threads <n>]"
                      << " [--ef <list>] [--rerank]\n";
            return 0;
        }
    }
    if (dataset.empty()) { std::cerr << "--dataset required\n"; return 1; }

    std::cout << "==================================================\n"
              << "HNSW + SAQ Native Search (1-bit pruning)\n"
              << "Dataset: " << dataset
              << "\n==================================================\n";
    Timer gt;

    auto dm = ConfigParser::parse_datasets(config_dir+"/datasets.conf");
    DatasetConfig dc;
    if (auto it = dm.find(dataset); it != dm.end()) dc = it->second;
    else { dc.name = dataset; dc.base_path = "/data/local/embedding_dataset/"+dataset;
           dc.ef_search_values = get_default_ef_search(); }
    if (!data_path.empty()) dc.base_path = data_path;
    if (threads > 0) dc.threads = threads;
    if (!efs.empty()) dc.ef_search_values = efs;
    if (dc.ef_search_values.empty()) dc.ef_search_values = get_default_ef_search();

    auto am = ConfigParser::parse_algorithm(config_dir+"/saq.conf");
    std::vector<AlgorithmParamSet> ps;
    if (auto it = am.find(dataset); it != am.end()) ps = it->second;
    else { std::cerr << "No [" << dataset << "] in saq.conf\n"; return 1; }

    omp_set_num_threads(dc.threads);
    std::cout << "Path: " << dc.base_path << "\nThreads: " << dc.threads << "\n\n[Loading...]\n";

    size_t d, nb, nq, gtk;
    float* xb = fvecs_read(dc.get_path(dc.base_file).c_str(), &d, &nb);
    float* xq = fvecs_read(dc.get_path(dc.query_file).c_str(), &d, &nq);
    int* gtt = ivecs_read(dc.get_path(dc.groundtruth_file).c_str(), &gtk, &nq);
    if (!xb||!xq||!gtt) { std::cerr << "Data load failed\n"; return 1; }
    std::cout << "DB: " << nb << "x" << d << "  Q: " << nq << "\n";

    auto* hi = load_or_build_hnsw(dc.get_hnsw_index_path(), d, dc.hnsw_M, dc.hnsw_efConstruction, nb, xb);
    if (!hi) return 1;
    auto* fi = dynamic_cast<faiss::IndexFlat*>(hi->storage);

    for (auto& p : ps) {
        float bits = 4; size_t clusters = 4096;
        if (auto it = p.params.find("bits"); it != p.params.end()) bits = std::stof(it->second);
        if (auto it = p.params.find("clusters"); it != p.params.end()) clusters = std::stoul(it->second);

        SAQNativeIndex saq(d, bits, clusters);
        Timer bt;
        // Always retrain: SaqData::load has a CHECK bug for padded dims,
        // and we need deterministic rotators for epsilon A/B comparisons.
        std::srand(42);  // Seed Eigen::Random (uses std::rand) for reproducible rotators.
        std::cout << "\n[Training (" << saq.params_string() << ")...]\n";
        saq.train_and_add(nb, xb);
        std::cout << "Time: " << bt.elapsed_ms() << " ms\n\n"
                  << std::setw(8) << "ef" << std::setw(12) << "QPS"
                  << std::setw(12) << ("R@"+std::to_string(dc.k))
                  << std::setw(12) << "Lat" << std::setw(14) << "ndis1"
                  << std::setw(14) << "ndis2" << std::setw(12) << "nhops\n"
                  << std::string(72,'-') << "\n";

        auto res = run_bench(hi, fi, &saq, xq, nq, d, gtt, gtk, dc.k,
                             dc.ef_search_values, dc.threads, rerank);

        std::string rd = "experiments/hnsw/results/"+dataset+"/saq_native";
        create_directory(rd);
        std::string rf = rd+"/saq_native_"+saq.params_string()+
            "_t"+std::to_string(dc.threads)+"_M"+std::to_string(dc.hnsw_M)+
            "_efc"+std::to_string(dc.hnsw_efConstruction)+".txt";
        std::ofstream os(rf);
        if (os) {
            os << "# SAQ Native | " << dataset << " | " << saq.params_string() << "\n"
               << std::setw(8)<<"ef"<<std::setw(12)<<"QPS"<<std::setw(12)<<"Recall"
               <<std::setw(12)<<"Lat"<<std::setw(14)<<"ndis1"<<std::setw(14)<<"ndis2"
               <<std::setw(12)<<"nhops\n";
            for (auto& r : res)
                os<<std::setw(8)<<r.ef<<std::setw(12)<<std::fixed<<std::setprecision(0)<<r.qps
                  <<std::setw(12)<<std::setprecision(4)<<r.recall<<std::setw(12)<<std::setprecision(3)
                  <<r.lat<<std::setw(14)<<std::setprecision(1)<<r.ss.c.mean
                  <<std::setw(14)<<r.ss.f.mean<<std::setw(12)<<r.ss.h.mean<<"\n";
            std::cout << "Saved: " << rf << "\n";
        }
    }
    delete hi; delete[] xb; delete[] xq; delete[] gtt;
    std::cout << "\nTotal: " << std::fixed << std::setprecision(1) << gt.elapsed_ms()/1000.0 << "s\n";
}
