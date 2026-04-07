#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdlib.h>
#include <vector>

#include <glog/logging.h>

#include "defines.hpp"
#include "quantization/fastscan/fastscan.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"

namespace saqlib {

class QuantBaseCode {
  public:
    Eigen::VectorXi code;
    float o_l2norm;
    float ip_cent_oa = 0;
    float fac_rescale;      // rescale factor for estimation
    float fac_error = 0;    // error factor for estimation
    float norm_ip_o_oa = 0; // <o, o_a> / |o| / |o_a|. Only used for quant metrics
};

struct ExFactor {
    float rescale = 0;
    float error = 0;
};

class SaqCluData;

class CAQClusterData {
    friend SaqCluData;

  public:
    static constexpr size_t kNumShortFactors = 2; // factors packed into shortdata

    size_t num_vec_;              // Num of vectors in this cluster
    size_t num_vec_align_;        // Padded number of vectors (multiple of 32)
    const size_t num_dim_padded_; // Padded number of dimension (multiple of 64)
    const size_t num_bits_;       // bits
    size_t num_blocks_;           // Num of blocks
  private:
    size_t shortb_factors_num_; // number of short block factors (in float)
    size_t shortb_code_bytes_;  // bytes of short block code
    size_t longb_code_bytes_;   // bytes of long block code

    size_t num_parallel_clusters_ = 1; // number of parallel clusters, that is, segments
    bool use_fastscan_ = true;

    bool should_free_ = false;
    float *short_factors_ = nullptr;   // short factors
    uint8_t *short_code_ = nullptr;    // short code
    uint8_t *long_code_ = nullptr;     // long code
    ExFactor *long_factors_ = nullptr; // long factors of vectors
    PID *ids_ = nullptr;               // PID of vectors
    FloatVec centroid_;                // Rotated centroid of clusters

  public:
    /**
     * @brief Construct a new Cluster:: Cluster object
     * Data in the cluster are mapped to large arrays in memory
     *
     * @param num number of vectors
     * @param short_data blocks of 1-bit codes and corresponding factors
     * @param long_code long code for re-ranking
     * @param ex_factor factors for re-ranking
     * @param ids id for vectors in the cluster
     */
    explicit CAQClusterData(size_t num_vec, size_t num_dim_paded, size_t num_bits)
        : num_vec_(num_vec),
          num_vec_align_(utils::rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_dim_padded_(num_dim_paded),
          num_bits_(num_bits),
          num_blocks_(utils::div_rd_up(num_vec, KFastScanSize)),
          shortb_factors_num_(KFastScanSize * kNumShortFactors),
          shortb_code_bytes_(num_bits ? num_dim_paded * KFastScanSize / 8 * sizeof(uint8_t) : 0),
          longb_code_bytes_(num_bits ? num_dim_paded * (num_bits - 1) / 8 : 0) {
        centroid_.resize(num_dim_paded);
    }

    void set_num_vec(size_t num_vec) {
        num_vec_ = num_vec;
        num_vec_align_ = utils::rd_up_to_multiple_of(num_vec, KFastScanSize);
        num_blocks_ = utils::div_rd_up(num_vec, KFastScanSize);
    }

    ~CAQClusterData() {
        if (should_free_) {
            std::free(short_factors_);
            std::free(short_code_);
            std::free(long_code_);
            std::free(long_factors_);
            std::free(ids_);
        }
    }

    // void allocate_data()
    // {
    //     should_free_ = true;
    //     short_factors_ = memory::align_mm<64, float>(shortb_factors_fcnt_ * num_blocks_);
    //     short_code_ = memory::align_mm<64, uint8_t>(shortb_code_bytes_ * num_blocks_);
    //     long_code_ = memory::align_mm<64, uint8_t>(longb_code_bytes_ * num_vec_align);
    //     EX_FACTOR = memory::align_mm<64, ExFactor>(num_vec_);
    //     IDs_ = memory::align_mm<64, PID>(num_vec_align);
    // }

    /**
     * @brief Return pointer to short code of i-th blocks in this cluster
     */
    auto short_code(size_t block_idx) { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code(size_t block_idx) const { return &short_code_[shortb_code_bytes_ * block_idx]; }
    auto short_code_single(size_t vec_idx) const {
        auto block_idx = vec_idx / KFastScanSize;
        auto j = vec_idx % KFastScanSize;
        return short_code(block_idx) + num_dim_padded_ / 8 * j;
    }

    auto factor_o_l2norm(size_t block_idx) { return &short_factors_[block_idx * shortb_factors_num_]; }
    auto factor_o_l2norm(size_t block_idx) const { return &short_factors_[block_idx * shortb_factors_num_]; }

    // ip_cent_oa is optional
    auto factor_ip_cent_oa(size_t block_idx) { return factor_o_l2norm(block_idx) + KFastScanSize; }
    auto factor_ip_cent_oa(size_t block_idx) const { return factor_o_l2norm(block_idx) + KFastScanSize; }

    /**
     * @brief Return long code for i-th vector in this cluster
     */
    uint8_t *long_code(size_t vec_idx) {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }
    uint8_t *long_code(size_t vec_idx) const {
        DCHECK_LT(vec_idx, num_vec_);
        return &long_code_[vec_idx * longb_code_bytes_];
    }

    /**
     * @brief Return long factor of i-th vector in this cluster
     */
    ExFactor &long_factor(size_t vec_idx) {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }
    ExFactor &long_factor(size_t vec_idx) const {
        return long_factors_[vec_idx * num_parallel_clusters_];
    }

    auto &centroid() { return centroid_; }
    auto &centroid() const { return centroid_; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_; }
    PID *ids() const { return this->ids_; }

    auto num_vec() const { return num_vec_; }
    auto num_blocks() const { return num_blocks_; }
    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }
    auto short_code_byte_num() const { return num_dim_padded_ / 8; }
    auto raw_short_factors_num() const { return KFastScanSize * kNumShortFactors; }
    auto raw_short_code_bytes() const {
        return num_bits_ ? num_dim_padded_ * KFastScanSize / 8 * sizeof(uint8_t) : 0;
    }
    auto raw_long_code_bytes() const {
        return num_bits_ ? num_dim_padded_ * (num_bits_ - 1) / 8 : 0;
    }

    void decode_short_block(size_t block_idx, size_t valid_vecs, uint8_t* decoded) const {
        const size_t code_bytes = short_code_byte_num();
        std::fill(decoded, decoded + code_bytes * KFastScanSize, 0);
        if (num_bits_ == 0 || valid_vecs == 0) {
            return;
        }

        const uint8_t* block_ptr = short_code(block_idx);
        if (use_fastscan_) {
            static constexpr std::array<size_t, 16> kInvPerm = {
                    0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
            for (size_t col = 0; col < code_bytes; ++col) {
                const uint8_t* packed_col = block_ptr + col * KFastScanSize;
                for (size_t vec = 0; vec < valid_vecs; ++vec) {
                    const size_t base_vec = vec % 16;
                    const size_t perm_idx = kInvPerm[base_vec];
                    const uint8_t high_nibble =
                            vec < 16 ? (packed_col[perm_idx] & 0x0f)
                                     : ((packed_col[perm_idx] >> 4) & 0x0f);
                    const uint8_t low_nibble =
                            vec < 16 ? (packed_col[perm_idx + 16] & 0x0f)
                                     : ((packed_col[perm_idx + 16] >> 4) & 0x0f);
                    decoded[vec * code_bytes + col] =
                            static_cast<uint8_t>((high_nibble << 4) | low_nibble);
                }
            }
        } else {
            std::vector<uint8_t> canonical(code_bytes * KFastScanSize);
            std::memcpy(canonical.data(), block_ptr, canonical.size());
            for (size_t j = 0; j < canonical.size(); j += 8) {
                std::swap(canonical[j + 0], canonical[j + 7]);
                std::swap(canonical[j + 1], canonical[j + 6]);
                std::swap(canonical[j + 2], canonical[j + 5]);
                std::swap(canonical[j + 3], canonical[j + 4]);
            }
            std::memcpy(decoded, canonical.data(), code_bytes * valid_vecs);
        }
    }

    void encode_short_block(const uint8_t* decoded, size_t valid_vecs, uint8_t* block_ptr) const {
        if (num_bits_ == 0) {
            return;
        }

        const size_t code_bytes = short_code_byte_num();
        std::vector<uint8_t> canonical(code_bytes * KFastScanSize, 0);
        std::memcpy(canonical.data(), decoded, code_bytes * valid_vecs);
        if (use_fastscan_) {
            fastscan::pack_codes(num_dim_padded_, canonical.data(), valid_vecs, block_ptr);
        } else {
            for (size_t j = 0; j < canonical.size(); j += 8) {
                std::swap(canonical[j + 0], canonical[j + 7]);
                std::swap(canonical[j + 1], canonical[j + 6]);
                std::swap(canonical[j + 2], canonical[j + 5]);
                std::swap(canonical[j + 3], canonical[j + 4]);
            }
            std::memcpy(block_ptr, canonical.data(), canonical.size());
        }
    }

    // void load(std::ifstream &input)
    // {
    //     input.read((char *)SHORT_DATA.data(), SHORT_DATA.size() * sizeof(uint8_t));
    //     input.read((char *)LONG_CODE.data(), LONG_CODE.size() * sizeof(uint8_t));
    //     input.read((char *)EX_FACTOR.data(), EX_FACTOR.size() * sizeof(ExFactor));
    //     input.read((char *)IDs_.data(), IDs_.size() * sizeof(PID));
    //     for (auto &centroid : centroids_) {
    //         input.read((char *)centroid.data(), centroid.cols() * sizeof(float));
    //     }
    // }
    // void save(std::ofstream &output) const
    // {
    //     output.write((char *)SHORT_DATA.data(), SHORT_DATA.size() * sizeof(uint8_t));
    //     output.write((char *)LONG_CODE.data(), LONG_CODE.size() * sizeof(uint8_t));
    //     output.write((char *)EX_FACTOR.data(), EX_FACTOR.size() * sizeof(ExFactor));
    //     output.write((char *)IDs_.data(), IDs_.size() * sizeof(PID));
    //     for (auto &centroid : centroids_) {
    //         output.write((char *)centroid.data(), centroid.cols() * sizeof(float));
    //     }
    // }
};

class SaqCluData {
    static constexpr size_t kLongCodeAlignBytes = 16;

  public:
    size_t num_vec_;             // Num of vectors in this segment
    size_t num_vec_align_;       // Num of vectors in this segment
    size_t num_blocks_;          // Num of blocks
    const size_t num_segments_;  // Num of segments
  private:
    std::vector<CAQClusterData> segments_;
    size_t shortb_factors_fcnt_ = 0;  // bytes of short factors for all segments
    size_t shortb_code_bytes_ = 0;    // bytes of short code for all segments
    size_t longb_code_bytes_ = 0;     // bytes of long block for all segments
    size_t longb_code_bytes_tot_ = 0; // bytes of long block for all segments

    // ========================= presistence data below =========================
    float *short_factors_;                                    // short factors
    uint8_t *short_code_;                                     // short code
    uint8_t *long_code_;                                      // long code
    ExFactor *long_factors_;                                  // extra factors of vectors
    std::vector<PID, memory::AlignedAllocator<PID, 64>> ids_; // PID of vectors
    bool use_compact_layout_ = false;
    bool use_fastscan_ = true;

  public:
    /**
     * @param num number of vectors
     * @param quant_plan_ quantization plan for each segment. <num_dims, bits>
     */
    explicit SaqCluData(size_t num_vec,
                        const std::vector<std::pair<size_t, size_t>> &quant_plan,
                        bool use_compact_layout = false,
                        bool use_fastscan = true)
        : num_vec_(num_vec),
          num_vec_align_(utils::rd_up_to_multiple_of(num_vec, KFastScanSize)),
          num_blocks_(utils::div_rd_up(num_vec, KFastScanSize)),
          num_segments_(quant_plan.size()),
          use_compact_layout_(use_compact_layout),
          use_fastscan_(use_fastscan) {
        if (num_segments_ == 1)
            use_compact_layout_ = true;

        segments_.reserve(quant_plan.size());
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto dim_padded = quant_plan[i].first;
            DCHECK_EQ(dim_padded % kDimPaddingSize, 0);
            auto &c = segments_.emplace_back(num_vec, dim_padded, quant_plan[i].second);
            c.num_parallel_clusters_ = num_segments_;
            c.use_fastscan_ = use_fastscan_;
            shortb_factors_fcnt_ += c.shortb_factors_num_;
            shortb_code_bytes_ += c.shortb_code_bytes_;

            if (use_compact_layout_) {
                longb_code_bytes_ += c.longb_code_bytes_;
                longb_code_bytes_tot_ += utils::rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                longb_code_bytes_ += utils::rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                longb_code_bytes_tot_ = longb_code_bytes_ * num_vec;
            }
        }

        // assign long code and EX_FACTOR
        if (quant_plan.size() == 1) {
            auto blk_bytes = (shortb_factors_fcnt_ * sizeof(float) + shortb_code_bytes_);
            short_code_ = memory::align_mm<64, uint8_t>(blk_bytes * num_blocks_);
            shortb_code_bytes_ = blk_bytes;
            short_factors_ = nullptr;
            shortb_factors_fcnt_ = 0;
            size_t ptr = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = reinterpret_cast<float *>(short_code_ + ptr);
                ptr += c.shortb_factors_num_ * sizeof(float);
                c.shortb_factors_num_ = blk_bytes / sizeof(float);

                c.short_code_ = short_code_ + ptr;
                ptr += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = blk_bytes;
            }
            // CHECK_EQ(ptr, blk_bytes);
            assert(ptr == blk_bytes);
        } else {
            // TODO: optimize layout of short factors and codes
            short_factors_ = memory::align_mm<64, float>(shortb_factors_fcnt_ * num_blocks_);
            short_code_ = memory::align_mm<64, uint8_t>(shortb_code_bytes_ * num_blocks_);
            size_t shortb_factors_begin = 0;
            size_t shortb_code_begin = 0;
            for (size_t i = 0; i < quant_plan.size(); ++i) {
                auto &c = segments_[i];

                c.short_factors_ = short_factors_ + shortb_factors_begin;
                shortb_factors_begin += c.shortb_factors_num_;
                c.shortb_factors_num_ = shortb_factors_fcnt_;

                c.short_code_ = short_code_ + shortb_code_begin;
                shortb_code_begin += c.shortb_code_bytes_;
                c.shortb_code_bytes_ = shortb_code_bytes_;
            }
            assert(shortb_factors_fcnt_ == shortb_factors_begin);
            assert(shortb_code_bytes_ == shortb_code_begin);
        }

        // assign long code and long_factor
        long_code_ = memory::align_mm<64, uint8_t>(longb_code_bytes_tot_);
        long_factors_ = memory::align_mm<64, ExFactor>(num_vec * num_segments_);
        ids_.resize(num_vec, 0);
        size_t longb_begin = 0;
        for (size_t i = 0; i < quant_plan.size(); ++i) {
            auto &c = segments_[i];
            if (use_compact_layout_) {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += utils::rd_up_to_multiple_of(c.longb_code_bytes_ * num_vec, kLongCodeAlignBytes);
            } else {
                c.long_code_ = long_code_ + longb_begin;
                longb_begin += utils::rd_up_to_multiple_of(c.longb_code_bytes_, kLongCodeAlignBytes);
                c.longb_code_bytes_ = longb_code_bytes_;
            }

            c.long_factors_ = long_factors_ + i;
            c.ids_ = ids_.data();
        }
        assert(longb_begin == longb_code_bytes_tot_ || longb_begin == longb_code_bytes_);
    }

    ~SaqCluData() {
        if (short_factors_) {
            std::free(short_factors_);
        }
        std::free(short_code_);
        std::free(long_code_);
        std::free(long_factors_);
    }

    auto &get_segment(size_t idx) { return segments_[idx]; }
    auto &get_segment(size_t idx) const { return segments_[idx]; }

    /**
     * @brief Return pointer to ids
     */
    PID *ids() { return this->ids_.data(); }
    const PID *ids() const { return ids_.data(); }

    auto iter() const { return num_vec_ / KFastScanSize; }
    auto remain() const { return num_vec_ % KFastScanSize; }
    auto use_compact_layout() const { return use_compact_layout_; }
    auto use_fastscan() const { return use_fastscan_; }

    std::vector<std::pair<size_t, size_t>> quant_plan() const {
        std::vector<std::pair<size_t, size_t>> plan;
        plan.reserve(num_segments_);
        for (const auto& seg : segments_) {
            plan.emplace_back(seg.num_dim_padded_, seg.num_bits_);
        }
        return plan;
    }

    void resize(size_t new_num_vec) {
        if (new_num_vec == num_vec_) {
            return;
        }

        SaqCluData grown(new_num_vec, quant_plan(), use_compact_layout_, use_fastscan_);

        for (size_t i = 0; i < num_segments_; ++i) {
            grown.segments_[i].centroid_ = segments_[i].centroid_;
        }

        const size_t old_num_vec = num_vec_;
        const size_t old_num_blocks = num_blocks_;
        const size_t old_short_factors_bytes =
                short_factors_ ? shortb_factors_fcnt_ * old_num_blocks * sizeof(float) : 0;
        const size_t old_short_code_bytes = shortb_code_bytes_ * old_num_blocks;
        const size_t old_long_factors_bytes = old_num_vec * num_segments_ * sizeof(ExFactor);

        if (old_short_factors_bytes > 0) {
            std::memcpy(grown.short_factors_, short_factors_, old_short_factors_bytes);
        }
        if (old_short_code_bytes > 0) {
            std::memcpy(grown.short_code_, short_code_, old_short_code_bytes);
        }

        if (old_num_vec > 0) {
            if (use_compact_layout_) {
                for (size_t i = 0; i < num_segments_; ++i) {
                    const auto raw_long_code_bytes = segments_[i].raw_long_code_bytes();
                    if (raw_long_code_bytes == 0) {
                        continue;
                    }
                    std::memcpy(
                            grown.segments_[i].long_code_,
                            segments_[i].long_code_,
                            raw_long_code_bytes * old_num_vec);
                }
            } else {
                std::memcpy(
                        grown.long_code_,
                        long_code_,
                        longb_code_bytes_ * old_num_vec);
            }
            std::memcpy(grown.long_factors_, long_factors_, old_long_factors_bytes);
            std::copy(ids_.begin(), ids_.end(), grown.ids_.begin());
        }

        swap_contents(grown);
    }

    void append(const SaqCluData& other) {
        if (other.num_vec_ == 0) {
            return;
        }

        CHECK_EQ(num_segments_, other.num_segments_) << "SaqCluData append num_segments mismatch";
        CHECK_EQ(use_compact_layout_, other.use_compact_layout_) << "SaqCluData append compact-layout mismatch";
        CHECK_EQ(use_fastscan_, other.use_fastscan_) << "SaqCluData append fastscan mismatch";
        for (size_t i = 0; i < num_segments_; ++i) {
            CHECK_EQ(segments_[i].num_dim_padded_, other.segments_[i].num_dim_padded_);
            CHECK_EQ(segments_[i].num_bits_, other.segments_[i].num_bits_);
        }

        const size_t old_num_vec = num_vec_;
        const size_t old_tail = old_num_vec % KFastScanSize;
        const size_t rewrite_block = old_tail ? (old_num_vec / KFastScanSize) : num_blocks_;

        resize(old_num_vec + other.num_vec_);
        std::copy(other.ids_.begin(), other.ids_.end(), ids_.begin() + old_num_vec);

        for (size_t seg_idx = 0; seg_idx < num_segments_; ++seg_idx) {
            auto& dst = segments_[seg_idx];
            const auto& src = other.segments_[seg_idx];
            const size_t raw_long_code_bytes = dst.raw_long_code_bytes();

            if (old_num_vec == 0) {
                dst.centroid_ = src.centroid_;
            }

            for (size_t i = 0; i < other.num_vec_; ++i) {
                if (raw_long_code_bytes > 0) {
                    std::memcpy(
                            dst.long_code(old_num_vec + i),
                            src.long_code(i),
                            raw_long_code_bytes);
                }
                dst.long_factor(old_num_vec + i) = src.long_factor(i);
            }

            std::vector<uint8_t> old_tail_codes(dst.short_code_byte_num() * KFastScanSize, 0);
            std::array<float, KFastScanSize> old_tail_norms{};
            std::array<float, KFastScanSize> old_tail_ips{};
            if (old_tail > 0) {
                dst.decode_short_block(rewrite_block, old_tail, old_tail_codes.data());
                std::memcpy(
                        old_tail_norms.data(),
                        dst.factor_o_l2norm(rewrite_block),
                        sizeof(float) * old_tail);
                std::memcpy(
                        old_tail_ips.data(),
                        dst.factor_ip_cent_oa(rewrite_block),
                        sizeof(float) * old_tail);
            }

            std::vector<uint8_t> src_block_codes(src.short_code_byte_num() * KFastScanSize, 0);
            std::vector<uint8_t> dst_block_codes(dst.short_code_byte_num() * KFastScanSize, 0);
            std::array<float, KFastScanSize> dst_block_norms{};
            std::array<float, KFastScanSize> dst_block_ips{};

            size_t src_vec_idx = 0;
            size_t src_block_idx = std::numeric_limits<size_t>::max();
            for (size_t block_idx = rewrite_block; block_idx < num_blocks_; ++block_idx) {
                const size_t block_begin = block_idx * KFastScanSize;
                const size_t block_count = std::min(KFastScanSize, num_vec_ - block_begin);
                size_t filled = 0;

                std::fill(dst_block_codes.begin(), dst_block_codes.end(), 0);
                dst_block_norms.fill(0);
                dst_block_ips.fill(0);

                if (block_idx == rewrite_block && old_tail > 0) {
                    std::memcpy(
                            dst_block_codes.data(),
                            old_tail_codes.data(),
                            old_tail * dst.short_code_byte_num());
                    std::memcpy(
                            dst_block_norms.data(),
                            old_tail_norms.data(),
                            sizeof(float) * old_tail);
                    std::memcpy(
                            dst_block_ips.data(),
                            old_tail_ips.data(),
                            sizeof(float) * old_tail);
                    filled = old_tail;
                }

                while (filled < block_count) {
                    const size_t cur_src_block = src_vec_idx / KFastScanSize;
                    const size_t src_in_block = src_vec_idx % KFastScanSize;
                    if (cur_src_block != src_block_idx) {
                        src_block_idx = cur_src_block;
                        const size_t src_valid =
                                std::min(KFastScanSize, src.num_vec_ - src_block_idx * KFastScanSize);
                        src.decode_short_block(src_block_idx, src_valid, src_block_codes.data());
                    }

                    const size_t src_avail = std::min(
                            {block_count - filled,
                             KFastScanSize - src_in_block,
                             src.num_vec_ - src_vec_idx});

                    if (dst.short_code_byte_num() > 0) {
                        std::memcpy(
                                dst_block_codes.data() + filled * dst.short_code_byte_num(),
                                src_block_codes.data() + src_in_block * src.short_code_byte_num(),
                                src_avail * dst.short_code_byte_num());
                    }
                    std::memcpy(
                            dst_block_norms.data() + filled,
                            src.factor_o_l2norm(src_block_idx) + src_in_block,
                            sizeof(float) * src_avail);
                    std::memcpy(
                            dst_block_ips.data() + filled,
                            src.factor_ip_cent_oa(src_block_idx) + src_in_block,
                            sizeof(float) * src_avail);
                    filled += src_avail;
                    src_vec_idx += src_avail;
                }

                if (dst.num_bits_ > 0) {
                    dst.encode_short_block(dst_block_codes.data(), block_count, dst.short_code(block_idx));
                }
                std::memcpy(
                        dst.factor_o_l2norm(block_idx),
                        dst_block_norms.data(),
                        sizeof(float) * KFastScanSize);
                std::memcpy(
                        dst.factor_ip_cent_oa(block_idx),
                        dst_block_ips.data(),
                        sizeof(float) * KFastScanSize);
            }
        }
    }

    void load(std::ifstream &input) {
        input.read((char *)short_factors_, shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
        input.read((char *)short_code_, shortb_code_bytes_ * num_blocks_);
        input.read((char *)long_code_, longb_code_bytes_ * num_vec_);
        input.read((char *)long_factors_, num_vec_ * num_segments_ * sizeof(ExFactor));
        input.read((char *)ids_.data(), ids_.size() * sizeof(PID));
        for (auto &clu : segments_) {
            input.read((char *)clu.centroid_.data(), clu.centroid_.cols() * sizeof(float));
        }
    }
    void save(std::ofstream &output) const {
        output.write((char *)short_factors_, shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
        output.write((char *)short_code_, shortb_code_bytes_ * num_blocks_);
        output.write((char *)long_code_, longb_code_bytes_ * num_vec_);
        output.write((char *)long_factors_, num_vec_ * num_segments_ * sizeof(ExFactor));
        output.write((char *)ids_.data(), ids_.size() * sizeof(PID));
        for (auto &clu : segments_) {
            output.write((char *)clu.centroid_.data(), clu.centroid_.cols() * sizeof(float));
        }
    }

  private:
    void swap_contents(SaqCluData& other) {
        using std::swap;
        swap(num_vec_, other.num_vec_);
        swap(num_vec_align_, other.num_vec_align_);
        swap(num_blocks_, other.num_blocks_);
        swap(segments_, other.segments_);
        swap(shortb_factors_fcnt_, other.shortb_factors_fcnt_);
        swap(shortb_code_bytes_, other.shortb_code_bytes_);
        swap(longb_code_bytes_, other.longb_code_bytes_);
        swap(longb_code_bytes_tot_, other.longb_code_bytes_tot_);
        swap(short_factors_, other.short_factors_);
        swap(short_code_, other.short_code_);
        swap(long_code_, other.long_code_);
        swap(long_factors_, other.long_factors_);
        swap(ids_, other.ids_);
        swap(use_compact_layout_, other.use_compact_layout_);
        swap(use_fastscan_, other.use_fastscan_);
    }
};
} // namespace saqlib
