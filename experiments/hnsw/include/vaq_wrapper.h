/*
 * VAQ (Variance-Aware Quantization) wrapper for HNSW benchmark framework.
 *
 * VAQ uses PCA-based rotation + variance-aware bit allocation + per-subspace
 * K-means to quantize vectors. Distance estimation uses precomputed LUTs.
 */

#pragma once

#include "quant_wrapper.h"

#include "VAQ.hpp"

#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include <immintrin.h>

namespace hnsw_bench {

/*****************************************************
 * VAQDistanceComputer: LUT-based distance estimation
 *****************************************************/

class VAQDistanceComputer : public faiss::DistanceComputer {
public:
    __m256i v_step, v_loop_increment;
    VAQDistanceComputer(const VAQ* vaq, size_t d_original, size_t d_padded)
        : vaq_(vaq), d_original_(d_original), d_padded_(d_padded) {
        lut_rows_ = 1 << vaq_->mMaxBitsPerSubs;
        lut_.resize((size_t)lut_rows_ * vaq_->mHighestSubs, 0.0f);
        // 1. 预计算步长增加量向量: [0, 1*rows, 2*rows, ..., 7*rows]
        // 这部分如果 lut_rows_ 不变，甚至可以挪到构造函数里或者作为成员变量
        v_step = _mm256_set_epi32(
            7 * lut_rows_, 6 * lut_rows_, 5 * lut_rows_, 4 * lut_rows_,
            3 * lut_rows_, 2 * lut_rows_, 1 * lut_rows_, 0 * lut_rows_
        );
        v_loop_increment = _mm256_set1_epi32(8 * lut_rows_);
    }

    void set_query(const float* x) override {
        // 1. Create padded query vector
        RowVectorXf q = RowVectorXf::Zero(d_padded_);
        memcpy(q.data(), x, sizeof(float) * d_original_);

        // 2. Project via PCA eigenvectors: q_proj = (q * EigenVectors).real()
        RowVectorXf query = (q * vaq_->mEigenVectors).real();

        // 3. Build LUT: for each subspace, compute L2sqr from query to centroids
        std::fill(lut_.begin(), lut_.end(), 0.0f);
        for (int subs = 0; subs < vaq_->mHighestSubs; subs++) {
            float* lut_ptr = lut_.data() + (size_t)lut_rows_ * subs;
            const float* qsub = query.data() + subs * vaq_->mSubsLen;
            const float* cents = vaq_->mCentroidsPerSubs[subs].data();
            int nc = vaq_->mCentroidsNum[subs];
            int dsub = vaq_->mSubsLen;
            faiss::fvec_L2sqr_ny(lut_ptr, qsub, cents, dsub, nc);
        }
    }

    /// Approximate L2 squared distance for vector i
    // float operator()(faiss::idx_t i) override {
    //     const uint16_t* codes =
    //         vaq_->mCodebook.data() + i * vaq_->mCodebook.cols();
    //     float dist = 0;
    //     for (int subs = 0; subs < vaq_->mHighestSubs; subs++) {
    //         dist += lut_[(size_t)lut_rows_ * subs + codes[subs]];
    //     }
    //     return dist;
    // }
    inline float horizontal_sum(const __m128 v) {
        // say, v is [x0, x1, x2, x3]

        // v0 is [x2, x3, ..., ...]
        const __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
        // v1 is [x0 + x2, x1 + x3, ..., ...]
        const __m128 v1 = _mm_add_ps(v, v0);
        // v2 is [x1 + x3, ..., .... ,...]
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        // v3 is [x0 + x1 + x2 + x3, ..., ..., ...]
        const __m128 v3 = _mm_add_ps(v1, v2);
        // return v3[0]
        return _mm_cvtss_f32(v3);
    }

    inline float horizontal_sum(const __m256 v) {
        // add high and low parts
        const __m128 v0 =
                _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        // perform horizontal sum on v0
        return horizontal_sum(v0);
    }

    float operator()(faiss::idx_t i) override {
        const uint16_t* codes = vaq_->mCodebook.data() + i * vaq_->mCodebook.cols();
        float dist = 0;
        
        int subs = 0;
        __m256 sum = _mm256_setzero_ps();

        // 3. 当前的基础偏移量向量
        // 初始值为当前 subs 对应的 offset
        __m256i v_current_offsets = v_step; 

        for (; subs <= vaq_->mHighestSubs - 8; subs += 8) {
            // 加载并转换 indices
            __m128i v_codes_16 = _mm_loadu_si128((const __m128i*)(codes + subs));
            __m256i v_codes_32 = _mm256_cvtepu16_epi32(v_codes_16);
            
            // 计算最终索引：直接向量加法，无需经过内存数组
            __m256i v_final_indices = _mm256_add_epi32(v_current_offsets, v_codes_32);

            // Gather 加载
            __m256 values = _mm256_i32gather_ps(lut_.data(), v_final_indices, 4);
            sum = _mm256_add_ps(sum, values);

            // 更新偏移量：为下一轮循环准备
            v_current_offsets = _mm256_add_epi32(v_current_offsets, v_loop_increment);
        }

        // Horizontal sum of the register
        // alignas(32) float res[8];
        // _mm256_store_ps(res, sum);
        // for (int j = 0; j < 8; ++j) dist += res[j];
        dist = horizontal_sum(sum);

        // Clean up remaining elements (scalar tail)
        for (; subs < vaq_->mHighestSubs; subs++) {
            dist += lut_[(size_t)lut_rows_ * subs + codes[subs]];
        }

        return dist;
    }

    /// Symmetric distance between two indexed vectors
    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint16_t* ci =
            vaq_->mCodebook.data() + i * vaq_->mCodebook.cols();
        const uint16_t* cj =
            vaq_->mCodebook.data() + j * vaq_->mCodebook.cols();
        float dist = 0;
        for (int subs = 0; subs < vaq_->mHighestSubs; subs++) {
            const float* cent_i =
                vaq_->mCentroidsPerSubs[subs].data() +
                ci[subs] * vaq_->mSubsLen;
            const float* cent_j =
                vaq_->mCentroidsPerSubs[subs].data() +
                cj[subs] * vaq_->mSubsLen;
            for (int k = 0; k < vaq_->mSubsLen; k++) {
                float diff = cent_i[k] - cent_j[k];
                dist += diff * diff;
            }
        }
        return dist;
    }

private:
    const VAQ* vaq_;
    size_t d_original_;
    size_t d_padded_;
    int lut_rows_;
    std::vector<float> lut_;
};

/*****************************************************
 * VAQWrapper
 *****************************************************/

/**
 * VAQ wrapper for the HNSW benchmark framework.
 *
 * Parameters:
 *   bitBudget     - total bits per vector (e.g., 64, 128, 256)
 *   subspaceNum   - number of subspaces (e.g., 16, 32)
 *   minBitsPerSub - minimum bits per subspace (e.g., 1, 7)
 *   maxBitsPerSub - maximum bits per subspace (e.g., 8, 13)
 *   varExplained  - variance ratio to retain (e.g., 0.95, 1.0)
 */
class VAQWrapper : public QuantWrapper {
public:
    VAQWrapper(size_t d, int bitBudget, int subspaceNum,
               int minBitsPerSub, int maxBitsPerSub,
               float varExplained = 1.0f)
        : d_(d),
          bitBudget_(bitBudget),
          subspaceNum_(subspaceNum),
          minBitsPerSub_(minBitsPerSub),
          maxBitsPerSub_(maxBitsPerSub),
          varExplained_(varExplained) {

        // Compute padded dimension (must be divisible by subspaceNum)
        int subsLen = d_ / subspaceNum_;
        if (d_ % subspaceNum_ != 0) {
            subsLen += 1;
        }
        d_padded_ = subsLen * subspaceNum_;

        // Configure VAQ method string
        std::ostringstream oss;
        oss << "VAQ" << bitBudget_
            << "m" << subspaceNum_
            << "min" << minBitsPerSub_
            << "max" << maxBitsPerSub_
            << "var" << varExplained_
            << ",EA";
        method_string_ = oss.str();
    }

    ~VAQWrapper() override = default;

    void train(size_t n, const float* x) override {
        // Initialize VAQ
        vaq_ = std::make_unique<VAQ>();
        vaq_->parseMethodString(method_string_);

        // Convert float* to padded Eigen matrix
        RowMatrixXf data = RowMatrixXf::Zero(n, d_padded_);
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            memcpy(data.row(i).data(), x + i * d_, sizeof(float) * d_);
        }

        // Train: PCA + variance balancing + bit allocation + centroids
        vaq_->train(data, /*verbose=*/true);

        // Encode: assign codebook codes using projected data
        // (train() already projected data in-place via eigenvectors)
        vaq_->encode(data);

        ntotal_ = n;
    }

    void add(size_t /*n*/, const float* /*x*/) override {
        // Encoding is done in train() since VAQ projects data in-place
        // during training and encode() uses the projected result.
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        return std::make_unique<VAQDistanceComputer>(vaq_.get(), d_, d_padded_);
    }

    std::string get_name() const override { return "VAQ"; }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "b" << bitBudget_
            << "_m" << subspaceNum_
            << "_min" << minBitsPerSub_
            << "_max" << maxBitsPerSub_;
        return oss.str();
    }

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return ntotal_; }

private:
    size_t d_;
    size_t d_padded_;
    int bitBudget_;
    int subspaceNum_;
    int minBitsPerSub_;
    int maxBitsPerSub_;
    float varExplained_;
    std::string method_string_;

    std::unique_ptr<VAQ> vaq_;
    size_t ntotal_ = 0;
};

/*****************************************************
 * Factory function
 *****************************************************/

inline std::unique_ptr<QuantWrapper> create_vaq_wrapper(
        size_t d,
        faiss::MetricType /*metric*/,
        const std::map<std::string, std::string>& params) {

    int bits = 256;
    int nsub = 32;
    int minbps = 7;
    int maxbps = 13;
    float var = 1.0f;

    auto it = params.find("bits");
    if (it != params.end()) bits = std::stoi(it->second);

    it = params.find("nsub");
    if (it != params.end()) nsub = std::stoi(it->second);

    it = params.find("minbps");
    if (it != params.end()) minbps = std::stoi(it->second);

    it = params.find("maxbps");
    if (it != params.end()) maxbps = std::stoi(it->second);

    it = params.find("var");
    if (it != params.end()) var = std::stof(it->second);

    return std::make_unique<VAQWrapper>(d, bits, nsub, minbps, maxbps, var);
}

} // namespace hnsw_bench
