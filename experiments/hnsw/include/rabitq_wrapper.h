/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

// RaBitQ library includes
#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/MetricType.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace hnsw_bench {

/**
 * RaBitQ Distance Computer for FAISS.
 * Implements FAISS DistanceComputer interface using RaBitQ distance estimation.
 *
 * Uses proper centroid-based normalization as in the original RaBitQ library:
 * - Each vector is quantized relative to its cluster centroid
 * - During search, g_add = ||q - c||^2 for L2 (where c is the centroid)
 */
class RaBitQDistanceComputer : public faiss::DistanceComputer {
public:
    RaBitQDistanceComputer(
            size_t dim,
            size_t padded_dim,
            size_t ex_bits,
            size_t ntotal,
            size_t num_clusters,
            const char* codes,                    // quantized codes storage
            const uint32_t* cluster_ids,          // cluster ID for each vector
            const float* rotated_centroids,       // rotated centroids (num_clusters x padded_dim)
            size_t code_size,
            size_t bin_data_size,
            rabitqlib::Rotator<float>* rotator,
            const rabitqlib::quant::RabitqConfig& config,
            rabitqlib::MetricType metric_type)
        : dim_(dim)
        , padded_dim_(padded_dim)
        , ex_bits_(ex_bits)
        , ntotal_(ntotal)
        , num_clusters_(num_clusters)
        , codes_(codes)
        , cluster_ids_(cluster_ids)
        , rotated_centroids_(rotated_centroids)
        , code_size_(code_size)
        , bin_data_size_(bin_data_size)
        , rotator_(rotator)
        , config_(config)
        , metric_type_(metric_type)
        , rotated_query_(padded_dim)
        , q_to_centroids_(num_clusters)
        , query_wrapper_(nullptr) {

        // Select ip function for extra bits
        ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);
    }

    void set_query(const float* x) override {
        // Rotate query vector
        rotator_->rotate(x, rotated_query_.data());

        // Precompute distances from query to all centroids
        // For L2: store sqrt(||q - c||^2) as norm, g_add = norm^2, g_error = norm
        for (size_t i = 0; i < num_clusters_; i++) {
            float dist_sq = rabitqlib::euclidean_sqr(
                rotated_query_.data(),
                rotated_centroids_ + i * padded_dim_,
                padded_dim_
            );
            q_to_centroids_[i] = std::sqrt(dist_sq);  // store norm
        }

        // Create query wrapper for distance estimation
        query_wrapper_ = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
            rotated_query_.data(),
            padded_dim_,
            ex_bits_,
            config_,
            metric_type_
        );
    }

    float operator()(faiss::idx_t i) override {
        const char* vec_data = codes_ + i * code_size_;
        const char* bin_data = vec_data;
        const char* ex_data = vec_data + bin_data_size_;

        // Get the centroid distance for this vector's cluster
        uint32_t cluster_id = cluster_ids_[i];
        float norm = q_to_centroids_[cluster_id];
        float g_add = norm * norm;  // ||q - c||^2
        float g_error = norm;       // ||q - c||

        float est_dist, low_dist, ip_x0_qr;

        if (ex_bits_ > 0) {
            // Full estimation with extra bits
            rabitqlib::split_single_fulldist(
                bin_data,
                ex_data,
                ip_func_,
                *query_wrapper_,
                padded_dim_,
                ex_bits_,
                est_dist,
                low_dist,
                ip_x0_qr,
                g_add,
                g_error
            );
        } else {
            // 1-bit only estimation
            rabitqlib::split_single_estdist(
                bin_data,
                *query_wrapper_,
                padded_dim_,
                ip_x0_qr,
                est_dist,
                low_dist,
                g_add,
                g_error
            );
        }

        return est_dist;
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        // Not implemented - we only need asymmetric distance
        return 0.0f;
    }

private:
    size_t dim_;
    size_t padded_dim_;
    size_t ex_bits_;
    size_t ntotal_;
    size_t num_clusters_;
    const char* codes_;
    const uint32_t* cluster_ids_;
    const float* rotated_centroids_;
    size_t code_size_;
    size_t bin_data_size_;
    rabitqlib::Rotator<float>* rotator_;
    rabitqlib::quant::RabitqConfig config_;
    rabitqlib::MetricType metric_type_;

    std::vector<float> rotated_query_;
    std::vector<float> q_to_centroids_;  // distance from query to each centroid
    std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query_wrapper_;
    float (*ip_func_)(const float*, const uint8_t*, size_t);
};

/**
 * RaBitQ Quantization wrapper.
 * Wraps the official RaBitQ library for use in HNSW benchmarks.
 *
 * Following the official RaBitQ library's approach:
 * 1. train() runs KMeans clustering to get centroids
 * 2. add() assigns vectors to clusters and quantizes relative to centroids
 * 3. DistanceComputer uses centroid info for proper g_add computation
 */
class RaBitQWrapper : public QuantWrapper {
public:
    /**
     * Create a RaBitQ wrapper.
     * @param d Dimension of vectors
     * @param total_bits Total bits for quantization (1 = binary only, 2-9 = with extra bits)
     * @param metric FAISS metric type (L2 or IP)
     * @param num_clusters Number of clusters for IVF (default 16 as in RaBitQ paper)
     */
    RaBitQWrapper(size_t d, size_t total_bits = 4, faiss::MetricType metric = faiss::METRIC_L2,
                  size_t num_clusters = 16)
        : d_(d)
        , total_bits_(total_bits)
        , metric_(metric)
        , num_clusters_(num_clusters)
        , ntotal_(0)
        , is_trained_(false) {

        // Validate total_bits (1-9 supported by RaBitQ)
        if (total_bits < 1 || total_bits > 9) {
            throw std::runtime_error("RaBitQ total_bits must be between 1 and 9");
        }

        ex_bits_ = total_bits - 1;  // Extra bits beyond 1-bit

        // Create rotator (FHT-based for efficiency)
        padded_dim_ = rabitqlib::round_up_to_multiple(d_, 64);
        rotator_.reset(rabitqlib::choose_rotator<float>(
            d_, rabitqlib::RotatorType::FhtKacRotator, padded_dim_));

        // Calculate storage sizes per vector
        bin_data_size_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
        ex_data_size_ = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        code_size_ = bin_data_size_ + ex_data_size_;

        // Initialize config for faster quantization
        config_ = rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);

        // Convert metric type
        metric_type_ = (metric == faiss::METRIC_INNER_PRODUCT)
            ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;

        // Allocate centroids storage (rotated)
        rotated_centroids_.resize(num_clusters_ * padded_dim_);
    }

    void train(size_t n, const float* x) override {
        // Use FAISS Clustering for KMeans
        faiss::ClusteringParameters cp;
        cp.niter = 25;
        cp.seed = 1234;
        cp.verbose = false;

        faiss::Clustering clus(d_, num_clusters_, cp);
        faiss::IndexFlatL2 index(d_);

        // Run clustering
        clus.train(n, x, index);

        // Store centroids
        centroids_.resize(num_clusters_ * d_);
        std::memcpy(centroids_.data(), clus.centroids.data(), num_clusters_ * d_ * sizeof(float));

        // Rotate centroids
        for (size_t i = 0; i < num_clusters_; i++) {
            rotator_->rotate(
                centroids_.data() + i * d_,
                rotated_centroids_.data() + i * padded_dim_
            );
        }

        // Build quantizer for fast cluster assignment
        quantizer_.reset(new faiss::IndexFlatL2(d_));
        quantizer_->add(num_clusters_, centroids_.data());

        is_trained_ = true;
    }

    void add(size_t n, const float* x) override {
        if (!is_trained_) {
            // Auto-train if not trained yet
            train(n, x);
        }

        // Resize storage
        size_t old_ntotal = ntotal_;
        ntotal_ += n;
        codes_.resize(ntotal_ * code_size_);
        cluster_ids_.resize(ntotal_);

        // Assign vectors to clusters
        std::vector<float> dists(n);
        std::vector<faiss::idx_t> labels(n);
        quantizer_->search(n, x, 1, dists.data(), labels.data());

        // Temporary storage for rotated vector
        std::vector<float> rotated(padded_dim_);

        // Quantize each vector relative to its cluster centroid
        for (size_t i = 0; i < n; i++) {
            size_t cid = labels[i];
            cluster_ids_[old_ntotal + i] = static_cast<uint32_t>(cid);

            // Rotate the vector
            rotator_->rotate(x + i * d_, rotated.data());

            // Get pointer to this vector's code storage
            char* vec_code = codes_.data() + (old_ntotal + i) * code_size_;
            char* bin_data = vec_code;
            char* ex_data = vec_code + bin_data_size_;

            // Quantize relative to the cluster centroid (rotated)
            rabitqlib::quant::quantize_split_single(
                rotated.data(),
                rotated_centroids_.data() + cid * padded_dim_,  // Use cluster centroid
                padded_dim_,
                ex_bits_,
                bin_data,
                ex_data,
                metric_type_,
                config_
            );
        }
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        return std::make_unique<RaBitQDistanceComputer>(
            d_,
            padded_dim_,
            ex_bits_,
            ntotal_,
            num_clusters_,
            codes_.data(),
            cluster_ids_.data(),
            rotated_centroids_.data(),
            code_size_,
            bin_data_size_,
            rotator_.get(),
            config_,
            metric_type_
        );
    }

    std::string get_name() const override {
        return "RaBitQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "bits" << total_bits_ << "_c" << num_clusters_;
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return ntotal_;
    }

    // RaBitQ-specific accessors
    size_t get_total_bits() const { return total_bits_; }
    size_t get_padded_dim() const { return padded_dim_; }
    size_t get_code_size() const { return code_size_; }
    size_t get_num_clusters() const { return num_clusters_; }

    // Accessors for native search
    const char* get_codes() const { return codes_.data(); }
    const uint32_t* get_cluster_ids() const { return cluster_ids_.data(); }
    const float* get_rotated_centroids() const { return rotated_centroids_.data(); }
    size_t get_bin_data_size() const { return bin_data_size_; }
    size_t get_ex_bits() const { return ex_bits_; }
    rabitqlib::Rotator<float>* get_rotator() const { return rotator_.get(); }
    const rabitqlib::quant::RabitqConfig& get_config() const { return config_; }
    rabitqlib::MetricType get_rabitq_metric() const { return metric_type_; }

private:
    size_t d_;
    size_t padded_dim_;
    size_t total_bits_;
    size_t ex_bits_;
    size_t num_clusters_;
    faiss::MetricType metric_;
    rabitqlib::MetricType metric_type_;
    size_t ntotal_;
    bool is_trained_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    rabitqlib::quant::RabitqConfig config_;

    size_t bin_data_size_;
    size_t ex_data_size_;
    size_t code_size_;

    std::vector<float> centroids_;          // Original centroids (num_clusters x d)
    std::vector<float> rotated_centroids_;  // Rotated centroids (num_clusters x padded_dim)
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;  // For cluster assignment

    std::vector<char> codes_;               // Quantized codes for all vectors
    std::vector<uint32_t> cluster_ids_;     // Cluster ID for each vector
};

/**
 * Factory function to create RaBitQWrapper from parameter map.
 * Expected params: "total_bits" or "bits" (optional, default 4)
 *                  "clusters" (optional, default 16)
 */
inline std::unique_ptr<QuantWrapper> create_rabitq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {

    size_t total_bits = 4;  // default: 1-bit + 3 extra bits
    size_t num_clusters = 16;  // default as in RaBitQ paper

    auto it = params.find("total_bits");
    if (it != params.end()) {
        total_bits = std::stoul(it->second);
    }

    // Also accept "bits" as parameter name
    it = params.find("bits");
    if (it != params.end()) {
        total_bits = std::stoul(it->second);
    }

    it = params.find("clusters");
    if (it != params.end()) {
        num_clusters = std::stoul(it->second);
    }

    return std::make_unique<RaBitQWrapper>(d, total_bits, metric, num_clusters);
}

} // namespace hnsw_bench
