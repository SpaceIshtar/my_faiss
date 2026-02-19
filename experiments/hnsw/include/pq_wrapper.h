/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/IndexPQ.h>
#include <faiss/impl/DistanceComputer.h>

#include <sstream>

namespace hnsw_bench {

/**
 * Product Quantization wrapper.
 * Wraps FAISS IndexPQ for use in HNSW benchmarks.
 */
class PQWrapper : public QuantWrapper {
public:
    /**
     * Create a PQ wrapper.
     * @param d Dimension of vectors
     * @param M Number of subquantizers (d must be divisible by M)
     * @param nbits Number of bits per subquantizer (typically 8)
     * @param metric Distance metric (L2 or IP)
     */
    PQWrapper(size_t d, size_t M, size_t nbits, faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), M_(M), nbits_(nbits), metric_(metric) {
        index_ = std::make_unique<faiss::IndexPQ>(d, M, nbits, metric);
    }

    void train(size_t n, const float* x) override {
        index_->train(n, x);
    }

    void add(size_t n, const float* x) override {
        index_->add(n, x);
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        return std::unique_ptr<faiss::DistanceComputer>(
            index_->get_FlatCodesDistanceComputer());
    }

    std::string get_name() const override {
        return "PQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "M" << M_ << "_nbits" << nbits_;
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return index_->ntotal;
    }

    faiss::Index* get_faiss_index() override { return index_.get(); }

    bool load(const std::string& path) override {
        return load_faiss_index(path, index_);
    }

    // PQ-specific accessors
    size_t get_M() const { return M_; }
    size_t get_nbits() const { return nbits_; }
    faiss::IndexPQ* get_index() { return index_.get(); }

private:
    size_t d_;
    size_t M_;
    size_t nbits_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexPQ> index_;
};

/**
 * Factory function to create PQWrapper from parameter map.
 * Expected params: "M" (required), "nbits" (optional, default 8)
 */
inline std::unique_ptr<QuantWrapper> create_pq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {

    size_t M = 8;  // default
    size_t nbits = 8;  // default

    auto it = params.find("M");
    if (it != params.end()) {
        M = std::stoul(it->second);
    }

    it = params.find("nbits");
    if (it != params.end()) {
        nbits = std::stoul(it->second);
    }

    return std::make_unique<PQWrapper>(d, M, nbits, metric);
}

} // namespace hnsw_bench
