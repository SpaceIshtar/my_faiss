/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/DistanceComputer.h>

#include <memory>
#include <sstream>

namespace hnsw_bench {

/**
 * Optimized Product Quantization (OPQ) wrapper.
 * OPQ learns a rotation matrix to minimize PQ reconstruction error.
 *
 * Wraps FAISS OPQMatrix + IndexPQ via IndexPreTransform for use in HNSW benchmarks.
 * The IndexPreTransform automatically handles query transformation in the
 * distance computer.
 */
class OPQWrapper : public QuantWrapper {
public:
    /**
     * Create an OPQ wrapper.
     * @param d Dimension of vectors
     * @param M Number of subquantizers (d must be divisible by M)
     * @param nbits Number of bits per subquantizer (typically 8)
     * @param metric Distance metric (L2 or IP)
     * @param niter Number of OPQ training iterations (default 50)
     */
    OPQWrapper(size_t d, size_t M, size_t nbits,
               faiss::MetricType metric = faiss::METRIC_L2,
               int niter = 50)
        : d_(d), M_(M), nbits_(nbits), metric_(metric), niter_(niter) {

        // Create OPQ rotation matrix (d -> d, no dimension change)
        opq_ = new faiss::OPQMatrix(d, M);
        opq_->niter = niter;

        // Create PQ index for the transformed vectors
        pq_index_ = new faiss::IndexPQ(d, M, nbits, metric);

        // Wrap with IndexPreTransform: OPQ rotation -> PQ quantization
        // IndexPreTransform takes ownership when own_fields = true
        index_ = std::make_unique<faiss::IndexPreTransform>(opq_, pq_index_);
        index_->own_fields = true;  // index_ will delete opq_ and pq_index_
    }

    void train(size_t n, const float* x) override {
        index_->train(n, x);
    }

    void add(size_t n, const float* x) override {
        index_->add(n, x);
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        // IndexPreTransform::get_distance_computer() returns a
        // PreTransformDistanceComputer that automatically applies
        // the OPQ rotation to queries before computing PQ distances
        return std::unique_ptr<faiss::DistanceComputer>(
            index_->get_distance_computer());
    }

    std::string get_name() const override {
        return "OPQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "M" << M_ << "_nbits" << nbits_ << "_niter" << niter_;
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return index_->ntotal;
    }

    // OPQ-specific accessors
    size_t get_M() const { return M_; }
    size_t get_nbits() const { return nbits_; }
    int get_niter() const { return niter_; }
    faiss::IndexPreTransform* get_index() { return index_.get(); }
    faiss::OPQMatrix* get_opq() { return opq_; }
    faiss::IndexPQ* get_pq_index() { return pq_index_; }

private:
    size_t d_;
    size_t M_;
    size_t nbits_;
    faiss::MetricType metric_;
    int niter_;

    // Note: opq_ and pq_index_ are owned by index_ (own_fields = true)
    faiss::OPQMatrix* opq_;
    faiss::IndexPQ* pq_index_;
    std::unique_ptr<faiss::IndexPreTransform> index_;
};

/**
 * Factory function to create OPQWrapper from parameter map.
 * Expected params:
 *   "M" (optional, default 8) - number of subquantizers
 *   "nbits" (optional, default 8) - bits per subquantizer
 *   "niter" (optional, default 50) - OPQ training iterations
 */
inline std::unique_ptr<QuantWrapper> create_opq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {

    size_t M = 8;      // default
    size_t nbits = 8;  // default
    int niter = 50;    // default

    auto it = params.find("M");
    if (it != params.end()) {
        M = std::stoul(it->second);
    }

    it = params.find("nbits");
    if (it != params.end()) {
        nbits = std::stoul(it->second);
    }

    it = params.find("niter");
    if (it != params.end()) {
        niter = std::stoi(it->second);
    }

    return std::make_unique<OPQWrapper>(d, M, nbits, metric, niter);
}

} // namespace hnsw_bench
