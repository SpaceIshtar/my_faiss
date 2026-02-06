/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/impl/DistanceComputer.h>

#include <memory>
#include <sstream>

namespace hnsw_bench {

/**
 * Residual Quantizer (RQ) wrapper.
 * Residual quantization iteratively quantizes the residual of the previous step.
 */
class RQWrapper : public QuantWrapper {
public:
    /**
     * Create a RQ wrapper.
     * @param d Dimension of vectors
     * @param M Number of subquantizers
     * @param nbits Number of bits per subquantizer
     * @param metric Distance metric (L2 or IP)
     */
    RQWrapper(size_t d, size_t M, size_t nbits,
              faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), M_(M), nbits_(nbits), metric_(metric) {
        index_ = std::make_unique<faiss::IndexResidualQuantizer>(
            d, M, nbits, metric);
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
        return "RQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "M" << M_ << "_nbits" << nbits_;
        return oss.str();
    }

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return index_->ntotal; }

private:
    size_t d_;
    size_t M_;
    size_t nbits_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexResidualQuantizer> index_;
};

/**
 * Local Search Quantizer (LSQ) wrapper.
 * LSQ uses local search optimization to find better quantization codes.
 */
class LSQWrapper : public QuantWrapper {
public:
    /**
     * Create a LSQ wrapper.
     * @param d Dimension of vectors
     * @param M Number of subquantizers
     * @param nbits Number of bits per subquantizer
     * @param metric Distance metric (L2 or IP)
     */
    LSQWrapper(size_t d, size_t M, size_t nbits,
               faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), M_(M), nbits_(nbits), metric_(metric) {
        index_ = std::make_unique<faiss::IndexLocalSearchQuantizer>(
            d, M, nbits, metric);
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
        return "LSQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "M" << M_ << "_nbits" << nbits_;
        return oss.str();
    }

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return index_->ntotal; }

private:
    size_t d_;
    size_t M_;
    size_t nbits_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexLocalSearchQuantizer> index_;
};

/**
 * Product Residual Quantizer (PRQ) wrapper.
 * PRQ splits the vector into subvectors, each quantized by a separate RQ.
 */
class PRQWrapper : public QuantWrapper {
public:
    /**
     * Create a PRQ wrapper.
     * @param d Dimension of vectors
     * @param nsplits Number of splits (product components)
     * @param Msub Number of subquantizers per RQ
     * @param nbits Number of bits per subquantizer
     * @param metric Distance metric (L2 or IP)
     */
    PRQWrapper(size_t d, size_t nsplits, size_t Msub, size_t nbits,
               faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), nsplits_(nsplits), Msub_(Msub), nbits_(nbits), metric_(metric) {
        index_ = std::make_unique<faiss::IndexProductResidualQuantizer>(
            d, nsplits, Msub, nbits, metric);
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
        return "PRQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "ns" << nsplits_ << "_Msub" << Msub_ << "_nbits" << nbits_;
        return oss.str();
    }

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return index_->ntotal; }

private:
    size_t d_;
    size_t nsplits_;
    size_t Msub_;
    size_t nbits_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexProductResidualQuantizer> index_;
};

/**
 * Product Local Search Quantizer (PLSQ) wrapper.
 * PLSQ splits the vector into subvectors, each quantized by a separate LSQ.
 */
class PLSQWrapper : public QuantWrapper {
public:
    /**
     * Create a PLSQ wrapper.
     * @param d Dimension of vectors
     * @param nsplits Number of splits (product components)
     * @param Msub Number of subquantizers per LSQ
     * @param nbits Number of bits per subquantizer
     * @param metric Distance metric (L2 or IP)
     */
    PLSQWrapper(size_t d, size_t nsplits, size_t Msub, size_t nbits,
                faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), nsplits_(nsplits), Msub_(Msub), nbits_(nbits), metric_(metric) {
        index_ = std::make_unique<faiss::IndexProductLocalSearchQuantizer>(
            d, nsplits, Msub, nbits, metric);
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
        return "PLSQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "ns" << nsplits_ << "_Msub" << Msub_ << "_nbits" << nbits_;
        return oss.str();
    }

    size_t get_dimension() const override { return d_; }
    size_t get_ntotal() const override { return index_->ntotal; }

private:
    size_t d_;
    size_t nsplits_;
    size_t Msub_;
    size_t nbits_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexProductLocalSearchQuantizer> index_;
};

/*****************************************************
 * Factory functions
 *****************************************************/

/**
 * Factory for RQ wrapper.
 * Params: "M" (default 8), "nbits" (default 8)
 */
inline std::unique_ptr<QuantWrapper> create_rq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t M = 8;
    size_t nbits = 8;

    auto it = params.find("M");
    if (it != params.end()) M = std::stoul(it->second);

    it = params.find("nbits");
    if (it != params.end()) nbits = std::stoul(it->second);

    return std::make_unique<RQWrapper>(d, M, nbits, metric);
}

/**
 * Factory for LSQ wrapper.
 * Params: "M" (default 8), "nbits" (default 8)
 */
inline std::unique_ptr<QuantWrapper> create_lsq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t M = 8;
    size_t nbits = 8;

    auto it = params.find("M");
    if (it != params.end()) M = std::stoul(it->second);

    it = params.find("nbits");
    if (it != params.end()) nbits = std::stoul(it->second);

    return std::make_unique<LSQWrapper>(d, M, nbits, metric);
}

/**
 * Factory for PRQ wrapper.
 * Params: "nsplits" (default 2), "Msub" (default 4), "nbits" (default 8)
 */
inline std::unique_ptr<QuantWrapper> create_prq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t nsplits = 2;
    size_t Msub = 4;
    size_t nbits = 8;

    auto it = params.find("nsplits");
    if (it != params.end()) nsplits = std::stoul(it->second);

    it = params.find("Msub");
    if (it != params.end()) Msub = std::stoul(it->second);

    it = params.find("nbits");
    if (it != params.end()) nbits = std::stoul(it->second);

    return std::make_unique<PRQWrapper>(d, nsplits, Msub, nbits, metric);
}

/**
 * Factory for PLSQ wrapper.
 * Params: "nsplits" (default 2), "Msub" (default 4), "nbits" (default 8)
 */
inline std::unique_ptr<QuantWrapper> create_plsq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t nsplits = 2;
    size_t Msub = 4;
    size_t nbits = 8;

    auto it = params.find("nsplits");
    if (it != params.end()) nsplits = std::stoul(it->second);

    it = params.find("Msub");
    if (it != params.end()) Msub = std::stoul(it->second);

    it = params.find("nbits");
    if (it != params.end()) nbits = std::stoul(it->second);

    return std::make_unique<PLSQWrapper>(d, nsplits, Msub, nbits, metric);
}

} // namespace hnsw_bench
