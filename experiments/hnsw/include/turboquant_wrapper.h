/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/MetricType.h>

#include "../../TURBOQUANT/codes/reproduction/turbo_quant_index_v3.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hnsw_bench {

inline size_t default_turboquant_threads() {
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
    return threads > 0 ? static_cast<size_t>(threads) : size_t(1);
#else
    return 1;
#endif
}

inline turboquant::TurboQuantIndexV3::Mode turboquant_default_mode(
        faiss::MetricType metric) {
    return metric == faiss::METRIC_INNER_PRODUCT
            ? turboquant::TurboQuantIndexV3::Mode::kInnerProduct
            : turboquant::TurboQuantIndexV3::Mode::kMSE;
}

inline turboquant::TurboQuantIndexV3::SearchMetric turboquant_search_metric(
        faiss::MetricType metric) {
    if (metric == faiss::METRIC_L2) {
        return turboquant::TurboQuantIndexV3::SearchMetric::kL2;
    }
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        return turboquant::TurboQuantIndexV3::SearchMetric::kInnerProduct;
    }
    throw std::runtime_error(
            "TurboQuantWrapper supports METRIC_L2 and METRIC_INNER_PRODUCT only");
}

inline turboquant::TurboQuantIndexV3::Mode parse_turboquant_mode(
        const std::string& value,
        faiss::MetricType metric) {
    if (value.empty()) {
        return turboquant_default_mode(metric);
    }
    if (value == "mse" || value == "MSE" || value == "l2") {
        return turboquant::TurboQuantIndexV3::Mode::kMSE;
    }
    if (value == "ip" || value == "inner_product" || value == "INNER_PRODUCT") {
        return turboquant::TurboQuantIndexV3::Mode::kInnerProduct;
    }
    throw std::runtime_error("Unknown TurboQuant mode: " + value);
}

class TurboQuantWrapper : public QuantWrapper {
   public:
    TurboQuantWrapper(
            size_t d,
            size_t bits = 4,
            uint64_t seed = 1234,
            turboquant::TurboQuantIndexV3::Mode mode =
                    turboquant::TurboQuantIndexV3::Mode::kMSE,
            size_t num_threads = default_turboquant_threads(),
            faiss::MetricType metric = faiss::METRIC_L2)
            : d_(d),
              bits_(bits),
              seed_(seed),
              mode_(mode),
              num_threads_(num_threads == 0 ? 1 : num_threads),
              metric_(metric),
              search_metric_(turboquant_search_metric(metric)),
              index_(turboquant::TurboQuantIndexV3::Config{
                      d_,
                      bits_,
                      mode_,
                      turboquant::TurboQuantIndexV3::RotationType::kHadamard,
                      seed_,
                      num_threads_,
              }) {
        if (d_ == 0) {
            throw std::runtime_error("TurboQuantWrapper: dimension must be > 0");
        }
        if (bits_ == 0 || bits_ > 9) {
            throw std::runtime_error("TurboQuantWrapper: bits must be in [1, 9]");
        }
    }

    void train(size_t n, const float* x) override {
        if (n == 0 || x == nullptr) {
            throw std::runtime_error(
                    "TurboQuantWrapper::train requires non-empty input");
        }
        index_.train(n, x);
    }

    void add(size_t n, const float* x) override {
        if (n == 0 || x == nullptr) {
            throw std::runtime_error(
                    "TurboQuantWrapper::add requires non-empty input");
        }
        index_.add(n, x);
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
#if TURBOQUANT_HAS_FAISS_DISTANCE_COMPUTER
        return index_.get_distance_computer(search_metric_);
#else
        throw std::runtime_error(
                "TurboQuantWrapper requires faiss::DistanceComputer support");
#endif
    }

    std::string get_name() const override {
        return "TurboQuantV3";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "bits" << bits_ << "_seed" << seed_;
        if (mode_ != turboquant_default_mode(metric_)) {
            oss << "_mode"
                << (mode_ == turboquant::TurboQuantIndexV3::Mode::kInnerProduct
                            ? "ip"
                            : "mse");
        }
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return index_.ntotal();
    }

    // V3 has no serialization support
    bool save(const std::string& /*path*/) override {
        return false;
    }

    bool load(const std::string& /*path*/) override {
        return false;
    }

   private:
    size_t d_;
    size_t bits_;
    uint64_t seed_;
    turboquant::TurboQuantIndexV3::Mode mode_;
    size_t num_threads_;
    faiss::MetricType metric_;
    turboquant::TurboQuantIndexV3::SearchMetric search_metric_;
    turboquant::TurboQuantIndexV3 index_;
};

inline std::unique_ptr<QuantWrapper> create_turboquant_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t bits = 4;
    uint64_t seed = 1234;
    size_t threads = default_turboquant_threads();
    std::string mode_str;

    auto it = params.find("bits");
    if (it != params.end()) {
        bits = std::stoul(it->second);
    }

    it = params.find("seed");
    if (it != params.end()) {
        seed = std::stoull(it->second);
    }

    it = params.find("threads");
    if (it != params.end()) {
        threads = std::stoul(it->second);
    }

    it = params.find("mode");
    if (it != params.end()) {
        mode_str = it->second;
    }

    return std::make_unique<TurboQuantWrapper>(
            d,
            bits,
            seed,
            parse_turboquant_mode(mode_str, metric),
            threads,
            metric);
}

} // namespace hnsw_bench
