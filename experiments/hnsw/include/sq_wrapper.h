/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/DistanceComputer.h>

#include <sstream>
#include <stdexcept>

namespace hnsw_bench {

/**
 * Scalar Quantization wrapper.
 * Wraps FAISS IndexScalarQuantizer for use in HNSW benchmarks.
 */
class SQWrapper : public QuantWrapper {
public:
    /**
     * Create an SQ wrapper.
     * @param d Dimension of vectors
     * @param qtype Quantizer type (e.g., QT_8bit, QT_4bit, QT_fp16)
     * @param metric Distance metric (L2 or IP)
     */
    SQWrapper(size_t d, faiss::ScalarQuantizer::QuantizerType qtype,
              faiss::MetricType metric = faiss::METRIC_L2)
        : d_(d), qtype_(qtype), metric_(metric) {
        index_ = std::make_unique<faiss::IndexScalarQuantizer>(d, qtype, metric);
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
        return "SQ";
    }

    std::string get_params_string() const override {
        return qtype_to_string(qtype_);
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return index_->ntotal;
    }

    // SQ-specific accessors
    faiss::ScalarQuantizer::QuantizerType get_qtype() const { return qtype_; }
    faiss::IndexScalarQuantizer* get_index() { return index_.get(); }

    // Helper to convert qtype to string
    static std::string qtype_to_string(faiss::ScalarQuantizer::QuantizerType qtype) {
        switch (qtype) {
            case faiss::ScalarQuantizer::QT_8bit: return "QT_8bit";
            case faiss::ScalarQuantizer::QT_4bit: return "QT_4bit";
            case faiss::ScalarQuantizer::QT_8bit_uniform: return "QT_8bit_uniform";
            case faiss::ScalarQuantizer::QT_4bit_uniform: return "QT_4bit_uniform";
            case faiss::ScalarQuantizer::QT_fp16: return "QT_fp16";
            case faiss::ScalarQuantizer::QT_8bit_direct: return "QT_8bit_direct";
            case faiss::ScalarQuantizer::QT_6bit: return "QT_6bit";
            case faiss::ScalarQuantizer::QT_8bit_direct_signed: return "QT_8bit_direct_signed";
            case faiss::ScalarQuantizer::QT_bf16: return "QT_bf16";
            default: return "unknown";
        }
    }

    // Helper to parse qtype from string
    static faiss::ScalarQuantizer::QuantizerType string_to_qtype(const std::string& s) {
        if (s == "QT_8bit" || s == "8bit") return faiss::ScalarQuantizer::QT_8bit;
        if (s == "QT_4bit" || s == "4bit") return faiss::ScalarQuantizer::QT_4bit;
        if (s == "QT_8bit_uniform") return faiss::ScalarQuantizer::QT_8bit_uniform;
        if (s == "QT_4bit_uniform") return faiss::ScalarQuantizer::QT_4bit_uniform;
        if (s == "QT_fp16" || s == "fp16") return faiss::ScalarQuantizer::QT_fp16;
        if (s == "QT_8bit_direct") return faiss::ScalarQuantizer::QT_8bit_direct;
        if (s == "QT_6bit" || s == "6bit") return faiss::ScalarQuantizer::QT_6bit;
        if (s == "QT_8bit_direct_signed") return faiss::ScalarQuantizer::QT_8bit_direct_signed;
        if (s == "QT_bf16" || s == "bf16") return faiss::ScalarQuantizer::QT_bf16;
        throw std::runtime_error("Unknown SQ quantizer type: " + s);
    }

private:
    size_t d_;
    faiss::ScalarQuantizer::QuantizerType qtype_;
    faiss::MetricType metric_;
    std::unique_ptr<faiss::IndexScalarQuantizer> index_;
};

/**
 * Factory function to create SQWrapper from parameter map.
 * Expected params: "qtype" (required, e.g., "QT_8bit", "QT_4bit")
 */
inline std::unique_ptr<QuantWrapper> create_sq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {

    auto qtype = faiss::ScalarQuantizer::QT_8bit;  // default

    auto it = params.find("qtype");
    if (it != params.end()) {
        qtype = SQWrapper::string_to_qtype(it->second);
    }

    return std::make_unique<SQWrapper>(d, qtype, metric);
}

} // namespace hnsw_bench
