/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include "OSQIndex.h"

#include <faiss/impl/DistanceComputer.h>

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace hnsw_bench {

class OSQDistanceComputer : public faiss::DistanceComputer {
public:
    explicit OSQDistanceComputer(const osq::OSQIndex* index)
        : index_(index) {}

    void set_query(const float* x) override {
        query_ = index_->encode_query(x);
    }

    float operator()(faiss::idx_t i) override {
        if (!query_.has_value()) {
            throw std::runtime_error("OSQDistanceComputer query is not set");
        }
        // HNSW expects lower values to be better. OSQ returns a normalized
        // similarity score in [0, 1], so convert it to a monotonic distance.
        return 1.0f - index_->score(*query_, i);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        if (i == j) {
            return 0.0f;
        }
        return 1.0f;
    }

private:
    const osq::OSQIndex* index_;
    std::optional<osq::OSQIndex::EncodedQuery> query_;
};

class OSQWrapper : public QuantWrapper {
public:
    OSQWrapper(
            size_t d,
            osq::Similarity similarity,
            osq::ScalarEncoding encoding,
            int num_threads = 0)
        : d_(d)
        , similarity_(similarity)
        , encoding_(encoding)
        , num_threads_(num_threads > 0 ? num_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency())))
        , index_(std::make_unique<osq::OSQIndex>(d, similarity, encoding)) {
        index_->set_num_threads(num_threads_);
    }

    void train(size_t n, const float* x) override {
        index_->train(n, x);
    }

    void add(size_t n, const float* x) override {
        index_->add(n, x);
    }

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override {
        return std::make_unique<OSQDistanceComputer>(index_.get());
    }

    std::string get_name() const override {
        return "OSQ";
    }

    std::string get_params_string() const override {
        std::ostringstream oss;
        oss << "encoding" << encoding_to_string(encoding_)
            << "_similarity" << similarity_to_string(similarity_);
        return oss.str();
    }

    size_t get_dimension() const override {
        return d_;
    }

    size_t get_ntotal() const override {
        return index_->ntotal();
    }

    bool save(const std::string& path) override {
        return index_->save(path);
    }

    bool load(const std::string& path) override {
        return index_->load(path);
    }

    static std::string similarity_to_string(osq::Similarity similarity) {
        switch (similarity) {
            case osq::Similarity::EUCLIDEAN: return "EUCLIDEAN";
            case osq::Similarity::COSINE: return "COSINE";
            case osq::Similarity::DOT_PRODUCT: return "DOT_PRODUCT";
            case osq::Similarity::MAX_INNER_PRODUCT: return "MAX_INNER_PRODUCT";
        }
        throw std::runtime_error("Unknown OSQ similarity");
    }

    static osq::Similarity string_to_similarity(const std::string& value) {
        if (value == "EUCLIDEAN" || value == "l2" || value == "L2") {
            return osq::Similarity::EUCLIDEAN;
        }
        if (value == "COSINE" || value == "cosine") {
            return osq::Similarity::COSINE;
        }
        if (value == "DOT_PRODUCT" || value == "dot" || value == "ip") {
            return osq::Similarity::DOT_PRODUCT;
        }
        if (value == "MAX_INNER_PRODUCT" || value == "mip") {
            return osq::Similarity::MAX_INNER_PRODUCT;
        }
        throw std::runtime_error("Unknown OSQ similarity: " + value);
    }

    static std::string encoding_to_string(osq::ScalarEncoding encoding) {
        switch (encoding) {
            case osq::ScalarEncoding::UNSIGNED_BYTE: return "UNSIGNED_BYTE";
            case osq::ScalarEncoding::PACKED_NIBBLE: return "PACKED_NIBBLE";
            case osq::ScalarEncoding::SEVEN_BIT: return "SEVEN_BIT";
            case osq::ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE: return "SINGLE_BIT_QUERY_NIBBLE";
            case osq::ScalarEncoding::DIBIT_QUERY_NIBBLE: return "DIBIT_QUERY_NIBBLE";
        }
        throw std::runtime_error("Unknown OSQ encoding");
    }

    static osq::ScalarEncoding string_to_encoding(const std::string& value) {
        if (value == "UNSIGNED_BYTE" || value == "8bit") {
            return osq::ScalarEncoding::UNSIGNED_BYTE;
        }
        if (value == "PACKED_NIBBLE" || value == "4bit") {
            return osq::ScalarEncoding::PACKED_NIBBLE;
        }
        if (value == "SEVEN_BIT" || value == "7bit") {
            return osq::ScalarEncoding::SEVEN_BIT;
        }
        if (value == "SINGLE_BIT_QUERY_NIBBLE" || value == "1bit_query_nibble") {
            return osq::ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE;
        }
        if (value == "DIBIT_QUERY_NIBBLE" || value == "2bit_query_nibble") {
            return osq::ScalarEncoding::DIBIT_QUERY_NIBBLE;
        }
        throw std::runtime_error("Unknown OSQ encoding: " + value);
    }

private:
    size_t d_;
    osq::Similarity similarity_;
    osq::ScalarEncoding encoding_;
    int num_threads_;
    std::unique_ptr<osq::OSQIndex> index_;
};

inline std::unique_ptr<QuantWrapper> create_osq_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    osq::Similarity similarity = (metric == faiss::METRIC_INNER_PRODUCT)
        ? osq::Similarity::DOT_PRODUCT
        : osq::Similarity::EUCLIDEAN;
    osq::ScalarEncoding encoding = osq::ScalarEncoding::PACKED_NIBBLE;
    int threads = 1;

    auto sim_it = params.find("similarity");
    if (sim_it != params.end()) {
        similarity = OSQWrapper::string_to_similarity(sim_it->second);
    }

    auto enc_it = params.find("encoding");
    if (enc_it != params.end()) {
        encoding = OSQWrapper::string_to_encoding(enc_it->second);
    }

    auto thread_it = params.find("threads");
    if (thread_it != params.end()) {
        threads = std::stoi(thread_it->second);
    }

    return std::make_unique<OSQWrapper>(d, similarity, encoding, threads);
}

} // namespace hnsw_bench
