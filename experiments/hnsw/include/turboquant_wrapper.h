/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "quant_wrapper.h"

#include <faiss/MetricType.h>

#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace hnsw_bench {

class TurboQuantWrapper;

class TurboQuantDistanceComputer : public faiss::DistanceComputer {
   public:
    explicit TurboQuantDistanceComputer(const TurboQuantWrapper* parent);

    void set_query(const float* x) override;

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;

   private:
    const TurboQuantWrapper* parent_;
    std::vector<float> rotated_query_;
    std::vector<float> lut_;
    float query_norm_sq_ = 0.0f;
};

class TurboQuantWrapper : public QuantWrapper {
   public:
    TurboQuantWrapper(
            size_t d,
            size_t bits = 4,
            uint32_t seed = 1234,
            faiss::MetricType metric = faiss::METRIC_L2);

    void train(size_t n, const float* x) override;

    void add(size_t n, const float* x) override;

    std::unique_ptr<faiss::DistanceComputer> get_distance_computer() override;

    std::string get_name() const override;

    std::string get_params_string() const override;

    size_t get_dimension() const override;

    size_t get_ntotal() const override;

    bool save(const std::string& path) override;

    bool load(const std::string& path) override;

    void rotate(const float* x, float* y) const;

    uint8_t quantize_scalar(float value) const;

    size_t get_levels() const;

    const std::vector<float>& centroids() const;

    const std::vector<uint8_t>& codes() const;

    const std::vector<float>& norms() const;

    const std::vector<float>& scaled_code_norm_sq() const;

   private:
    static constexpr float kLloydGridLo = -8.0f;
    static constexpr float kLloydGridHi = 8.0f;
    static constexpr size_t kLloydGridSize = 32768;

    template <typename T>
    static void write_value(std::ofstream& ofs, const T& value);

    template <typename T>
    static void read_value(std::ifstream& ifs, T& value);

    template <typename T>
    static void write_vector(std::ofstream& ofs, const std::vector<T>& values);

    template <typename T>
    static void read_vector(std::ifstream& ifs, std::vector<T>& values);

    static float gaussian_pdf(float x);

    void build_rotation_matrix();

    void build_codebook();

    static std::vector<float> solve_standard_normal_lloyd_max(size_t bits);

    size_t d_;
    size_t bits_;
    uint32_t seed_;
    faiss::MetricType metric_;
    size_t nlevels_ = 0;
    size_t ntotal_ = 0;
    bool is_trained_ = false;

    std::vector<float> centroids_;
    std::vector<float> boundaries_;
    std::vector<float> rotation_matrix_;
    std::vector<uint8_t> codes_;
    std::vector<float> norms_;
    std::vector<float> scaled_code_norm_sq_;

    friend class TurboQuantDistanceComputer;
};

std::unique_ptr<QuantWrapper> create_turboquant_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params);

} // namespace hnsw_bench
