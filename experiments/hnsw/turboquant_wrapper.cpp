/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/turboquant_wrapper.h"

#include <faiss/VectorTransform.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hnsw_bench {

TurboQuantDistanceComputer::TurboQuantDistanceComputer(
        const TurboQuantWrapper* parent)
        : parent_(parent),
          rotated_query_(parent->d_),
          lut_(parent->d_ * parent->nlevels_) {}

void TurboQuantDistanceComputer::set_query(const float* x) {
    parent_->rotate(x, rotated_query_.data());
    query_norm_sq_ = 0.0f;
    for (size_t j = 0; j < parent_->d_; ++j) {
        const float qv = rotated_query_[j];
        query_norm_sq_ += qv * qv;
        const size_t lut_base = j * parent_->nlevels_;
        for (size_t level = 0; level < parent_->nlevels_; ++level) {
            lut_[lut_base + level] = qv * parent_->centroids_[level];
        }
    }
}

float TurboQuantDistanceComputer::operator()(faiss::idx_t i) {
    const uint8_t* code = parent_->codes_.data() + static_cast<size_t>(i) * parent_->d_;

    float dot = 0.0f;
    for (size_t j = 0; j < parent_->d_; ++j) {
        dot += lut_[j * parent_->nlevels_ + code[j]];
    }

    return query_norm_sq_ + parent_->scaled_code_norm_sq_[i] -
            2.0f * parent_->norms_[i] * dot;
}

float TurboQuantDistanceComputer::symmetric_dis(faiss::idx_t, faiss::idx_t) {
    throw std::runtime_error(
            "TurboQuantDistanceComputer::symmetric_dis not implemented");
}

TurboQuantWrapper::TurboQuantWrapper(
        size_t d,
        size_t bits,
        uint32_t seed,
        faiss::MetricType metric)
        : d_(d), bits_(bits), seed_(seed), metric_(metric) {
    if (d_ == 0) {
        throw std::runtime_error("TurboQuantWrapper: dimension must be > 0");
    }
    if (bits_ == 0 || bits_ > 8) {
        throw std::runtime_error("TurboQuantWrapper: bits must be in [1, 8]");
    }
    if (metric_ != faiss::METRIC_L2) {
        throw std::runtime_error(
                "TurboQuantWrapper currently supports METRIC_L2 only");
    }
    nlevels_ = size_t(1) << bits_;
}

void TurboQuantWrapper::train(size_t n, const float* x) {
    if (n == 0 || x == nullptr) {
        throw std::runtime_error("TurboQuantWrapper::train requires non-empty input");
    }

    build_rotation_matrix();
    build_codebook();
    is_trained_ = true;
}

void TurboQuantWrapper::add(size_t n, const float* x) {
    if (!is_trained_) {
        throw std::runtime_error("TurboQuantWrapper::add called before train");
    }
    if (n == 0 || x == nullptr) {
        throw std::runtime_error("TurboQuantWrapper::add requires non-empty input");
    }

    ntotal_ = n;
    codes_.assign(n * d_, uint8_t(0));
    norms_.assign(n, 0.0f);
    scaled_code_norm_sq_.assign(n, 0.0f);

    std::vector<float> unit(d_);
    std::vector<float> rotated(d_);

    for (size_t i = 0; i < n; ++i) {
        const float* xi = x + i * d_;
        float norm_sq = 0.0f;
        for (size_t j = 0; j < d_; ++j) {
            norm_sq += xi[j] * xi[j];
        }
        float norm = std::sqrt(norm_sq);
        norms_[i] = norm;

        if (norm > 0.0f) {
            const float inv_norm = 1.0f / norm;
            for (size_t j = 0; j < d_; ++j) {
                unit[j] = xi[j] * inv_norm;
            }
        } else {
            std::fill(unit.begin(), unit.end(), 0.0f);
        }

        rotate(unit.data(), rotated.data());

        float code_norm_sq = 0.0f;
        uint8_t* code_ptr = codes_.data() + i * d_;
        for (size_t j = 0; j < d_; ++j) {
            const uint8_t code = quantize_scalar(rotated[j]);
            code_ptr[j] = code;
            const float c = centroids_[code];
            code_norm_sq += c * c;
        }
        scaled_code_norm_sq_[i] = norm_sq * code_norm_sq;
    }
}

std::unique_ptr<faiss::DistanceComputer> TurboQuantWrapper::get_distance_computer() {
    if (!is_trained_) {
        throw std::runtime_error(
                "TurboQuantWrapper::get_distance_computer called before train/load");
    }
    return std::make_unique<TurboQuantDistanceComputer>(this);
}

std::string TurboQuantWrapper::get_name() const {
    return "TurboQuant";
}

std::string TurboQuantWrapper::get_params_string() const {
    std::ostringstream oss;
    oss << "bits" << bits_ << "_seed" << seed_;
    return oss.str();
}

size_t TurboQuantWrapper::get_dimension() const {
    return d_;
}

size_t TurboQuantWrapper::get_ntotal() const {
    return ntotal_;
}

bool TurboQuantWrapper::save(const std::string& path) {
    if (!is_trained_) {
        return false;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }

    const char magic[8] = {'T', 'Q', 'W', 'R', 'A', 'P', '1', '\0'};
    ofs.write(magic, sizeof(magic));

    write_value(ofs, d_);
    write_value(ofs, bits_);
    write_value(ofs, seed_);
    write_value(ofs, nlevels_);
    write_value(ofs, ntotal_);

    write_vector(ofs, centroids_);
    write_vector(ofs, boundaries_);
    write_vector(ofs, rotation_matrix_);
    write_vector(ofs, norms_);
    write_vector(ofs, scaled_code_norm_sq_);
    write_vector(ofs, codes_);

    return ofs.good();
}

bool TurboQuantWrapper::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    char magic[8];
    ifs.read(magic, sizeof(magic));
    const char expected[8] = {'T', 'Q', 'W', 'R', 'A', 'P', '1', '\0'};
    if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
        return false;
    }

    size_t d = 0;
    size_t bits = 0;
    uint32_t seed = 0;
    size_t nlevels = 0;
    size_t ntotal = 0;

    read_value(ifs, d);
    read_value(ifs, bits);
    read_value(ifs, seed);
    read_value(ifs, nlevels);
    read_value(ifs, ntotal);

    if (!ifs.good() || d != d_ || bits != bits_) {
        return false;
    }

    seed_ = seed;
    nlevels_ = nlevels;
    ntotal_ = ntotal;

    read_vector(ifs, centroids_);
    read_vector(ifs, boundaries_);
    read_vector(ifs, rotation_matrix_);
    read_vector(ifs, norms_);
    read_vector(ifs, scaled_code_norm_sq_);
    read_vector(ifs, codes_);

    if (!ifs.good()) {
        return false;
    }

    if (centroids_.size() != nlevels_ || boundaries_.size() + 1 != nlevels_ ||
        rotation_matrix_.size() != d_ * d_ || norms_.size() != ntotal_ ||
        scaled_code_norm_sq_.size() != ntotal_ || codes_.size() != ntotal_ * d_) {
        return false;
    }

    is_trained_ = true;
    return true;
}

void TurboQuantWrapper::rotate(const float* x, float* y) const {
    for (size_t i = 0; i < d_; ++i) {
        const float* row = rotation_matrix_.data() + i * d_;
        float sum = 0.0f;
        for (size_t j = 0; j < d_; ++j) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

uint8_t TurboQuantWrapper::quantize_scalar(float value) const {
    const auto it = std::upper_bound(boundaries_.begin(), boundaries_.end(), value);
    return static_cast<uint8_t>(it - boundaries_.begin());
}

size_t TurboQuantWrapper::get_levels() const {
    return nlevels_;
}

const std::vector<float>& TurboQuantWrapper::centroids() const {
    return centroids_;
}

const std::vector<uint8_t>& TurboQuantWrapper::codes() const {
    return codes_;
}

const std::vector<float>& TurboQuantWrapper::norms() const {
    return norms_;
}

const std::vector<float>& TurboQuantWrapper::scaled_code_norm_sq() const {
    return scaled_code_norm_sq_;
}

template <typename T>
void TurboQuantWrapper::write_value(std::ofstream& ofs, const T& value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void TurboQuantWrapper::read_value(std::ifstream& ifs, T& value) {
    ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void TurboQuantWrapper::write_vector(
        std::ofstream& ofs,
        const std::vector<T>& values) {
    size_t size = values.size();
    write_value(ofs, size);
    if (size > 0) {
        ofs.write(reinterpret_cast<const char*>(values.data()), sizeof(T) * size);
    }
}

template <typename T>
void TurboQuantWrapper::read_vector(
        std::ifstream& ifs,
        std::vector<T>& values) {
    size_t size = 0;
    read_value(ifs, size);
    values.resize(size);
    if (size > 0) {
        ifs.read(reinterpret_cast<char*>(values.data()), sizeof(T) * size);
    }
}

float TurboQuantWrapper::gaussian_pdf(float x) {
    static constexpr float inv_sqrt_2pi = 0.3989422804014327f;
    return inv_sqrt_2pi * std::exp(-0.5f * x * x);
}

void TurboQuantWrapper::build_rotation_matrix() {
    faiss::RandomRotationMatrix rrm(d_, d_);
    rrm.init(static_cast<int>(seed_));
    rotation_matrix_ = rrm.A;
}

void TurboQuantWrapper::build_codebook() {
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d_));
    const std::vector<float> standard_centroids =
            solve_standard_normal_lloyd_max(bits_);

    centroids_.resize(standard_centroids.size());
    for (size_t i = 0; i < standard_centroids.size(); ++i) {
        centroids_[i] = standard_centroids[i] * sigma;
    }

    boundaries_.resize(centroids_.size() - 1);
    for (size_t i = 0; i + 1 < centroids_.size(); ++i) {
        boundaries_[i] = 0.5f * (centroids_[i] + centroids_[i + 1]);
    }
}

std::vector<float> TurboQuantWrapper::solve_standard_normal_lloyd_max(size_t bits) {
    const size_t nlevels = size_t(1) << bits;
    const float step =
            (kLloydGridHi - kLloydGridLo) / static_cast<float>(kLloydGridSize - 1);

    std::vector<float> xs(kLloydGridSize);
    std::vector<float> weights(kLloydGridSize);
    for (size_t i = 0; i < kLloydGridSize; ++i) {
        xs[i] = kLloydGridLo + step * static_cast<float>(i);
        weights[i] = gaussian_pdf(xs[i]);
    }

    std::vector<float> centroids(nlevels);
    for (size_t i = 0; i < nlevels; ++i) {
        const float alpha =
                (static_cast<float>(i) + 0.5f) / static_cast<float>(nlevels);
        centroids[i] = kLloydGridLo + alpha * (kLloydGridHi - kLloydGridLo);
    }

    std::vector<float> boundaries(nlevels - 1);
    for (size_t iter = 0; iter < 200; ++iter) {
        for (size_t i = 0; i + 1 < nlevels; ++i) {
            boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
        }

        std::vector<float> next = centroids;
        size_t begin = 0;
        float max_shift = 0.0f;
        for (size_t level = 0; level < nlevels; ++level) {
            size_t end = begin;
            const float upper = (level + 1 < nlevels)
                    ? boundaries[level]
                    : std::numeric_limits<float>::infinity();
            while (end < kLloydGridSize && xs[end] <= upper) {
                ++end;
            }

            double denom = 0.0;
            double numer = 0.0;
            for (size_t idx = begin; idx < end; ++idx) {
                denom += weights[idx];
                numer += static_cast<double>(xs[idx]) * weights[idx];
            }
            if (denom > 0.0) {
                next[level] = static_cast<float>(numer / denom);
            }

            max_shift = std::max(max_shift, std::fabs(next[level] - centroids[level]));
            begin = end;
        }

        centroids.swap(next);
        if (max_shift < 1e-6f) {
            break;
        }
    }

    return centroids;
}

std::unique_ptr<QuantWrapper> create_turboquant_wrapper(
        size_t d,
        faiss::MetricType metric,
        const std::map<std::string, std::string>& params) {
    size_t bits = 4;
    uint32_t seed = 1234;

    auto it = params.find("bits");
    if (it != params.end()) {
        bits = std::stoul(it->second);
    }

    it = params.find("seed");
    if (it != params.end()) {
        seed = static_cast<uint32_t>(std::stoul(it->second));
    }

    return std::make_unique<TurboQuantWrapper>(d, bits, seed, metric);
}

template void TurboQuantWrapper::write_value<size_t>(
        std::ofstream&,
        const size_t&);
template void TurboQuantWrapper::write_value<uint32_t>(
        std::ofstream&,
        const uint32_t&);
template void TurboQuantWrapper::write_value<float>(
        std::ofstream&,
        const float&);
template void TurboQuantWrapper::read_value<size_t>(std::ifstream&, size_t&);
template void TurboQuantWrapper::read_value<uint32_t>(std::ifstream&, uint32_t&);
template void TurboQuantWrapper::read_value<float>(std::ifstream&, float&);
template void TurboQuantWrapper::write_vector<float>(
        std::ofstream&,
        const std::vector<float>&);
template void TurboQuantWrapper::write_vector<uint8_t>(
        std::ofstream&,
        const std::vector<uint8_t>&);
template void TurboQuantWrapper::read_vector<float>(
        std::ifstream&,
        std::vector<float>&);
template void TurboQuantWrapper::read_vector<uint8_t>(
        std::ifstream&,
        std::vector<uint8_t>&);

} // namespace hnsw_bench
