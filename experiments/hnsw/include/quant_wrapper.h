/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <faiss/impl/DistanceComputer.h>

namespace hnsw_bench {

/**
 * Base class for quantization method wrappers.
 * Provides a unified interface for different quantization methods (PQ, SQ, etc.)
 */
class QuantWrapper {
public:
    virtual ~QuantWrapper() = default;

    /**
     * Train the quantizer on the given data.
     * @param n Number of training vectors
     * @param x Training data (n * d floats)
     */
    virtual void train(size_t n, const float* x) = 0;

    /**
     * Add vectors to the quantizer (encode them).
     * @param n Number of vectors to add
     * @param x Vector data (n * d floats)
     */
    virtual void add(size_t n, const float* x) = 0;

    /**
     * Get a distance computer for this quantizer.
     * The returned object is NOT thread-safe - each thread should get its own.
     * @return A new distance computer instance
     */
    virtual std::unique_ptr<faiss::DistanceComputer> get_distance_computer() = 0;

    /**
     * Get the name of this quantization method.
     * @return Method name (e.g., "PQ", "SQ")
     */
    virtual std::string get_name() const = 0;

    /**
     * Get a string representation of the current parameters.
     * @return Parameter string (e.g., "M=32_nbits=8")
     */
    virtual std::string get_params_string() const = 0;

    /**
     * Get the dimension of vectors.
     */
    virtual size_t get_dimension() const = 0;

    /**
     * Get the number of vectors added.
     */
    virtual size_t get_ntotal() const = 0;
};

/**
 * Factory function type for creating quantization wrappers.
 */
using QuantWrapperFactory = std::function<std::unique_ptr<QuantWrapper>(
    size_t d,                           // dimension
    faiss::MetricType metric,           // L2 or IP
    const std::map<std::string, std::string>& params  // algorithm-specific params
)>;

} // namespace hnsw_bench
