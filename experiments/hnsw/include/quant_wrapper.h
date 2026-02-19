/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/index_io.h>
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

    /**
     * Get the underlying faiss::Index pointer for serialization.
     * Returns nullptr if the wrapper does not support serialization.
     */
    virtual faiss::Index* get_faiss_index() { return nullptr; }

    /**
     * Save the quantizer index to disk.
     * Default: uses faiss::write_index via get_faiss_index().
     * Non-faiss quantizers (e.g., RaBitQ) should override.
     * @return true if saved successfully.
     */
    virtual bool save(const std::string& path) {
        faiss::Index* idx = get_faiss_index();
        if (!idx) {
            std::cerr << "Warning: quantizer does not support serialization" << std::endl;
            return false;
        }
        faiss::write_index(idx, path.c_str());
        return true;
    }

    /**
     * Load the quantizer index from disk.
     * The wrapper must already be constructed with correct params.
     * @return true if loaded successfully, false if file doesn't exist or fails.
     */
    virtual bool load(const std::string& path) {
        return false;  // subclasses override
    }
};

/**
 * Factory function type for creating quantization wrappers.
 */
using QuantWrapperFactory = std::function<std::unique_ptr<QuantWrapper>(
    size_t d,                           // dimension
    faiss::MetricType metric,           // L2 or IP
    const std::map<std::string, std::string>& params  // algorithm-specific params
)>;

/*****************************************************
 * Quantizer index save/load helpers
 *****************************************************/

/**
 * Generate save path for a quantizer index.
 */
inline std::string get_quant_index_path(
        const std::string& base_path,
        const std::string& algorithm,
        const std::string& params_str) {
    return base_path + "/index/quant_" + algorithm + "_" + params_str + ".faiss";
}

/**
 * Helper template for faiss-based wrappers to implement load().
 * Reads a faiss::Index from disk, dynamic_casts to IndexType, and
 * replaces the wrapper's internal index pointer.
 * @return true if file exists and loads successfully with correct type.
 */
template <typename IndexType>
bool load_faiss_index(const std::string& path, std::unique_ptr<IndexType>& index) {
    std::ifstream f(path);
    if (!f.good()) return false;
    f.close();

    faiss::Index* idx = faiss::read_index(path.c_str());
    auto* typed = dynamic_cast<IndexType*>(idx);
    if (!typed) {
        delete idx;
        return false;
    }
    index.reset(typed);
    return true;
}

} // namespace hnsw_bench
