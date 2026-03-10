#pragma once

#include "OptimizedScalarQuantizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace osq {

using idx_t = long long;

struct SearchResult {
  float score;
  idx_t id;
};

class OSQIndex {
 public:
  struct EncodedQuery {
    std::vector<uint8_t> q;
    QuantizationResult corr;
  };

  OSQIndex(size_t dims, Similarity similarity, ScalarEncoding encoding);

  void train(size_t n, const float* x);
  void add(size_t n, const float* x);

  void search(size_t nq, const float* queries, size_t k, float* distances, idx_t* labels) const;
  EncodedQuery encode_query(const float* q) const;
  float score(const EncodedQuery& query, idx_t id) const;
  float score(const float* query, idx_t id) const;
  bool save(const std::string& path) const;
  bool load(const std::string& path);

  void set_num_threads(int n);
  int num_threads() const { return num_threads_; }

  size_t ntotal() const { return docs_.size(); }
  size_t d() const { return dims_; }

  const std::vector<float>& centroid() const { return centroid_; }

 private:
  EncodedVector encode_doc(const float* x) const;
  float score_query_doc(const EncodedQuery& q, const EncodedVector& x) const;

  static void l2_normalize(std::vector<float>& v);
  static float scale_max_inner_product_score(float s);

  static int dot_uint8(const uint8_t* a, const uint8_t* b, size_t n);
  static int dot_int4_with_packed_doc(const uint8_t* q, const uint8_t* packed_doc, size_t dims);
  static int dot_int4_with_binary_doc_transposed(
      const uint8_t* q_transposed,
      const uint8_t* binary_doc,
      size_t dims);
  static int dot_int4_with_dibit_doc_transposed(
      const uint8_t* q_transposed,
      const uint8_t* dibit_doc,
      size_t dims);

  size_t dims_;
  Similarity similarity_;
  ScalarEncoding encoding_;
  bool trained_;
  int num_threads_;

  std::vector<float> centroid_;
  float centroid_dp_;

  OptimizedScalarQuantizer quantizer_;
  std::vector<EncodedVector> docs_;
};

} // namespace osq
