#pragma once

#include <bits/stdc++.h>
#include <cstddef>
#include <immintrin.h>

#include <cstdint>
#include <memory>

#include "utils/memory.hpp"
#include "utils/space.hpp"

namespace saqlib::utils {

template <size_t kBits>
class CodeHelper {
    template <typename T>
    static void froce_compact(uint8_t *o_compact, T *o_raw, size_t num_dim) {
        for (size_t i = 0; i < num_dim * kBits / 8; i++) {
            o_compact[i] = 0;
        }
        size_t shift = 0;
        for (size_t d = 0; d < num_dim; d += 1) {
            auto t = o_raw[d];
            for (size_t i = 0; i < kBits; i++, t >>= 1) {
                o_compact[i] |= (t & 1) << shift;
            }
            ++shift;
            if (shift == 8) {
                shift = 0;
                o_compact += kBits;
            }
        }
        assert(shift == 0);
    }

    template <typename T>
    static void froce_decompact(const uint8_t *__restrict__ y, T *out, size_t D) {
        if (kBits == 0)
            return;
        uint8_t shift_v = 1;
        size_t y_p = 0;
        for (size_t d = 0; d < D; d++) {
            out[d] = 0;
            for (size_t i = 0; i < kBits; i++) {
                out[d] |= ((y[y_p + i] & shift_v) != 0) << i;
            }
            shift_v <<= 1;
            if (shift_v == 0) {
                shift_v = 1;
                y_p += kBits;
            }
        }
        assert(y_p == D * kBits / 8);
    }

  public:
    static void compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
        froce_compact(o_compact, o_raw, num_dim);
    }

    static void compacted_code16(uint8_t *o_compact, const uint16_t *o_raw16, size_t num_dim) {
        if constexpr (kBits == 0) {
            return;
        }
        auto o_raw8 = std::make_unique<uint8_t[]>(num_dim);
        if (kBits > 8) {
            for (size_t i = 0; i < num_dim; i++) {
                o_compact[i] = o_raw16[i] & 0xFF;
            }
            o_compact += num_dim;
            for (size_t i = 0; i < num_dim; i++) {
                o_raw8[i] = o_raw16[i] >> 8;
            }
            CodeHelper<kBits - 8>::compacted_code8(o_compact, o_raw8.get(), num_dim);
        } else {
            std::copy(o_raw16, o_raw16 + num_dim, o_raw8.get());
            compacted_code8(o_compact, o_raw8.get(), num_dim);
        }
    }

    static float compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
        if constexpr (kBits == 0)
            return 0;
        if constexpr (kBits > 8) {
            auto ip = CodeHelper<8>::compute_ip(query, y, D);
            return ip + 256 * CodeHelper<kBits - 8>::compute_ip(query, y + D, D);
        }
        auto rec = memory::make_unique_array<uint8_t>(D, 64);
        froce_decompact(y, rec.get(), D);
        return CodeHelper<8>::compute_ip(query, rec.get(), D);
    }

    // 4-way batched IP: compute <query, y_k> for k in {0,1,2,3} against four
    // codes sharing the same query. The point of this routine is not raw FLOPs
    // (decoding cost is the same as 4×compute_ip), but memory-level parallelism:
    // issuing four independent code loads per iteration lets the OoO engine
    // overlap their cache-miss stalls, which is the common case during HNSW
    // traversal where each code lives on a different cache line.
    //
    // Specializations exist for every kBits in [1,8]; each one mirrors the
    // decoding scheme of the corresponding compute_ip and simply hoists the
    // query loads and mask constants across the four codes. For kBits>8 we
    // recurse into CodeHelper<8> (low byte) + CodeHelper<kBits-8> (upper bits),
    // matching compacted_code16's layout of "D lower bytes then compacted upper".
    //
    // IMPORTANT: Do NOT use froce_decompact here — many kBits have specialized
    // compacted_code8 layouts that differ from the generic bit-interleaved one.
    static void compute_ip_4(
            const float *__restrict__ query,
            const uint8_t *__restrict__ y0,
            const uint8_t *__restrict__ y1,
            const uint8_t *__restrict__ y2,
            const uint8_t *__restrict__ y3,
            size_t D,
            float &r0, float &r1, float &r2, float &r3) {
        if constexpr (kBits == 0) {
            r0 = r1 = r2 = r3 = 0.0f;
            return;
        }
        if constexpr (kBits > 8) {
            float a0, a1, a2, a3, b0, b1, b2, b3;
            CodeHelper<8>::compute_ip_4(query, y0, y1, y2, y3, D, a0, a1, a2, a3);
            CodeHelper<kBits - 8>::compute_ip_4(
                    query, y0 + D, y1 + D, y2 + D, y3 + D, D, b0, b1, b2, b3);
            r0 = a0 + 256.0f * b0; r1 = a1 + 256.0f * b1;
            r2 = a2 + 256.0f * b2; r3 = a3 + 256.0f * b3;
            return;
        }
        // Unspecialized fallback: 4 sequential compute_ip calls. Correct for any
        // kBits, but forfeits the memory-level parallelism benefit; every kBits
        // in [1,8] should have a specialization below.
        r0 = compute_ip(query, y0, D);
        r1 = compute_ip(query, y1, D);
        r2 = compute_ip(query, y2, D);
        r3 = compute_ip(query, y3, D);
    }
};

template <>
inline void CodeHelper<1>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    for (size_t i = 0; i < num_dim; i += 8) {
        o_compact[i / 8] = 0;
        for (size_t j = 0; j < 8; j++) {
            o_compact[i / 8] |= ((o_raw[i + j] & 1) << j);
        }
    }
}

template <>
inline float CodeHelper<1>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ mask, size_t len) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps(); // Initialize the accumulator to 0

    // Process 16 float elements per loop
    for (size_t i = 0; i < len; i += 16) {
        // Calculate the mask position corresponding to the current block
        const auto m = mask + (i / 8);
        __mmask16 k = _cvtu32_mask16(m[0] + ((uint32_t)m[1] << 8));

        // Load data based on the mask (unselected positions are set to 0)
        __m512 values = _mm512_maskz_load_ps(k, query + i);

        // Accumulate to the accumulator
        acc = _mm512_add_ps(acc, values);
    }

    // Sum all elements in the accumulator
    return _mm512_reduce_add_ps(acc);
#else
    float ans = 0;
    for (size_t i = 0; i < len; i += 8) {
        auto m = mask[(i / 8)];
        for (int j = 0; j < 8; ++j) {
            if (m & (1 << j)) {
                ans += x[i + j];
            }
        }
    }
    return ans;
#endif
}

// 4-way specialization for kBits=1.
// Per 16 query floats we load one __m512, then apply 4 independent 16-bit
// masks (one per code) via maskz_mov — the query is shared across codes and
// the four mask-loads/mov/adds expose memory-level parallelism cheaply.
template <>
inline void CodeHelper<1>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t len,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    for (size_t i = 0; i < len; i += 16) {
        const __m512 q = _mm512_loadu_ps(query + i); // loaded once, shared by all 4
        auto accumulate = [&](const uint8_t *m, __m512 &acc) {
            __mmask16 kk = _cvtu32_mask16(
                    static_cast<uint32_t>(m[0]) | (static_cast<uint32_t>(m[1]) << 8));
            acc = _mm512_add_ps(acc, _mm512_maskz_mov_ps(kk, q));
        };
        const size_t off = i / 8;
        accumulate(y0 + off, acc0);
        accumulate(y1 + off, acc1);
        accumulate(y2 + off, acc2);
        accumulate(y3 + off, acc3);
    }
    r0 = _mm512_reduce_add_ps(acc0);
    r1 = _mm512_reduce_add_ps(acc1);
    r2 = _mm512_reduce_add_ps(acc2);
    r3 = _mm512_reduce_add_ps(acc3);
#else
    r0 = compute_ip(query, y0, len); r1 = compute_ip(query, y1, len);
    r2 = compute_ip(query, y2, len); r3 = compute_ip(query, y3, len);
#endif
}

template <>
inline void CodeHelper<2>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    // Create a mask to isolate the two least significant bits of each byte
    __m128i mask = _mm_set1_epi8(0b00000011);

    // Process the data in chunks of 64 bytes
    for (size_t d = 0; d < num_dim; d += 64) {
        // Load 64 bytes of raw data into four 128-bit vectors
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        // Apply the mask to extract the two least significant bits from each vector
        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2); // Shift left by 2 bits
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4); // Shift left by 4 bits
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6); // Shift left by 6 bits

        // Combine the processed vectors into a single compact representation
        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63));

        // Store the compacted data into the output buffer
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);

        // Move to the next chunk of raw data and output buffer
        o_raw += 64;
        o_compact += 16;
    }
}

template <>
inline float CodeHelper<2>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact));

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        o_compact += 16;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

// 4-way specialization for kBits=2.
// Per 64 values, each code stores 16 bytes packing (bit0,bit1) of 64 values.
// Same decoding as compute_ip, but query loads are hoisted across the 4 codes.
template <>
inline void CodeHelper<2>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < D; i += 64) {
        // Shared query loads (4 × 16 floats).
        const __m512 q0 = _mm512_loadu_ps(&query[i +  0]);
        const __m512 q1 = _mm512_loadu_ps(&query[i + 16]);
        const __m512 q2 = _mm512_loadu_ps(&query[i + 32]);
        const __m512 q3 = _mm512_loadu_ps(&query[i + 48]);

        auto accumulate = [&](const uint8_t *cp, __m512 &acc) {
            const __m128i cpt = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cp));
            const __m128i v0 = _mm_and_si128(cpt, mask);
            const __m128i v1 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
            const __m128i v2 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
            const __m128i v3 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);
            acc = _mm512_fmadd_ps(q0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v0)), acc);
            acc = _mm512_fmadd_ps(q1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v1)), acc);
            acc = _mm512_fmadd_ps(q2, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v2)), acc);
            acc = _mm512_fmadd_ps(q3, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v3)), acc);
        };

        const size_t off = i / 4;
        accumulate(y0 + off, sum0);
        accumulate(y1 + off, sum1);
        accumulate(y2 + off, sum2);
        accumulate(y3 + off, sum3);
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

template <>
inline void CodeHelper<3>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    // Create a mask to isolate the two least significant bits of each byte
    __m128i mask = _mm_set1_epi8(0b11);
    // __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t d = 0; d < num_dim; d += 64) {
        // Load 64 bytes of raw data into four 128-bit vectors
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        // Apply the mask to extract the two least significant bits from each vector
        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

        // Combine the processed vectors into a single compact representation
        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63));

        // Store the compacted data into the output buffer
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);
        o_compact += 16;

        // Initialize top_bit to store the top bits of the raw data
        int64_t top_bit = 0;
        int64_t top_mask = 0x0101010101010101;
        // Extract the top bits from the raw data
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t *>(o_raw + i);
            top_bit |= ((cur_codes >> 2) & top_mask) << (i / 8);
        }
        // Copy the top bits to the output buffer
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

template <>
inline float CodeHelper<3>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask = _mm_set1_epi8(0b11);
    __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact));
        o_compact += 16;

        int64_t top_bit = *reinterpret_cast<int64_t *>(o_compact);
        o_compact += 8;

        __m128i vec_00_to_15 = _mm_and_si128(cpt, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(cpt, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(cpt, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(cpt, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

// 4-way specialization for kBits=3.
// Packed layout (see compacted_code8): per 64 values, 16 bytes carry the two
// low bits (4 values/byte) and 8 bytes carry the top bit in a transposed form
// where byte j bit k = bit2 of value k*8+j.
//
// Strategy: load all 4 codes' compact data up front (to issue 4 independent
// memory streams early), then for each sub-group load the query once and do
// 4 decodes/FMADDs — one per code — so all 4 accumulators have independent
// FMA chains running in parallel.
template <>
inline void CodeHelper<3>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    const __m128i mask     = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);

    auto decode_sub = [&](__m128i cpt, int64_t top, int sg) -> __m128i {
        switch (sg) {
        case 0: return _mm_or_si128(
                _mm_and_si128(_mm_set_epi64x(top << 1, top << 2), top_mask),
                _mm_and_si128(cpt, mask));
        case 1: return _mm_or_si128(
                _mm_and_si128(_mm_set_epi64x(top >> 1, top >> 0), top_mask),
                _mm_and_si128(_mm_srli_epi16(cpt, 2), mask));
        case 2: return _mm_or_si128(
                _mm_and_si128(_mm_set_epi64x(top >> 3, top >> 2), top_mask),
                _mm_and_si128(_mm_srli_epi16(cpt, 4), mask));
        default: return _mm_or_si128(
                _mm_and_si128(_mm_set_epi64x(top >> 5, top >> 4), top_mask),
                _mm_and_si128(_mm_srli_epi16(cpt, 6), mask));
        }
    };

    for (size_t i = 0; i < D; i += 64) {
        const size_t off = (i / 64) * 24;
        // Issue 4 independent code-stream loads up front.
        const __m128i cpt0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off));
        const __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off));
        const __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off));
        const __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off));
        int64_t top0, top1, top2, top3;
        std::memcpy(&top0, y0 + off + 16, 8);
        std::memcpy(&top1, y1 + off + 16, 8);
        std::memcpy(&top2, y2 + off + 16, 8);
        std::memcpy(&top3, y3 + off + 16, 8);

        // For each sub-group, load query once and fmadd into all 4 accumulators.
        // The four FMADDs inside each sub-group are mutually independent
        // (different accumulators), giving 4-way ILP within and cross-sg chains.
        for (int sg = 0; sg < 4; ++sg) {
            const __m512 q = _mm512_loadu_ps(&query[i + sg * 16]);
            const __m128i v0 = decode_sub(cpt0, top0, sg);
            const __m128i v1 = decode_sub(cpt1, top1, sg);
            const __m128i v2 = decode_sub(cpt2, top2, sg);
            const __m128i v3 = decode_sub(cpt3, top3, sg);
            sum0 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v0)), sum0);
            sum1 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v1)), sum1);
            sum2 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v2)), sum2);
            sum3 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v3)), sum3);
        }
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

template <>
inline void CodeHelper<4>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    for (size_t j = 0; j < num_dim; j += 32) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        vec_16_to_31 = _mm_slli_epi16(vec_16_to_31, 4);

        __m128i compact = _mm_or_si128(vec_00_to_15, vec_16_to_31);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact), compact);

        o_raw += 32;
        o_compact += 16;
    }
}

template <>
inline float CodeHelper<4>::compute_ip(const float *__restrict__ x, const uint8_t *__restrict__ y, size_t D) {
    __m128i mask = _mm_set1_epi8(0b1111);
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 32) {
        __m128i a8 = _mm_loadu_epi32(&y[i / 2]);
        __m128i b8 = a8;
        __m512 x1 = _mm512_load_ps(&x[i]);
        __m512 x2 = _mm512_load_ps(&x[i + 16]);

        // get lower(0 to 15) and upper(16 to 31) 4 bits
        a8 = _mm_and_si128(a8, mask);
        b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

        __m512 af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
        sum = _mm512_fmadd_ps(af, x1, sum);
        __m512 bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
        sum = _mm512_fmadd_ps(bf, x2, sum);
    }
    return _mm512_reduce_add_ps(sum);
}

// 4-way specialization for kBits=4.
// Per 32 values, each code holds 16 bytes packing (lo4|hi4) nibbles.
// Query chunks x1,x2 are loaded once and shared across the four codes.
template <>
inline void CodeHelper<4>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    const __m128i mask = _mm_set1_epi8(0b1111);

    for (size_t i = 0; i < D; i += 32) {
        const __m512 x1 = _mm512_load_ps(&query[i]);
        const __m512 x2 = _mm512_load_ps(&query[i + 16]);

        auto accumulate = [&](const uint8_t *cp, __m512 &acc) {
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cp));
            const __m128i lo = _mm_and_si128(a, mask);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(a, 4), mask);
            acc = _mm512_fmadd_ps(x1, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), acc);
            acc = _mm512_fmadd_ps(x2, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), acc);
        };

        const size_t off = i / 2;
        accumulate(y0 + off, sum0);
        accumulate(y1 + off, sum1);
        accumulate(y2 + off, sum2);
        accumulate(y3 + off, sum3);
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

template <>
inline void CodeHelper<5>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    CodeHelper<1>::compacted_code8(o_compact + (num_dim * 4 / 8), o_raw, num_dim);
    auto o4 = memory::make_unique_array<uint8_t>(num_dim, 16);
    for (size_t i = 0; i < num_dim; i++) {
        o4[i] = o_raw[i] >> 1;
    }
    CodeHelper<4>::compacted_code8(o_compact, o4.get(), num_dim);
}

template <>
inline float CodeHelper<5>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    return 2 * CodeHelper<4>::compute_ip(query, y, D) + CodeHelper<1>::compute_ip(query, y + (D * 4 / 8), D);
}

// 4-way specialization for kBits=5. Mirrors compute_ip's 4-bit/1-bit split and
// delegates to the corresponding batch_4 routines so both halves enjoy
// memory-level parallelism.
template <>
inline void CodeHelper<5>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
    const size_t off = D * 4 / 8;
    float a0, a1, a2, a3, b0, b1, b2, b3;
    CodeHelper<4>::compute_ip_4(query, y0, y1, y2, y3, D, a0, a1, a2, a3);
    CodeHelper<1>::compute_ip_4(
            query, y0 + off, y1 + off, y2 + off, y3 + off, D, b0, b1, b2, b3);
    r0 = 2 * a0 + b0;
    r1 = 2 * a1 + b1;
    r2 = 2 * a2 + b2;
    r3 = 2 * a3 + b3;
}

template <>
inline void CodeHelper<6>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    __m128i mask2 = _mm_set1_epi8(0b11000000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    for (size_t d = 0; d < num_dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            vec_00_to_15, _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 0), compact);

        compact = _mm_or_si128(
            vec_16_to_31, _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask4),
            _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 32), compact);

        o_raw += 64;
        o_compact += 48;
    }
}

template <>
inline float CodeHelper<6>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 0));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4));
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4));

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        o_compact += 48;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

// 4-way specialization for kBits=6.
// Packed layout: per 64 values → 48 bytes split as three 16-byte blocks cpt1,
// cpt2, cpt3. sub-group-0 and sub-group-1 take 6 bits directly from cpt1/cpt2;
// sub-group-2 and sub-group-3 recombine bits from cpt1/cpt2 (high 2) with
// cpt3 (low 4 + high 4).
//
// Strategy mirrors kBits=3: load all four codes' compact bytes up front so
// the four memory streams can run in parallel, then iterate sub-groups with
// 4-way independent FMADDs that keep all accumulators busy.
template <>
inline void CodeHelper<6>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(0b00110000);
    const __m128i mask4 = _mm_set1_epi8(0b00001111);

    // Decode one code's sub-group `sg` from its three compact blocks.
    auto decode_sub = [&](__m128i cp1, __m128i cp2, __m128i cp3, int sg) -> __m128i {
        switch (sg) {
        case 0: return _mm_and_si128(cp1, mask6);
        case 1: return _mm_and_si128(cp2, mask6);
        case 2: return _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(cp1, 2), mask2),
                _mm_and_si128(cp3, mask4));
        default: return _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16(cp2, 2), mask2),
                _mm_and_si128(_mm_srli_epi16(cp3, 4), mask4));
        }
    };

    for (size_t i = 0; i < D; i += 64) {
        const size_t off = (i / 64) * 48;
        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off +  0));
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off + 16));
        const __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off + 32));
        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off +  0));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off + 16));
        const __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off + 32));
        const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off +  0));
        const __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off + 16));
        const __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off + 32));
        const __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off +  0));
        const __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off + 16));
        const __m128i c3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off + 32));

        for (int sg = 0; sg < 4; ++sg) {
            const __m512 q = _mm512_loadu_ps(&query[i + sg * 16]);
            const __m128i v0 = decode_sub(a0, b0, c0, sg);
            const __m128i v1 = decode_sub(a1, b1, c1, sg);
            const __m128i v2 = decode_sub(a2, b2, c2, sg);
            const __m128i v3 = decode_sub(a3, b3, c3, sg);
            sum0 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v0)), sum0);
            sum1 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v1)), sum1);
            sum2 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v2)), sum2);
            sum3 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v3)), sum3);
        }
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

template <>
inline void CodeHelper<7>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    __m128i mask2 = _mm_set1_epi8(0b11000000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < num_dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_32_to_47, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 0), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask4),
            _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask4), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(o_compact + 32), compact);
        o_compact += 48;

        int64_t top_bit = 0;
        int64_t top_mask = 0x0101010101010101;
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t *>(o_raw + i);
            top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_compact += 8;
        o_raw += 64;
    }
}

template <>
inline float CodeHelper<7>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D) {
    __m512 sum = _mm512_setzero_ps();
    uint8_t *o_compact = const_cast<uint8_t *>(y);
    float result = 0;

    __m128i mask6 = _mm_set1_epi8(0b00111111);
    __m128i mask2 = _mm_set1_epi8(0b00110000);
    __m128i mask4 = _mm_set1_epi8(0b00001111);
    __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < D; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 0));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<__m128i *>(o_compact + 32));

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt1, 2), mask2), _mm_and_si128(cpt3, mask4));
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi16(cpt2, 2), mask2),
            _mm_and_si128(_mm_srli_epi16(cpt3, 4), mask4));
        o_compact += 48;

        int64_t top_bit = *reinterpret_cast<int64_t *>(o_compact);
        o_compact += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 xx, yy;

        xx = _mm512_loadu_ps(&query[i]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 16]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 32]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.

        xx = _mm512_loadu_ps(&query[i + 48]);
        yy = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(
            xx, yy, sum); // I heard that this may cause underclocking on some CPUs.
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

// 4-way specialization for kBits=7.
// Packed layout: kBits=6 region (48 bytes) followed by 8 bytes of transposed
// bit6 (byte j bit k = bit6 of value k*8+j). Same layout strategy as kBits=6
// — upfront loads of the 4 codes' compact blocks plus top bytes, then a
// sub-group loop with 4-way independent FMADDs — with the extra top-bit
// merge on each sub-group.
template <>
inline void CodeHelper<7>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    const __m128i mask6    = _mm_set1_epi8(0b00111111);
    const __m128i mask2    = _mm_set1_epi8(0b00110000);
    const __m128i mask4    = _mm_set1_epi8(0b00001111);
    const __m128i top_mask = _mm_set1_epi8(0b1000000);

    auto decode_sub = [&](__m128i cp1, __m128i cp2, __m128i cp3, int64_t top, int sg) -> __m128i {
        __m128i base, tb;
        switch (sg) {
        case 0:
            base = _mm_and_si128(cp1, mask6);
            tb   = _mm_and_si128(_mm_set_epi64x(top << 5, top << 6), top_mask);
            break;
        case 1:
            base = _mm_and_si128(cp2, mask6);
            tb   = _mm_and_si128(_mm_set_epi64x(top << 3, top << 4), top_mask);
            break;
        case 2:
            base = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(cp1, 2), mask2),
                    _mm_and_si128(cp3, mask4));
            tb   = _mm_and_si128(_mm_set_epi64x(top << 1, top << 2), top_mask);
            break;
        default:
            base = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(cp2, 2), mask2),
                    _mm_and_si128(_mm_srli_epi16(cp3, 4), mask4));
            tb   = _mm_and_si128(_mm_set_epi64x(top >> 1, top << 0), top_mask);
            break;
        }
        return _mm_or_si128(tb, base);
    };

    for (size_t i = 0; i < D; i += 64) {
        const size_t off = (i / 64) * 56;
        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off +  0));
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off + 16));
        const __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y0 + off + 32));
        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off +  0));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off + 16));
        const __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y1 + off + 32));
        const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off +  0));
        const __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off + 16));
        const __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y2 + off + 32));
        const __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off +  0));
        const __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off + 16));
        const __m128i c3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y3 + off + 32));
        int64_t top0, top1, top2, top3;
        std::memcpy(&top0, y0 + off + 48, 8);
        std::memcpy(&top1, y1 + off + 48, 8);
        std::memcpy(&top2, y2 + off + 48, 8);
        std::memcpy(&top3, y3 + off + 48, 8);

        for (int sg = 0; sg < 4; ++sg) {
            const __m512 q = _mm512_loadu_ps(&query[i + sg * 16]);
            const __m128i v0 = decode_sub(a0, b0, c0, top0, sg);
            const __m128i v1 = decode_sub(a1, b1, c1, top1, sg);
            const __m128i v2 = decode_sub(a2, b2, c2, top2, sg);
            const __m128i v3 = decode_sub(a3, b3, c3, top3, sg);
            sum0 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v0)), sum0);
            sum1 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v1)), sum1);
            sum2 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v2)), sum2);
            sum3 = _mm512_fmadd_ps(q, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v3)), sum3);
        }
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

template <>
inline void CodeHelper<8>::compacted_code8(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    std::memcpy(o_compact, o_raw, sizeof(uint8_t) * num_dim);
}

template <>
inline float CodeHelper<8>::compute_ip(const float *__restrict__ x, const uint8_t *__restrict__ y, size_t D) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 16) {
        __m512 xx = _mm512_load_ps(&x[i]);
        __m512 yy =
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&y[i])));
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    result = _mm512_reduce_add_ps(sum);
#else
    for (size_t i = 0; i < L; i += 4) {
        result += x[i] * static_cast<float>(y[i]);
        result += x[i + 1] * static_cast<float>(y[i + 1]);
        result += x[i + 2] * static_cast<float>(y[i + 2]);
        result += x[i + 3] * static_cast<float>(y[i + 3]);
    }
#endif
    return result;
}

// 4-way specialization for kBits=8.
// Each 16-float query chunk is loaded once and fed to all 4 codes. The 4
// FMADDs within each iteration target distinct accumulators, so they are
// mutually independent and the OoO engine happily schedules them on the
// two FMA ports. Keep the structure simple — one accumulator per code is
// enough here because there is no decoding overhead to hide.
template <>
inline void CodeHelper<8>::compute_ip_4(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0,
        const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2,
        const uint8_t *__restrict__ y3,
        size_t D,
        float &r0, float &r1, float &r2, float &r3) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    auto load_u8_to_ps = [](const uint8_t *p) {
        return _mm512_cvtepi32_ps(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))));
    };

    for (size_t i = 0; i < D; i += 16) {
        const __m512 q = _mm512_load_ps(&query[i]); // loaded once, shared by all 4
        sum0 = _mm512_fmadd_ps(q, load_u8_to_ps(&y0[i]), sum0);
        sum1 = _mm512_fmadd_ps(q, load_u8_to_ps(&y1[i]), sum1);
        sum2 = _mm512_fmadd_ps(q, load_u8_to_ps(&y2[i]), sum2);
        sum3 = _mm512_fmadd_ps(q, load_u8_to_ps(&y3[i]), sum3);
    }
    r0 = _mm512_reduce_add_ps(sum0);
    r1 = _mm512_reduce_add_ps(sum1);
    r2 = _mm512_reduce_add_ps(sum2);
    r3 = _mm512_reduce_add_ps(sum3);
#else
    r0 = compute_ip(query, y0, D); r1 = compute_ip(query, y1, D);
    r2 = compute_ip(query, y2, D); r3 = compute_ip(query, y3, D);
#endif
}

// template <size_t bits>
// inline void CodeHelper<bits>::compacted_code16(uint8_t *o_compact, const uint16_t *o_raw16, size_t num_dim)

// template <size_t bits>
// inline float CodeHelper<bits>::compute_ip(const float *__restrict__ query, const uint8_t *__restrict__ y, size_t D)

inline auto get_IP_FUNC(int bits) -> float (*)(const float *__restrict__, const uint8_t *__restrict__, size_t) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compute_ip;
    case 1:
        return CodeHelper<1>::compute_ip;
    case 2:
        return CodeHelper<2>::compute_ip;
    case 3:
        return CodeHelper<3>::compute_ip;
    case 4:
        return CodeHelper<4>::compute_ip;
    case 5:
        return CodeHelper<5>::compute_ip;
    case 6:
        return CodeHelper<6>::compute_ip;
    case 7:
        return CodeHelper<7>::compute_ip;
    case 8:
        return CodeHelper<8>::compute_ip;
    case 9:
        return CodeHelper<9>::compute_ip;
    case 10:
        return CodeHelper<10>::compute_ip;
    case 11:
        return CodeHelper<11>::compute_ip;
    case 12:
        return CodeHelper<12>::compute_ip;
    case 13:
        return CodeHelper<13>::compute_ip;
    case 14:
        return CodeHelper<14>::compute_ip;
    case 15:
        return CodeHelper<15>::compute_ip;
    case 16:
        return CodeHelper<16>::compute_ip;
    default:
        std::cerr << "Error: Unsupported bits: " << bits << std::endl;
        assert(false);
    }
    return nullptr;
}

// 4-way IP function pointer type and factory
using IP_FUNC_4_t = void (*)(
        const float *__restrict__,
        const uint8_t *__restrict__, const uint8_t *__restrict__,
        const uint8_t *__restrict__, const uint8_t *__restrict__,
        size_t, float &, float &, float &, float &);

template <size_t kBits>
inline void ip_func_4_dispatch(
        const float *__restrict__ query,
        const uint8_t *__restrict__ y0, const uint8_t *__restrict__ y1,
        const uint8_t *__restrict__ y2, const uint8_t *__restrict__ y3,
        size_t D, float &r0, float &r1, float &r2, float &r3) {
    CodeHelper<kBits>::compute_ip_4(query, y0, y1, y2, y3, D, r0, r1, r2, r3);
}

inline IP_FUNC_4_t get_IP_FUNC_4(int bits) {
    switch (bits) {
    case 0:  return ip_func_4_dispatch<0>;
    case 1:  return ip_func_4_dispatch<1>;
    case 2:  return ip_func_4_dispatch<2>;
    case 3:  return ip_func_4_dispatch<3>;
    case 4:  return ip_func_4_dispatch<4>;
    case 5:  return ip_func_4_dispatch<5>;
    case 6:  return ip_func_4_dispatch<6>;
    case 7:  return ip_func_4_dispatch<7>;
    case 8:  return ip_func_4_dispatch<8>;
    case 9:  return ip_func_4_dispatch<9>;
    case 10: return ip_func_4_dispatch<10>;
    case 11: return ip_func_4_dispatch<11>;
    case 12: return ip_func_4_dispatch<12>;
    case 13: return ip_func_4_dispatch<13>;
    case 14: return ip_func_4_dispatch<14>;
    case 15: return ip_func_4_dispatch<15>;
    case 16: return ip_func_4_dispatch<16>;
    default:
        std::cerr << "Error: Unsupported bits for IP_FUNC_4: " << bits << std::endl;
        assert(false);
    }
    return nullptr;
}

inline auto get_compacted_code16_func(int bits) -> void (*)(uint8_t *o_compact, const uint16_t *o_raw, size_t num_dim) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compacted_code16;
    case 1:
        return CodeHelper<1>::compacted_code16;
    case 2:
        return CodeHelper<2>::compacted_code16;
    case 3:
        return CodeHelper<3>::compacted_code16;
    case 4:
        return CodeHelper<4>::compacted_code16;
    case 5:
        return CodeHelper<5>::compacted_code16;
    case 6:
        return CodeHelper<6>::compacted_code16;
    case 7:
        return CodeHelper<7>::compacted_code16;
    case 8:
        return CodeHelper<8>::compacted_code16;
    case 9:
        return CodeHelper<9>::compacted_code16;
    case 10:
        return CodeHelper<10>::compacted_code16;
    case 11:
        return CodeHelper<11>::compacted_code16;
    case 12:
        return CodeHelper<12>::compacted_code16;
    case 13:
        return CodeHelper<13>::compacted_code16;
    case 14:
        return CodeHelper<14>::compacted_code16;
    case 15:
        return CodeHelper<15>::compacted_code16;
    case 16:
        return CodeHelper<16>::compacted_code16;
    default:
        assert(false);
    }
    return nullptr;
}

inline auto get_compacted_code8_func(int bits) -> void (*)(uint8_t *o_compact, const uint8_t *o_raw, size_t num_dim) {
    switch (bits) {
    case 0:
        return CodeHelper<0>::compacted_code8;
    case 1:
        return CodeHelper<1>::compacted_code8;
    case 2:
        return CodeHelper<2>::compacted_code8;
    case 3:
        return CodeHelper<3>::compacted_code8;
    case 4:
        return CodeHelper<4>::compacted_code8;
    case 5:
        return CodeHelper<5>::compacted_code8;
    case 6:
        return CodeHelper<6>::compacted_code8;
    case 7:
        return CodeHelper<7>::compacted_code8;
    case 8:
        return CodeHelper<8>::compacted_code8;
    case 9:
        return CodeHelper<9>::compacted_code8;
    case 10:
        return CodeHelper<10>::compacted_code8;
    case 11:
        return CodeHelper<11>::compacted_code8;
    case 12:
        return CodeHelper<12>::compacted_code8;
    case 13:
        return CodeHelper<13>::compacted_code8;
    case 14:
        return CodeHelper<14>::compacted_code8;
    case 15:
        return CodeHelper<15>::compacted_code8;
    case 16:
        return CodeHelper<16>::compacted_code8;
    default:
        assert(false);
    }
    return nullptr;
}
} // namespace saqlib::utils
