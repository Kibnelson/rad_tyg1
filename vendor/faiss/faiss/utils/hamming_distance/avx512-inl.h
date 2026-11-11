/*
 * MinHash-bitmap Jaccard distance for the RAD V2 experiment.
 * The representation is a set of occupied bitmap buckets, not raw MinHash.
 * Restores the original hamming.h -> hamdis-inl.h -> avx512-inl.h
 * bitmap extension. Ordinary Hamming uses fresh Faiss's *_tpl classes.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && \
        (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define RAD_MINHASH_BITMAP_X86_TARGET 1
#endif

namespace faiss {

inline bool minhash_bitmap_avx512_available() {
#ifdef RAD_MINHASH_BITMAP_X86_TARGET
    static const bool available = __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512vpopcntdq");
    return available;
#else
    return false;
#endif
}

struct MinHashBitmapCounts {
    uint64_t population = 0;
    uint64_t difference = 0;
};

// Explicit scalar modes must not acquire packed counting loops through
// auto-vectorization in an otherwise AVX-512-enabled Faiss translation unit.
// Scalar POPCNT and single-lane floating-point arithmetic remain allowed.
#if defined(__GNUC__) && !defined(__clang__) && defined(RAD_MINHASH_BITMAP_X86_TARGET)
#define RAD_BITMAP_SCALAR __attribute__((target("no-avx,no-avx2,no-avx512f"), optimize("no-tree-vectorize,no-tree-slp-vectorize"), noinline))
#elif defined(__clang__) && defined(RAD_MINHASH_BITMAP_X86_TARGET)
#define RAD_BITMAP_SCALAR __attribute__((target("no-avx,no-avx2,no-avx512f"), noinline))
#else
#define RAD_BITMAP_SCALAR
#endif

template <bool NeedPopulation, bool NeedDifference>
RAD_BITMAP_SCALAR inline MinHashBitmapCounts minhash_bitmap_counts_strict_scalar(
        const uint8_t* a, const uint8_t* b, size_t nbytes) {
    MinHashBitmapCounts counts;
    size_t i = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (; i + 8 <= nbytes; i += 8) {
        uint64_t aw, bw;
        __builtin_memcpy(&bw, b + i, 8);
        if constexpr (NeedPopulation) {
            counts.population += __builtin_popcountll(bw);
        }
        if constexpr (NeedDifference) {
            __builtin_memcpy(&aw, a + i, 8);
            counts.difference += __builtin_popcountll(aw ^ bw);
        }
    }
    for (; i < nbytes; ++i) {
        if constexpr (NeedPopulation) {
            counts.population += __builtin_popcount(unsigned(b[i]));
        }
        if constexpr (NeedDifference) {
            counts.difference += __builtin_popcount(unsigned(a[i] ^ b[i]));
        }
    }
    return counts;
}

// If a is null, count only B. Otherwise also count A XOR B.
inline MinHashBitmapCounts minhash_bitmap_counts_scalar(
        const uint8_t* a, const uint8_t* b, size_t nbytes) {
    MinHashBitmapCounts counts;
    size_t i = 0;
    for (; i + 8 <= nbytes; i += 8) {
        uint64_t aw = 0, bw;
        std::memcpy(&bw, b + i, 8);
        if (a) {
            std::memcpy(&aw, a + i, 8);
            counts.difference += __builtin_popcountll(aw ^ bw);
        }
        counts.population += __builtin_popcountll(bw);
    }
    for (; i < nbytes; ++i) {
        counts.population += __builtin_popcount(unsigned(b[i]));
        if (a) {
            counts.difference += __builtin_popcount(unsigned(a[i] ^ b[i]));
        }
    }
    return counts;
}

#ifdef RAD_MINHASH_BITMAP_X86_TARGET
// Compile only this function for VPOPCNTDQ. Other code can run on CPUs that
// expose base AVX-512 without vector popcount. Runtime dispatch checks both.
__attribute__((target("avx512f,avx512vpopcntdq")))
inline MinHashBitmapCounts minhash_bitmap_counts_avx512(
        const uint8_t* a, const uint8_t* b, size_t nbytes) {
    __m512i accb = _mm512_setzero_si512();
    __m512i accx = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= nbytes; i += 64) {
        const __m512i vb = _mm512_loadu_si512(b + i);
        accb = _mm512_add_epi64(accb, _mm512_popcnt_epi64(vb));
        if (a) {
            const __m512i va = _mm512_loadu_si512(a + i);
            accx = _mm512_add_epi64(
                    accx, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
        }
    }
    auto tail = minhash_bitmap_counts_scalar(a ? a + i : nullptr, b + i, nbytes - i);
    tail.population += uint64_t(_mm512_reduce_add_epi64(accb));
    tail.difference += uint64_t(_mm512_reduce_add_epi64(accx));
    return tail;
}

__attribute__((target("avx512f,avx512vpopcntdq")))
inline uint64_t minhash_bitmap_xor_avx512(
        const uint8_t* a, const uint8_t* b, size_t nbytes) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= nbytes; i += 64) {
        const __m512i va = _mm512_loadu_si512(a + i);
        const __m512i vb = _mm512_loadu_si512(b + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
    }
    uint64_t result = uint64_t(_mm512_reduce_add_epi64(acc));
    if (i < nbytes) {
        result += minhash_bitmap_counts_strict_scalar<false, true>(a + i, b + i, nbytes - i).difference;
    }
    return result;
}
#endif

inline MinHashBitmapCounts minhash_bitmap_counts(
        const uint8_t* a, const uint8_t* b, size_t nbytes) {
#ifdef RAD_MINHASH_BITMAP_X86_TARGET
    if (minhash_bitmap_avx512_available()) {
        return minhash_bitmap_counts_avx512(a, b, nbytes);
    }
#endif
    return minhash_bitmap_counts_scalar(a, b, nbytes);
}

inline MinHashBitmapCounts minhash_bitmap_counts_selected(
        const uint8_t* a, const uint8_t* b, size_t nbytes,
        bool use_simd, bool strict_scalar) {
#ifdef RAD_MINHASH_BITMAP_X86_TARGET
    if (use_simd) {
        return minhash_bitmap_counts_avx512(a, b, nbytes);
    }
#endif
    if (strict_scalar) {
        return a ? minhash_bitmap_counts_strict_scalar<true, true>(a, b, nbytes)
                 : minhash_bitmap_counts_strict_scalar<true, false>(nullptr, b, nbytes);
    }
    return minhash_bitmap_counts_scalar(a, b, nbytes);
}

inline uint64_t minhash_bitmap_xor_selected(
        const uint8_t* a, const uint8_t* b, size_t nbytes, bool use_simd) {
#ifdef RAD_MINHASH_BITMAP_X86_TARGET
    if (use_simd) {
        return minhash_bitmap_xor_avx512(a, b, nbytes);
    }
#endif
    return minhash_bitmap_counts_strict_scalar<false, true>(a, b, nbytes).difference;
}

// Same SSE reciprocal estimate + Newton refinement as the cached paper
// adapter. This is scalar floating-point arithmetic, not bitmap SIMD counting.
inline float minhash_bitmap_paper_rcp(float x) {
#if defined(__SSE__)
    const __m128 vx = _mm_set_ss(x);
    __m128 r = _mm_rcp_ss(vx);
    r = _mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(vx, r)));
    return _mm_cvtss_f32(r);
#else
    return 1.0f / x;
#endif
}

inline float minhash_bitmap_cached_distance(
        uint64_t pa, uint64_t pb, uint64_t px, bool direct_division) {
    const uint64_t denominator = pa + pb + px;
    const float safe_denominator = float(denominator + (denominator == 0));
    return direct_division ? (2.0f * float(px)) / safe_denominator
                           : (2.0f * float(px)) * minhash_bitmap_paper_rcp(safe_denominator);
}

// Original bitmap computer name and entry points, shared by all runtime modes.
struct HammingComputerDefault {
    const uint8_t* a8 = nullptr;
    size_t nbytes = 0;
    size_t quotient8 = 0, remainder8 = 0;
    uint64_t pa_cached = 0;
    bool use_simd = false;
    bool strict_scalar = false;

    HammingComputerDefault() = default;

    HammingComputerDefault(const uint8_t* bytes, size_t size,
                           bool simd = minhash_bitmap_avx512_available(), bool scalar = false) {
        set(bytes, size, simd, scalar);
    }

    void set(const uint8_t* bytes, size_t size,
             bool simd = minhash_bitmap_avx512_available(), bool scalar = false) {
        a8 = bytes;
        nbytes = size;
        quotient8 = size / 8;
        remainder8 = size % 8;
        use_simd = simd;
        strict_scalar = scalar;
        pa_cached = minhash_bitmap_counts_selected(nullptr, bytes, size, use_simd, strict_scalar).population;
    }

    uint64_t hamming(const uint8_t* candidate) const {
        return minhash_bitmap_xor_selected(a8, candidate, nbytes, use_simd);
    }

    // Saved cached headers expose this similarity helper. The cached HNSW
    // adapters instead form distance directly from pa/pb/px to retain their
    // division/reciprocal arithmetic (see FlatHammingDis in IndexBinaryHNSW.cpp).
    float jaccard_with_pb(const uint8_t* candidate, uint64_t pb) const {
        const uint64_t px = hamming(candidate);
        const uint64_t sum = pa_cached + pb;
        return sum + px ? float(sum - px) / float(sum + px) : 1.0f;
    }

    float jaccard_onthefly(const uint8_t* candidate) const {
        const auto counts = minhash_bitmap_counts_selected(a8, candidate, nbytes, use_simd, strict_scalar);
        const uint64_t sum = pa_cached + counts.population;
        const uint64_t denominator = sum + counts.difference;
        // Preserve V1: similarity of two all-zero bitmaps is zero.
        return denominator
                ? float(sum - counts.difference) / float(denominator)
                : 0.0f;
    }
};

} // namespace faiss
