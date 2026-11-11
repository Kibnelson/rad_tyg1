/*
 * Raw 112-position MinHash Jaccard distance for the isolated MH experiment.
 * Reference: reference/mh_jaccard_5m/legacy/.../avx512-inl.h:524-618.
 * Packed bytes hold complete uint32 hash values, not bitmap buckets.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && \
        (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define RAD_MH_JACCARD_X86_TARGET 1
#endif

namespace faiss {

constexpr size_t minhash_jaccard_positions = 112;
constexpr size_t minhash_jaccard_code_size = minhash_jaccard_positions * 4;

inline bool minhash_jaccard_avx512_available() {
#ifdef RAD_MH_JACCARD_X86_TARGET
    static const bool available = __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("fma") && __builtin_cpu_supports("popcnt");
    return available;
#else
    return false;
#endif
}

inline int32_t minhash_jaccard_quantized_distance(uint32_t matches) {
    const float similarity = float(matches) / float(minhash_jaccard_positions);
    const float distance = 1.0f - similarity;
    // The preserved old -O3/-mfma build contracts this multiply-add. Retain
    // the same float arithmetic rather than rounding final HNSW scores only.
    return int32_t(std::floor(std::fma(distance, 100.0f, 1e-6f)));
}

// Independent portable oracle; the production metric never falls back here.
#if defined(__GNUC__) && !defined(__clang__) && defined(RAD_MH_JACCARD_X86_TARGET)
__attribute__((target("no-avx,no-avx2,no-avx512f"),
               optimize("no-tree-vectorize,no-tree-slp-vectorize"), noinline))
#elif defined(__clang__) && defined(RAD_MH_JACCARD_X86_TARGET)
__attribute__((target("no-avx,no-avx2,no-avx512f"), noinline))
#endif
inline uint32_t minhash_jaccard_matches_scalar(const uint8_t* a, const uint8_t* b) {
    uint32_t matches = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < minhash_jaccard_positions; ++i) {
        uint32_t av, bv;
        __builtin_memcpy(&av, a + i * 4, 4);
        __builtin_memcpy(&bv, b + i * 4, 4);
        matches += av == bv;
    }
    return matches;
}

#ifdef RAD_MH_JACCARD_X86_TARGET
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((target("avx512f,popcnt,fma"), optimize("O3"), noinline))
#else
__attribute__((target("avx512f,popcnt,fma"), noinline))
#endif
inline int32_t minhash_jaccard_distance_avx512(
        const uint8_t* a, const uint8_t* b, int n32 = 112) {
    // Mirror the ordinary loop selected in the preserved old source by its
    // __AVX512F guard. The saved old build auto-vectorized this loop, rather
    // than executing that source's hand-written eq-mask/popcount branch.
    // Retain a runtime lane count, as in the old HammingComputerDefault. The
    // IndexBinaryHNSW adapter qualifies storage as exactly 112 positions.
    uint32_t I = 0;
    uint32_t U = 0;
    for (int idx = 0; idx < n32; ++idx) {
        uint32_t av, bv;
        __builtin_memcpy(&av, a + idx * 4, 4);
        __builtin_memcpy(&bv, b + idx * 4, 4);
        ++U;
        if (av == bv) {
            ++I;
        }
    }
    if (U == 0) {
        return 1;
    }
    const float jaccard = float(I) / float(U);
    const float distance = 1.0f - jaccard;
    return int32_t(std::floor(std::fma(distance, 100.0f, 1e-6f)));
}
#endif

} // namespace faiss
