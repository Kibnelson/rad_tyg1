// Same four little-endian bytes, MurmurHash3_x64_128 first word, and bit layout
// as encoding.to_bitmaps_python. Canonical public-domain hash: external/simd.
#include <cstdint>
#include <cstring>
#include <omp.h>
#include "MurmurHash3.h"

extern "C" int bitmap_encode(const uint32_t* sketches, uint8_t* output,
        int64_t rows, int width, uint64_t bits, uint32_t seed, int threads) {
    if (!rows) return 0;
    const uint64_t bytes = bits / 8;
    const bool power2 = (bits & (bits - 1)) == 0;
    int team = 1;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        team = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            auto* target = output + row * bytes;
            std::memset(target, 0, bytes);
            for (int j = 0; j < width; ++j) {
                const uint32_t v = sketches[row * width + j];
                const uint8_t key[4] = {uint8_t(v), uint8_t(v >> 8),
                                       uint8_t(v >> 16), uint8_t(v >> 24)};
                uint64_t hash[2];
                MurmurHash3_x64_128(key, 4, seed, hash);
                const uint64_t bucket = power2 ? hash[0] & (bits - 1) : hash[0] % bits;
                target[bucket >> 3] |= uint8_t(1u << (bucket & 7));
            }
        }
    }
    return team;
}
