// Standalone routing kernels. No Faiss symbols or distance implementation changes.
#include <cstdint>
#include <cstddef>
#include <omp.h>

static uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

extern "C" void routing_keys(const uint32_t* signatures, uint64_t* output,
    int64_t n, int width, int bands, int rows, uint64_t seed, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t q = 0; q < n; ++q) {
        for (int b = 0; b < bands; ++b) {
            uint64_t key = mix64(seed ^ static_cast<uint64_t>(b));
            for (int r = 0; r < rows; ++r)
                key = mix64(key ^ signatures[q * width + b * rows + r]);
            output[q * bands + b] = key;
        }
    }
}

extern "C" void routing_add(const uint64_t* keys, int64_t count, uint64_t* bits,
    uint64_t nbits, int probes, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i = 0; i < count; ++i) {
        for (int p = 0; p < probes; ++p) {
            uint64_t position = mix64(keys[i] + UINT64_C(0x9e3779b97f4a7c15) * p) % nbits;
            __atomic_fetch_or(bits + (position >> 6), UINT64_C(1) << (position & 63), __ATOMIC_RELAXED);
        }
    }
}

extern "C" void routing_any(const uint64_t* keys, int64_t n, int bands,
    const uint64_t* bits, uint64_t nbits, int probes, uint8_t* output, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t q = 0; q < n; ++q) {
        bool any = false;
        for (int b = 0; b < bands && !any; ++b) {
            bool present = true;
            for (int p = 0; p < probes; ++p) {
                uint64_t position = mix64(keys[q * bands + b] + UINT64_C(0x9e3779b97f4a7c15) * p) % nbits;
                if (!(bits[position >> 6] & (UINT64_C(1) << (position & 63)))) {
                    present = false;
                    break;
                }
            }
            any |= present;
        }
        output[q] = static_cast<uint8_t>(any);
    }
}
