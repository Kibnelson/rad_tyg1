/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryHNSW.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>

#include <cstdint>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/minhash_jaccard.h>

#include <faiss/impl/simd_dispatch.h>

#include <random>

// Scalar (NONE) fallback for dynamic dispatch
#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/binary_hamming/IndexBinaryHNSW_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {

namespace {
std::mutex minhash_bitmap_cache_mutex;

bool minhash_bitmap_cached(MetricType metric) {
    return metric == METRIC_MinhashBitmapCacheSIMD ||
            metric == METRIC_MinhashBitmapCacheScalar;
}

bool minhash_bitmap_simd(MetricType metric) {
    return metric == METRIC_MinhashBitmapCacheSIMD ||
            metric == METRIC_MinhashBitmapNoCacheSIMD ||
            (metric == METRIC_MinhashBitmap && minhash_bitmap_avx512_available());
}

void validate_minhash_bitmap_mode(MetricType metric, int bits) {
    if (metric == METRIC_MinhashJaccard) {
        FAISS_THROW_IF_NOT_MSG(bits == int(minhash_jaccard_code_size * 8),
                "Raw MinHash Jaccard requires 112 uint32 hashes (3584 storage bits)");
        FAISS_THROW_IF_NOT_MSG(minhash_jaccard_avx512_available(),
                "Raw MinHash Jaccard requires AVX-512F, FMA and POPCNT; no scalar fallback");
        return;
    }
    if (!is_minhash_bitmap_metric(metric)) {
        return;
    }
    if (metric == METRIC_MinhashBitmapCacheSIMD ||
            metric == METRIC_MinhashBitmapNoCacheSIMD) {
        FAISS_THROW_IF_NOT_MSG(minhash_bitmap_avx512_available(),
                "Selected minhash_bitmap SIMD mode requires AVX-512 VPOPCNTDQ; "
                "use a scalar mode or v2_auto on this CPU");
    }
    FAISS_THROW_IF_NOT_MSG(!minhash_bitmap_cached(metric) || bits <= 65535,
            "Cached minhash_bitmap modes require at most 65535 bits for uint16 counts");
}

// Keep the baseline adapter/factory names. The namespace distinguishes this
// bitmap adapter from fresh Faiss's ordinary per-ISA FlatHammingDis template.
namespace minhash_bitmap {

template <class HammingComputer>
struct FlatHammingDis : DistanceComputer {
    const uint8_t* bytes;
    const size_t code_size;
    HammingComputer hc;
    const MetricType metric;
    const uint16_t* populations;
    const bool use_simd;
    const bool strict_scalar;

    FlatHammingDis(const IndexBinaryFlat& flat, MetricType mode,
                   const uint16_t* cache)
            : bytes(flat.xb.data()), code_size(flat.code_size), metric(mode),
              populations(cache), use_simd(minhash_bitmap_simd(mode)),
              strict_scalar(mode == METRIC_MinhashBitmapCacheScalar ||
                            mode == METRIC_MinhashBitmapNoCacheScalar) {}

    void set_query(const float* query) override {
        hc.set(reinterpret_cast<const uint8_t*>(query), code_size, use_simd, strict_scalar);
    }

    float operator()(idx_t i) override {
        if (minhash_bitmap_cached(metric)) {
            const uint64_t px = hc.hamming(bytes + size_t(i) * code_size);
            return minhash_bitmap_cached_distance(hc.pa_cached, populations[i], px,
                    metric == METRIC_MinhashBitmapCacheSIMD);
        }
        const float j = hc.jaccard_onthefly(bytes + size_t(i) * code_size);
        return 1.0f - j;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        if (minhash_bitmap_cached(metric)) {
            const uint64_t px = minhash_bitmap_xor_selected(
                    bytes + size_t(i) * code_size, bytes + size_t(j) * code_size,
                    code_size, use_simd);
            // Both cached paper adapters use the reciprocal path for pruning.
            return minhash_bitmap_cached_distance(populations[i], populations[j], px, false);
        }
        HammingComputerDefault hi(bytes + size_t(i) * code_size, code_size, use_simd, strict_scalar);
        const float jacc = hi.jaccard_onthefly(bytes + size_t(j) * code_size);
        return 1.0f - jacc;
    }
};

struct BuildDistanceComputer {
    using T = DistanceComputer*;
    MetricType metric;
    const uint16_t* populations;

    template <class HammingComputer>
    DistanceComputer* f(IndexBinaryFlat* flat_storage) {
        return new FlatHammingDis<HammingComputer>(*flat_storage, metric, populations);
    }
};

} // namespace minhash_bitmap

struct RawMinHashJaccardDis : DistanceComputer {
    const uint8_t* bytes;
    const size_t code_size;
    const int n32;
    const uint8_t* query = nullptr;

    explicit RawMinHashJaccardDis(const IndexBinaryFlat& flat)
            : bytes(flat.xb.data()), code_size(flat.code_size), n32(flat.code_size / 4) {}

    void set_query(const float* x) override {
        query = reinterpret_cast<const uint8_t*>(x);
    }

    float operator()(idx_t i) override {
#ifdef RAD_MH_JACCARD_X86_TARGET
        return float(minhash_jaccard_distance_avx512(
                query, bytes + size_t(i) * code_size, n32));
#else
        FAISS_THROW_MSG("Raw MinHash Jaccard requires AVX-512F");
#endif
    }

    float symmetric_dis(idx_t i, idx_t j) override {
#ifdef RAD_MH_JACCARD_X86_TARGET
        return float(minhash_jaccard_distance_avx512(
                bytes + size_t(i) * code_size,
                bytes + size_t(j) * code_size, n32));
#else
        FAISS_THROW_MSG("Raw MinHash Jaccard requires AVX-512F");
#endif
    }
};

inline int32_t minhash_bitmap_output(float distance, idx_t label) {
    // HNSW pads missing neighbors with label -1 and an infinite/large distance.
    // Preserve a sortable sentinel instead of converting infinity to int32.
    return label < 0 || !std::isfinite(distance)
            ? std::numeric_limits<int32_t>::max()
            : int32_t(std::floor(distance * 100.0f + 1e-6f));
}

inline int32_t minhash_jaccard_output(float distance, idx_t label) {
    // The distance computer already returns the old integer distance. Scaling
    // here again would silently turn a duplicate distance 25 into 2500.
    return label < 0 || !std::isfinite(distance)
            ? std::numeric_limits<int32_t>::max()
            : int32_t(distance);
}
} // namespace

bool IndexBinaryHNSW::minhash_bitmap_uses_avx512() const {
    return is_minhash_bitmap_metric(metric_type) && minhash_bitmap_simd(metric_type) &&
            minhash_bitmap_avx512_available();
}

bool IndexBinaryHNSW::minhash_jaccard_uses_avx512() const {
    return metric_type == METRIC_MinhashJaccard && minhash_jaccard_avx512_available();
}

size_t IndexBinaryHNSW::minhash_bitmap_popcount_cache_size() const {
    std::lock_guard<std::mutex> lock(minhash_bitmap_cache_mutex);
    return minhash_bitmap_popcounts_.size();
}

void IndexBinaryHNSW::ensure_minhash_bitmap_popcounts_() const {
    if (!minhash_bitmap_cached(metric_type)) {
        return;
    }
    validate_minhash_bitmap_mode(metric_type, d);
    auto* flat = dynamic_cast<const IndexBinaryFlat*>(storage);
    FAISS_THROW_IF_NOT_MSG(flat, "minhash_bitmap requires IndexBinaryFlat storage");
    std::lock_guard<std::mutex> lock(minhash_bitmap_cache_mutex);
    const size_t first = minhash_bitmap_popcounts_.size();
    const size_t total = size_t(flat->ntotal);
    if (first >= total) {
        return;
    }
    minhash_bitmap_popcounts_.resize(total);
    const bool simd = minhash_bitmap_simd(metric_type);
#pragma omp parallel for if(total - first >= 16384) schedule(static)
    for (int64_t i = int64_t(first); i < int64_t(total); ++i) {
        const auto count = minhash_bitmap_counts_selected(nullptr,
                flat->xb.data() + size_t(i) * code_size, code_size, simd, !simd);
        minhash_bitmap_popcounts_[i] = uint16_t(count.population);
    }
}

/**************************************************************
 * IndexBinaryHNSW implementation
 **************************************************************/

IndexBinaryHNSW::IndexBinaryHNSW() {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(int d_, int M)
        : IndexBinary(d_),
          hnsw(M),
          own_fields(true),
          storage(new IndexBinaryFlat(d_)) {
    is_trained = true;
}

IndexBinaryHNSW::IndexBinaryHNSW(IndexBinary* storage_, int M)
        : IndexBinary(storage_->d),
          hnsw(M),
          own_fields(false),
          storage(storage_) {
    is_trained = true;
}

IndexBinaryHNSW::~IndexBinaryHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexBinaryHNSW::train(idx_t n, const uint8_t* x) {
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexBinaryHNSW::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    validate_minhash_bitmap_mode(metric_type, d);
    // Allocate/rebuild before entering the parallel search region.
    ensure_minhash_bitmap_popcounts_();
    const SearchParametersHNSW* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(
                params, "IndexBinaryHNSW params have incorrect type");
    }

    // we use the buffer for distances as float but convert them back
    // to int in the end
    float* distances_f = (float*)distances;

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances_f, labels, k);

    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

#pragma omp parallel
    {
        std::unique_ptr<VisitedTable> vt = VisitedTable::create(ntotal);
        std::unique_ptr<DistanceComputer> dis(get_distance_computer());
        RH::SingleResultHandler res(bres);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            dis->set_query((float*)(x + i * code_size));
            // Given that IndexBinaryHNSW is not an IndexHNSW, we pass nullptr
            // as the index parameter. This state does not get used in the
            // search function, as it is merely there to enable Panorama
            // execution for IndexHNSWFlatPanorama.
            HNSWStats stats = hnsw.search(*dis, nullptr, res, *vt, params_in);
            n1 += stats.n1;
            n2 += stats.n2;
            ndis += stats.ndis;
            nhops += stats.nhops;
            res.end();
        }
    }

    if (get_search_stats_enabled()) {
        hnsw_stats.combine_atomic({n1, n2, ndis, nhops});
    }

#pragma omp parallel for
    for (idx_t i = 0; i < n * k; ++i) {
        if (is_minhash_bitmap_metric(metric_type)) {
            distances[i] = minhash_bitmap_output(distances_f[i], labels[i]);
        } else if (metric_type == METRIC_MinhashJaccard) {
            distances[i] = minhash_jaccard_output(distances_f[i], labels[i]);
        } else {
            distances[i] = std::round(distances_f[i]);
        }
    }
}

void IndexBinaryHNSW::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_NOT(is_trained);
    validate_minhash_bitmap_mode(metric_type, d);
    size_t n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;
    ensure_minhash_bitmap_popcounts_();

    bool preset_levels = hnsw.levels.size() == static_cast<size_t>(ntotal);

    hnsw_add_vertices_deterministic(
            hnsw,
            n0,
            n,
            d,
            init_level0,
            keep_max_size_level0,
            preset_levels,
            verbose,
            [this] { return get_distance_computer(); },
            [this, x, n0](DistanceComputer& dc, HNSW::storage_idx_t pt_id) {
                dc.set_query((const float*)(x + (pt_id - n0) * code_size));
            });
}

void IndexBinaryHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
    std::lock_guard<std::mutex> lock(minhash_bitmap_cache_mutex);
    minhash_bitmap_popcounts_.clear();
}

void IndexBinaryHNSW::reconstruct(idx_t key, uint8_t* recons) const {
    storage->reconstruct(key, recons);
}

DistanceComputer* IndexBinaryHNSW::get_distance_computer() const {
    IndexBinaryFlat* flat_storage = dynamic_cast<IndexBinaryFlat*>(storage);
    FAISS_THROW_IF_MSG(
            flat_storage == nullptr,
            "IndexBinaryHNSW requires IndexBinaryFlat storage");
    if (metric_type == METRIC_MinhashJaccard) {
        validate_minhash_bitmap_mode(metric_type, d);
        FAISS_THROW_IF_NOT_MSG(flat_storage->code_size == minhash_jaccard_code_size,
                "Raw MinHash Jaccard storage must hold 448 bytes per signature");
        return new RawMinHashJaccardDis(*flat_storage);
    }
    if (is_minhash_bitmap_metric(metric_type)) {
        validate_minhash_bitmap_mode(metric_type, d);
        ensure_minhash_bitmap_popcounts_();
        minhash_bitmap::BuildDistanceComputer bdc{metric_type,
                minhash_bitmap_cached(metric_type) ? minhash_bitmap_popcounts_.data() : nullptr};
        return dispatch_HammingComputer(code_size, bdc, flat_storage);
    }
    return with_simd_level([&]<SIMDLevel SL>() {
        return make_binary_hnsw_distance_computer_fixSL<SL>(
                code_size, flat_storage);
    });
}

namespace {

// The sample of an IndexBinaryHNSW. Hamming distances are integers, so a
// nearer node has a strictly smaller distance.
struct BinaryLinkSample : HNSWLinkSampleSearch {
    const IndexBinaryHNSW& index;
    const uint8_t* x;

    BinaryLinkSample(const IndexBinaryHNSW& index, idx_t n, const uint8_t* x)
            : index(index), x(x) {
        this->n = n;
    }

    void search(const SearchParametersHNSW* params, idx_t* labels)
            const override {
        std::vector<int32_t> distances(n);
        index.search(n, x, 1, distances.data(), labels, params);
    }

    void search_exact(
            size_t m,
            const idx_t* ids,
            const idx_t* found,
            idx_t* exact,
            uint8_t* nearer) const override {
        const size_t cs = index.code_size;
        std::vector<uint8_t> rows(m * cs);
        for (size_t j = 0; j < m; j++) {
            memcpy(rows.data() + j * cs, x + ids[j] * cs, cs);
        }
        std::vector<int32_t> distances(m);
        index.storage->search(m, rows.data(), 1, distances.data(), exact);
#pragma omp parallel
        {
            std::unique_ptr<DistanceComputer> dis(
                    index.get_distance_computer());
#pragma omp for schedule(static)
            for (int64_t j = 0; j < static_cast<int64_t>(m); j++) {
                if (exact[j] < 0 || exact[j] == found[j]) {
                    continue;
                }
                dis->set_query(
                        reinterpret_cast<const float*>(rows.data() + j * cs));
                nearer[j] = (*dis)(exact[j]) < (*dis)(found[j]);
            }
        }
    }
};

} // namespace

HNSWLevel0LinkStats IndexBinaryHNSW::add_level_0_links_from_sample(
        idx_t n,
        const uint8_t* x,
        const HNSWLevel0LinkParameters& params) {
    FAISS_THROW_IF_MSG(metric_type == METRIC_MinhashJaccard,
            "Raw MinHash Jaccard does not use Hamming-based level-0 sample refinement");
    const BinaryLinkSample sample(*this, n, x);
    std::unique_ptr<DistanceComputer> dis(get_distance_computer());
    return hnsw_add_level_0_links_from_sample(
            hnsw, ntotal, *dis, false, sample, params);
}

/**************************************************************
 * IndexBinaryHNSWCagra implementation
 **************************************************************/

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra() : IndexBinaryHNSW() {
    storage = nullptr;
}

IndexBinaryHNSWCagra::IndexBinaryHNSWCagra(int d_, int M)
        : IndexBinaryHNSW(d_, M) {
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexBinaryHNSWCagra::add(idx_t n, const uint8_t* x) {
    FAISS_THROW_IF_MSG(
            base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexBinaryHNSW::add(n, x);
}

void IndexBinaryHNSWCagra::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    validate_minhash_bitmap_mode(metric_type, d);
    if (!base_level_only) {
        IndexBinaryHNSW::search(n, x, k, distances, labels, params);
    } else {
        FAISS_THROW_IF_NOT_MSG(
                ntotal > 0, "IndexBinaryHNSWCagra: cannot search empty index");
        FAISS_THROW_IF_NOT_MSG(
                num_base_level_search_entrypoints > 0,
                "IndexBinaryHNSWCagra: "
                "num_base_level_search_entrypoints must be > 0");

        float* distances_f = (float*)distances;

        using RH = HeapBlockResultHandler<HNSW::C>;
        RH bres(n, distances_f, labels, k);

        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            dis->set_query((float*)(x + i * code_size));

            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                float distance = (*dis)(idx);

                if (distance < nearest_d[i]) {
                    nearest[i] = static_cast<storage_idx_t>(idx);
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

#pragma omp parallel
        {
            std::unique_ptr<VisitedTable> vt = VisitedTable::create(ntotal);
            std::unique_ptr<DistanceComputer> dis(get_distance_computer());
            HNSWStats search_stats;
            RH::SingleResultHandler res(bres);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                res.begin(i);
                dis->set_query((float*)(x + i * code_size));

                hnsw.search_level_0(
                        *dis,
                        res,
                        1,
                        &nearest[i],
                        &nearest_d[i],
                        1, // search_type
                        search_stats,
                        *vt,
                        params);

                res.end();
            }
            if (get_search_stats_enabled()) {
                hnsw_stats.combine_atomic(search_stats);
            }
        }

#pragma omp parallel for
        for (idx_t i = 0; i < n * k; ++i) {
            if (is_minhash_bitmap_metric(metric_type)) {
                distances[i] = minhash_bitmap_output(distances_f[i], labels[i]);
            } else if (metric_type == METRIC_MinhashJaccard) {
                distances[i] = minhash_jaccard_output(distances_f[i], labels[i]);
            } else {
                distances[i] = std::round(distances_f[i]);
            }
        }
    }
}

} // namespace faiss
