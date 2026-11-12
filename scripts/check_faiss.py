#!/usr/bin/env python3
"""Qualify the actual raw-MinHash distance, identity search and native reload."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import faiss
    from engine.mh_jaccard import pack_signatures, legacy_distance_for_matches
    assert hasattr(faiss, 'METRIC_MinhashJaccard'), 'The installed Faiss is not this modified source build'
    faiss.omp_set_num_threads(2)
    query = (np.arange(112, dtype=np.uint32) + np.uint32(0x80000000))[None, :]
    signatures = np.repeat(query, 113, axis=0)
    for matches in range(113):
        signatures[matches, matches:] ^= np.uint32(0x01000001)
    vectors = pack_signatures(signatures)
    index = faiss.IndexBinaryHNSW(3584, 96)
    index.metric_type = faiss.METRIC_MinhashJaccard
    assert index.minhash_jaccard_uses_avx512(), 'Native AVX-512 raw-MinHash kernel unavailable'
    index.hnsw.efConstruction, index.hnsw.efSearch = 384, 400
    index.add(vectors)
    q = pack_signatures(query)
    dc = index.get_distance_computer()
    dc.set_query(faiss.cast_integer_to_float_ptr(q.ctypes.data))
    expected = [legacy_distance_for_matches(i) for i in range(113)]
    assert [int(dc(i)) for i in range(113)] == expected
    assert [int(dc.symmetric_dis(112, i)) for i in range(113)] == expected
    distances, labels = index.search(q, 4)
    assert any(int(i) == 112 and int(d) == 0 for i, d in zip(labels[0], distances[0]))
    with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
        path = str(Path(directory) / 'index.faiss')
        faiss.write_index_binary(index, path)
        restored = faiss.read_index_binary(path)
        new_distances, new_labels = restored.search(q, 4)
        assert np.array_equal(new_distances, distances) and np.array_equal(new_labels, labels)
    result = dict(actual_native_distance_checked=True, equality_counts=113, identity_hit=True,
                  native_save_reload=True, metric='METRIC_MinhashJaccard', graph_M=96,
                  ef_construction=384, ef_search=400, top_k=4,
                  matches78_distance=expected[78], matches77_distance=expected[77],
                  scope='Native distance, identity search and save/reload qualification.')
    (ROOT / 'build/faiss_qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Raw-MinHash Faiss distance, search and save/reload checks passed.')


if __name__ == '__main__':
    main()
