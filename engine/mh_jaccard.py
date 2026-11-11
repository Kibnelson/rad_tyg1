"""Raw MinHash storage and independent arithmetic for the MH experiment.

Packing preserves the historical big-endian uint32 bytes. These bytes are
opaque storage: Bloom routing continues to receive the original sketches.
"""
import numpy as np

POSITIONS = 112
STORAGE_BYTES = POSITIONS * 4
STORAGE_BITS = STORAGE_BYTES * 8
REPRESENTATION = dict(kind='raw_minhash_uint32_be', positions=POSITIONS,
    bytes_per_position=4, storage_bytes=STORAGE_BYTES,
    distance_rule='legacy_scaled_floor_v1')


def pack_signatures(sketches):
    values = np.asarray(sketches)
    if values.dtype != np.dtype('uint32') or values.ndim != 2 or values.shape[1] != POSITIONS:
        raise ValueError('Raw MinHash requires uint32[N,112] signatures without truncation')
    return np.ascontiguousarray(values.astype('>u4').view(np.uint8).reshape(len(values), STORAGE_BYTES))


def unpack_signatures(vectors):
    values = np.asarray(vectors)
    if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != STORAGE_BYTES:
        raise ValueError('Raw MinHash storage requires uint8[N,448] vectors')
    return np.ascontiguousarray(values).view('>u4').reshape(len(values), POSITIONS).astype(np.uint32)


def legacy_distance_for_matches(matches):
    """Independent float32 oracle; final multiply/add mirrors the saved FMA.

    The saved native build contracts ``distance*100 + 1e-6f`` into one FMA.
    Binary64 evaluates that small expression exactly enough before the single
    float32 rounding. Native qualification also compares the frozen old C++
    implementation, rather than relying only on this Python oracle.
    """
    if isinstance(matches, bool) or not isinstance(matches, (int, np.integer)) or not 0 <= matches <= POSITIONS:
        raise ValueError('Matching positions must be an integer from 0 through 112')
    similarity = np.float32(matches) / np.float32(POSITIONS)
    distance = np.float32(1) - similarity
    scaled = np.float32(float(distance) * 100.0 + float(np.float32(1e-6)))
    return int(np.floor(scaled))


def positional_distance(a, b):
    values = np.asarray([a, b])
    if values.dtype != np.dtype('uint32') or values.shape != (2, POSITIONS):
        raise ValueError('Compare two uint32[112] signatures')
    return legacy_distance_for_matches(int(np.count_nonzero(values[0] == values[1])))
