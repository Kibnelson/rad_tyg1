"""Feature contract for the proposed Milvus baseline, without a server dependency.

Raw tokens are complete five-word shingles, matching RAD's text feature space.
Knowhere's native MinHash refinement compares signatures, so exact text Jaccard
must remain an explicit, separately timed operation on these raw shingle sets.
"""
from collections.abc import Mapping
from fractions import Fraction
import json
import operator

import numpy as np

from .encoding import SketchSpec

TOKEN_SET_FIELD = 'token_set'
TOKEN_OVERFLOW_FIELD = 'token_set_overflow'
VARCHAR_BYTES = 65535
TOKEN_OVERFLOW_CAPACITY = 4096


def hit_document_id(hit):
    """Read the collection's named Int64 key, with legacy top-level id support."""
    if not isinstance(hit, Mapping):
        raise ValueError('Expected a mapping for a Milvus search hit')
    # Test membership before indexing: SDK Hit.__getitem__ otherwise searches
    # entity output fields too, where an unrelated field named id may exist.
    key = 'doc_id' if 'doc_id' in hit else 'id'
    if key not in hit:
        raise ValueError("Milvus search hit is missing primary key 'doc_id' (or legacy 'id')")
    value = hit[key]
    if isinstance(value, bool):
        raise ValueError('Milvus document primary key must be an integer')
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError('Milvus document primary key must be an integer') from error


def raw_shingles(text, window=5):
    if not isinstance(text, str):
        raise TypeError('Expected unmodified document text')
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError('The shingle window must be a positive integer')
    words = text.split()
    if not words:
        return frozenset()
    return frozenset(' '.join(words[i:i + window])
                     for i in range(max(1, len(words) - window + 1)))


def serialize_raw_shingles(shingles):
    # JSON keeps each entire shingle intact; joining shingles with spaces would
    # erase the token boundaries. Never truncate long sets to fit VARCHAR.
    shingles = list(shingles)
    if any(not isinstance(token, str) for token in shingles):
        raise TypeError('Raw shingles must be strings')
    return json.dumps(sorted(set(shingles)), ensure_ascii=False, separators=(',', ':'))


def deserialize_raw_shingles(payload):
    values = json.loads(payload)
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError('Expected a JSON array of raw shingle strings')
    return frozenset(values)


def token_set_fields(shingles,max_bytes=VARCHAR_BYTES,max_overflow=TOKEN_OVERFLOW_CAPACITY):
    """Lossless JSON in one VARCHAR plus optional ARRAY<VARCHAR> continuations."""
    if not 4<=max_bytes<=VARCHAR_BYTES or not 0<=max_overflow<=TOKEN_OVERFLOW_CAPACITY:
        raise ValueError('Invalid Milvus token-set field limits')
    raw=serialize_raw_shingles(shingles).encode('utf-8')
    chunks=[]
    start=0
    while start<len(raw):
        end=min(start+max_bytes,len(raw))
        while end<len(raw) and raw[end]&0xc0==0x80:
            end-=1
        chunks.append(raw[start:end].decode('utf-8'))
        if len(chunks)>max_overflow+1:
            raise ValueError('Token set exceeds Milvus VARCHAR continuation capacity; no tokens were truncated')
        start=end
    return {TOKEN_SET_FIELD:chunks[0],TOKEN_OVERFLOW_FIELD:chunks[1:]}


def token_set_from_fields(row,max_bytes=VARCHAR_BYTES,max_overflow=TOKEN_OVERFLOW_CAPACITY):
    first=row.get(TOKEN_SET_FIELD)
    tail=row.get(TOKEN_OVERFLOW_FIELD)
    if (not isinstance(first,str) or not isinstance(tail,list) or len(tail)>max_overflow
            or any(not isinstance(piece,str) for piece in tail)):
        raise ValueError('Milvus response is missing complete token-set VARCHAR fields')
    chunks=[first,*tail]
    if any(len(piece.encode('utf-8'))>max_bytes for piece in chunks):
        raise ValueError('Milvus token-set VARCHAR exceeds its declared UTF-8 byte limit')
    return deserialize_raw_shingles(''.join(chunks))


def exact_jaccard(left, right):
    left, right = frozenset(left), frozenset(right)
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def passes_jaccard(left, right, threshold='0.70'):
    # Integer comparison preserves the inclusive boundary at exactly 0.70.
    bound = Fraction(str(threshold))
    if not 0 <= bound <= 1:
        raise ValueError('Jaccard threshold must be between zero and one')
    left, right = frozenset(left), frozenset(right)
    union = len(left | right)
    return (len(left & right) * bound.denominator >= union * bound.numerator
            if union else True)


def signature_bytes(sketch, spec=SketchSpec()):
    """Encode the existing 112 uint32 hash positions, not RAD's 4096-bit bitmap."""
    values = np.asarray(sketch)
    if values.shape != (spec.K,) or values.dtype.kind not in 'iu':
        raise ValueError(f'Expected {spec.K} integer MinHash positions')
    if values.min() < 0 or values.max() > 0xffffffff:
        raise ValueError('MinHash values must fit uint32 without truncation')
    return values.astype('<u4').tobytes()
