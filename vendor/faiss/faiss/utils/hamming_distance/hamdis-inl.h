/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Restore the original bitmap HammingComputer include/dispatch chain.
// Fresh Faiss's ordinary HammingComputer*_tpl dispatch remains independent.
#ifndef FAISS_hamming_inl_h
#define FAISS_hamming_inl_h

#include <faiss/utils/hamming_distance/common.h>
#include <faiss/utils/hamming_distance/avx512-inl.h>

namespace faiss {

// Like the baseline, every bitmap length uses the default computer; no
// fixed-size Hamming-only specialization may bypass bitmap Jaccard.
template <int CODE_SIZE>
struct HammingComputer : HammingComputerDefault {
    HammingComputer(const uint8_t* a, int code_size)
            : HammingComputerDefault(a, code_size) {}
};

template <class Consumer, class... Types>
typename Consumer::T dispatch_HammingComputer(
        int /* code_size */, Consumer& consumer, Types... args) {
    return consumer.template f<HammingComputerDefault>(args...);
}

} // namespace faiss

#endif
