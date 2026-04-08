// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <openvino/openvino.hpp>

namespace ov::genai::modeling {

/// Reusable scratch buffers for fast sampling (avoids per-step allocation).
struct SamplingContext {
    std::vector<std::pair<float, int64_t>> ranked;      // top-K candidates
    std::vector<std::pair<float, int64_t>> candidates;  // full-vocab (Case 3 only)
    std::vector<float> probs;                           // softmax probs
};

/// Extract the last token's logits from [1, S, V] into a float32 buffer.
/// Handles f32, f16, and bf16 logit tensors.
void extract_last_logits_f32(const ov::Tensor& logits, std::vector<float>& out);

/// Greedy argmax over float32 data.
int64_t argmax_f32(const std::vector<float>& data);

/// Fast multinomial sampling — O(V) + O(K log K) complexity.
///
/// Strategy (for the common case: temperature>0, top_k>0, top_p<1):
///   1. nth_element to partition top-K logits            — O(V) avg
///   2. Sort only the K candidates                       — O(K log K)
///   3. Apply temperature + softmax on K candidates only — O(K)
///   4. Apply top-P cutoff on K candidates               — O(K)
///   5. Direct CDF sampling                              — O(K)
int64_t sample_fast(const float* logits,
                    size_t vocab_size,
                    float temperature,
                    float top_p,
                    size_t top_k,
                    std::mt19937& rng,
                    SamplingContext& ctx);

}  // namespace ov::genai::modeling
