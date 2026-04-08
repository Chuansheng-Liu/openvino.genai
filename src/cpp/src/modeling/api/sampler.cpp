// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

namespace ov::genai::modeling {

void extract_last_logits_f32(const ov::Tensor& logits, std::vector<float>& out) {
    const auto shape = logits.get_shape();
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;
    out.resize(vocab);
    if (logits.get_element_type() == ov::element::f32) {
        std::memcpy(out.data(), logits.data<const float>() + offset, vocab * sizeof(float));
    } else if (logits.get_element_type() == ov::element::f16) {
        const auto* src = logits.data<const ov::float16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* src = logits.data<const ov::bfloat16>() + offset;
        for (size_t i = 0; i < vocab; ++i)
            out[i] = static_cast<float>(src[i]);
    } else {
        throw std::runtime_error("Unsupported logits dtype for logit processing");
    }
}

int64_t argmax_f32(const std::vector<float>& data) {
    return static_cast<int64_t>(std::max_element(data.begin(), data.end()) - data.begin());
}

int64_t sample_fast(const float* logits,
                    size_t vocab_size,
                    float temperature,
                    float top_p,
                    size_t top_k,
                    std::mt19937& rng,
                    SamplingContext& ctx) {
    OPENVINO_ASSERT(vocab_size > 0, "logits must not be empty");
    OPENVINO_ASSERT(temperature > 0.0f, "temperature must be positive for sampling");

    const bool use_top_k = (top_k > 0 && top_k < vocab_size);
    const bool use_top_p = (top_p > 0.0f && top_p < 1.0f);

    // --- Case 1: No top-K, no top-P — full-vocab softmax + CDF sampling ---
    if (!use_top_k && !use_top_p) {
        float max_logit = *std::max_element(logits, logits + vocab_size);
        float inv_temp = 1.0f / temperature;

        ctx.probs.resize(vocab_size);
        float total = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float val = std::exp((logits[i] - max_logit) * inv_temp);
            ctx.probs[i] = val;
            total += val;
        }

        if (!(total > 0.0f) || !std::isfinite(total)) {
            return static_cast<int64_t>(std::max_element(logits, logits + vocab_size) - logits);
        }

        std::uniform_real_distribution<float> udist(0.0f, total);
        float dart = udist(rng);
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += ctx.probs[i];
            if (cumsum >= dart) {
                return static_cast<int64_t>(i);
            }
        }
        return static_cast<int64_t>(vocab_size - 1);
    }

    // --- Case 2: top-K (with optional top-P) — the common path ---
    if (use_top_k) {
        size_t k = std::min(top_k, vocab_size);
        ctx.ranked.resize(k);

        for (size_t i = 0; i < k; ++i) {
            ctx.ranked[i] = {logits[i], static_cast<int64_t>(i)};
        }

        size_t min_pos = 0;
        float min_val = ctx.ranked[0].first;
        for (size_t i = 1; i < k; ++i) {
            if (ctx.ranked[i].first < min_val) {
                min_val = ctx.ranked[i].first;
                min_pos = i;
            }
        }

        for (size_t i = k; i < vocab_size; ++i) {
            if (logits[i] > min_val) {
                ctx.ranked[min_pos] = {logits[i], static_cast<int64_t>(i)};
                min_val = ctx.ranked[0].first;
                min_pos = 0;
                for (size_t j = 1; j < k; ++j) {
                    if (ctx.ranked[j].first < min_val) {
                        min_val = ctx.ranked[j].first;
                        min_pos = j;
                    }
                }
            }
        }

        std::sort(ctx.ranked.begin(), ctx.ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float max_logit = ctx.ranked[0].first;
        float inv_temp = 1.0f / temperature;
        ctx.probs.resize(k);
        float total = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            float val = std::exp((ctx.ranked[i].first - max_logit) * inv_temp);
            ctx.probs[i] = val;
            total += val;
        }

        if (!(total > 0.0f) || !std::isfinite(total)) {
            return ctx.ranked[0].second;
        }

        size_t num_candidates = k;
        if (use_top_p) {
            float threshold = top_p * total;
            float cumsum = 0.0f;
            for (size_t i = 0; i < k; ++i) {
                cumsum += ctx.probs[i];
                if (cumsum >= threshold) {
                    num_candidates = i + 1;
                    total = cumsum;
                    break;
                }
            }
        }
        num_candidates = std::max<size_t>(1, num_candidates);

        std::uniform_real_distribution<float> udist(0.0f, total);
        float dart = udist(rng);
        float cumsum = 0.0f;
        for (size_t i = 0; i < num_candidates; ++i) {
            cumsum += ctx.probs[i];
            if (cumsum >= dart) {
                return ctx.ranked[i].second;
            }
        }
        return ctx.ranked[num_candidates - 1].second;
    }

    // --- Case 3: top-P only (no top-K) ---
    ctx.candidates.resize(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i] = {logits[i], static_cast<int64_t>(i)};
    }

    float max_logit = std::max_element(ctx.candidates.begin(), ctx.candidates.end(),
                                       [](const auto& a, const auto& b) { return a.first < b.first; })->first;
    float inv_temp = 1.0f / temperature;
    float total = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        ctx.candidates[i].first = std::exp((ctx.candidates[i].first - max_logit) * inv_temp);
        total += ctx.candidates[i].first;
    }

    if (!(total > 0.0f) || !std::isfinite(total)) {
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] == max_logit) return static_cast<int64_t>(i);
        }
        return static_cast<int64_t>(0);
    }

    float threshold = top_p * total;

    size_t num_candidates = vocab_size;
    for (size_t step = 16; step <= 1024; step *= 2) {
        if (vocab_size <= step) break;
        std::partial_sort(ctx.candidates.begin(),
                          ctx.candidates.begin() + static_cast<ptrdiff_t>(step),
                          ctx.candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float cumsum = 0.0f;
        for (size_t i = 0; i < step; ++i) {
            cumsum += ctx.candidates[i].first;
            if (cumsum >= threshold) {
                num_candidates = i + 1;
                total = cumsum;
                goto nucleus_found;
            }
        }
    }
    std::sort(ctx.candidates.begin(), ctx.candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    {
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += ctx.candidates[i].first;
            if (cumsum >= threshold) {
                num_candidates = i + 1;
                total = cumsum;
                break;
            }
        }
    }

nucleus_found:
    num_candidates = std::max<size_t>(1, num_candidates);
    std::uniform_real_distribution<float> udist(0.0f, total);
    float dart = udist(rng);
    float cumsum = 0.0f;
    for (size_t i = 0; i < num_candidates; ++i) {
        cumsum += ctx.candidates[i].first;
        if (cumsum >= dart) {
            return ctx.candidates[i].second;
        }
    }
    return ctx.candidates[num_candidates - 1].second;
}

}  // namespace ov::genai::modeling
