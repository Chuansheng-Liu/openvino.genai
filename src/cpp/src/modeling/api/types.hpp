// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ov::genai::modeling {

/// Sampling parameters for token selection.
struct SamplingParams {
    float temperature = 0.0f;       // 0.0 = greedy
    float top_p = 0.95f;
    size_t top_k = 20;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float min_p = 0.0f;
    size_t rng_seed = 0;            // 0 = random_device
};

/// Generation parameters controlling inference behavior.
struct GenerateParams {
    int max_new_tokens = 256;
    SamplingParams sampling;
    bool enable_thinking = true;
    std::vector<std::string> stop_strings;
    int timeout_ms = 0;             // 0 = no timeout
};

/// Stream event types for progressive output.
enum class StreamEvent {
    PREFILL_DONE,       // prefill complete, decode about to start
    TOKEN,              // a token was generated
    THINKING_START,     // entered <think> block
    THINKING_END,       // exited </think> block
    FINISH,             // generation complete
};

/// Reasons for stopping generation.
enum class StopReason {
    EOS,                // natural end (stop token)
    MAX_TOKENS,         // reached max_new_tokens
    STOP_STRING,        // matched a stop string
    USER_STOP,          // user called stop()
    TIMEOUT,            // timeout_ms exceeded
    ERROR,              // inference error
};

/// Data block delivered via streaming callback.
struct StreamChunk {
    StreamEvent event;

    // TOKEN event fields
    int64_t token_id = -1;
    std::string token_text;
    bool is_thinking = false;

    // FINISH event fields (also available as GenerateResult)
    StopReason stop_reason = StopReason::EOS;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int thinking_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    double throughput = 0.0;
};

/// Callback for streaming output. Return false to stop generation.
using StreamCallback = std::function<bool(const StreamChunk& chunk)>;

/// Final generation result returned by Session::generate().
struct GenerateResult {
    std::vector<int64_t> token_ids;
    std::string text;               // final answer (excluding thinking)
    std::string thinking_text;      // thinking block text (if any)
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int thinking_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    double throughput = 0.0;
    StopReason stop_reason = StopReason::EOS;
};

}  // namespace ov::genai::modeling
