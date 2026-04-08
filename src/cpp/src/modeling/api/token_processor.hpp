// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "openvino/genai/tokenizer.hpp"

namespace ov::genai::modeling {

/// Handles UTF-8 safe incremental detokenization.
///
/// Accumulates token IDs, re-decodes the full cache each step, and emits
/// only the delta text since the last emission. Incomplete UTF-8 sequences
/// are buffered until the next token completes them.
///
/// Follows the same "delay N tokens" approach as ov::genai::TextStreamer.
class TokenProcessor {
public:
    explicit TokenProcessor(ov::genai::Tokenizer& tokenizer);

    /// Process a new token. Returns the text delta to emit (may be empty
    /// if the decoded text ends with an incomplete UTF-8 sequence).
    std::string process(int64_t token_id);

    /// Flush any remaining buffered text. Call at end of generation.
    std::string flush();

    /// Reset state for a new generation round.
    void reset();

    /// All accumulated token IDs so far.
    const std::vector<int64_t>& token_ids() const { return tokens_; }

    /// Full decoded text so far (including any unprinted tail).
    const std::string& decoded_text() const { return last_decoded_; }

private:
    static bool ends_with_replacement_char(const std::string& text);

    ov::genai::Tokenizer& tokenizer_;
    std::vector<int64_t> tokens_;
    std::string last_decoded_;
    size_t printed_len_ = 0;

    // Delay emission by a few tokens to handle cases where adding
    // a token can retroactively shorten the decoded text.
    static constexpr size_t kDelayTokens = 3;
};

}  // namespace ov::genai::modeling
