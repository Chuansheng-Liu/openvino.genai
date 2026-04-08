// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/token_processor.hpp"

#include <cstring>

namespace ov::genai::modeling {

TokenProcessor::TokenProcessor(ov::genai::Tokenizer& tokenizer)
    : tokenizer_(tokenizer) {}

bool TokenProcessor::ends_with_replacement_char(const std::string& text) {
    // U+FFFD replacement character in UTF-8: 0xEF 0xBF 0xBD
    constexpr char replacement[] = "\xef\xbf\xbd";
    return text.size() >= 3 &&
           text.compare(text.size() - 3, 3, replacement) == 0;
}

std::string TokenProcessor::process(int64_t token_id) {
    tokens_.push_back(token_id);

    // Re-decode entire token cache from scratch (ensures correctness when
    // tokens combine to form different characters than individually decoded).
    last_decoded_ = tokenizer_.decode(tokens_, ov::genai::skip_special_tokens(true));

    // If the decoded text ends with the replacement character, the last
    // byte sequence is incomplete UTF-8 — don't emit anything yet.
    if (ends_with_replacement_char(last_decoded_)) {
        return {};
    }

    // Newline: flush everything immediately.
    if (!last_decoded_.empty() && last_decoded_.back() == '\n' &&
        last_decoded_.size() > printed_len_) {
        std::string delta(last_decoded_, printed_len_,
                          last_decoded_.size() - printed_len_);
        // Reset cache for a fresh start after newline.
        tokens_.clear();
        last_decoded_.clear();
        printed_len_ = 0;
        return delta;
    }

    // Delay: don't emit the last kDelayTokens worth of text, since
    // adding future tokens could retroactively change decoded output.
    if (tokens_.size() <= kDelayTokens) {
        return {};
    }

    // Decode the "stable" prefix (everything except last kDelayTokens).
    // We decode up to (size - kDelayTokens) tokens to find stable length.
    std::vector<int64_t> stable_tokens(tokens_.begin(),
                                       tokens_.end() - kDelayTokens);
    std::string stable_text = tokenizer_.decode(stable_tokens,
                                                ov::genai::skip_special_tokens(true));

    if (ends_with_replacement_char(stable_text)) {
        // Even the stable prefix is incomplete — wait for more tokens.
        return {};
    }

    size_t stable_len = stable_text.size();
    if (stable_len > printed_len_) {
        std::string delta(last_decoded_, printed_len_,
                          stable_len - printed_len_);
        printed_len_ = stable_len;
        return delta;
    }

    return {};
}

std::string TokenProcessor::flush() {
    if (tokens_.empty()) {
        return {};
    }

    last_decoded_ = tokenizer_.decode(tokens_, ov::genai::skip_special_tokens(true));

    std::string delta;
    if (last_decoded_.size() > printed_len_) {
        delta.assign(last_decoded_, printed_len_,
                     last_decoded_.size() - printed_len_);
    }

    tokens_.clear();
    last_decoded_.clear();
    printed_len_ = 0;
    return delta;
}

void TokenProcessor::reset() {
    tokens_.clear();
    last_decoded_.clear();
    printed_len_ = 0;
}

}  // namespace ov::genai::modeling
