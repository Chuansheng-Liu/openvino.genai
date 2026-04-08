// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/thinking_tracker.hpp"

#include <algorithm>
#include <cstring>

namespace ov::genai::modeling {

ThinkingResult ThinkingTracker::process(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    ThinkingResult result;

    // Prepend any leftover buffer from partial tag matches.
    std::string input = buffer_ + text;
    buffer_.clear();

    size_t pos = 0;

    while (pos < input.size()) {
        if (state_ == ThinkingState::BEFORE_THINKING) {
            // Look for <think> tag
            auto found = input.find(kThinkOpen, pos);
            if (found == std::string::npos) {
                // Check if the tail could be a partial match for "<think>"
                size_t tail_start = (input.size() >= kThinkOpenLen - 1)
                                        ? input.size() - (kThinkOpenLen - 1)
                                        : pos;
                bool partial = false;
                for (size_t i = tail_start; i < input.size(); ++i) {
                    size_t remain = input.size() - i;
                    if (remain < kThinkOpenLen &&
                        input.compare(i, remain, kThinkOpen, remain) == 0) {
                        buffer_ = input.substr(i);
                        // Everything before the partial match is content
                        // (in BEFORE_THINKING, text before <think> is discarded
                        //  or treated as content if thinking never starts)
                        result.content_text += input.substr(pos, i - pos);
                        partial = true;
                        break;
                    }
                }
                if (!partial) {
                    // No <think> found at all — everything is content
                    result.content_text += input.substr(pos);
                }
                return result;
            }
            // Text before <think> is content (pre-think text)
            if (found > pos) {
                result.content_text += input.substr(pos, found - pos);
            }
            state_ = ThinkingState::IN_THINKING;
            pos = found + kThinkOpenLen;

        } else if (state_ == ThinkingState::IN_THINKING) {
            // Look for </think> tag
            auto found = input.find(kThinkClose, pos);
            if (found == std::string::npos) {
                // Check for partial match at the tail
                size_t tail_start = (input.size() >= kThinkCloseLen - 1)
                                        ? input.size() - (kThinkCloseLen - 1)
                                        : pos;
                bool partial = false;
                for (size_t i = tail_start; i < input.size(); ++i) {
                    size_t remain = input.size() - i;
                    if (remain < kThinkCloseLen &&
                        input.compare(i, remain, kThinkClose, remain) == 0) {
                        buffer_ = input.substr(i);
                        result.thinking_text += input.substr(pos, i - pos);
                        partial = true;
                        break;
                    }
                }
                if (!partial) {
                    // Still in thinking, accumulate all
                    result.thinking_text += input.substr(pos);
                }
                return result;
            }
            // Text before </think> is thinking
            if (found > pos) {
                result.thinking_text += input.substr(pos, found - pos);
            }
            state_ = ThinkingState::AFTER_THINKING;
            pos = found + kThinkCloseLen;

        } else {
            // AFTER_THINKING — everything is content
            result.content_text += input.substr(pos);
            return result;
        }
    }

    return result;
}

void ThinkingTracker::reset() {
    state_ = ThinkingState::BEFORE_THINKING;
    buffer_.clear();
}

}  // namespace ov::genai::modeling
