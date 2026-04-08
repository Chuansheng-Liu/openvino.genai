// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ov::genai::modeling {

/// State of the thinking block parser.
enum class ThinkingState {
    BEFORE_THINKING,  ///< Haven't seen <think> yet
    IN_THINKING,      ///< Inside <think>...</think> block
    AFTER_THINKING,   ///< Passed </think>, now emitting content
};

/// Result of processing a text delta through the thinking tracker.
struct ThinkingResult {
    std::string thinking_text;  ///< Text that belongs to the thinking block
    std::string content_text;   ///< Text that belongs to the main content
};

/// Tracks `<think>` / `</think>` boundaries in streaming text.
///
/// Qwen3.5 in thinking mode emits:
///   <think>\n...thinking...\n</think>\n\ncontent...
///
/// This tracker splits streaming text deltas into thinking vs content
/// portions, handling partial tag matches across chunk boundaries.
class ThinkingTracker {
public:
    /// Process a text delta, splitting it into thinking and content parts.
    ThinkingResult process(const std::string& text);

    /// Current state of the tracker.
    ThinkingState state() const { return state_; }

    /// Whether we are currently inside the thinking block.
    bool is_thinking() const { return state_ == ThinkingState::IN_THINKING; }

    /// Reset to initial state.
    void reset();

    /// Set state to IN_THINKING (used when <think> is in prompt, not in output).
    void start_in_thinking() { state_ = ThinkingState::IN_THINKING; }

private:
    ThinkingState state_ = ThinkingState::BEFORE_THINKING;
    std::string buffer_;  // Partial tag match buffer

    static constexpr const char* kThinkOpen = "<think>";
    static constexpr const char* kThinkClose = "</think>";
    static constexpr size_t kThinkOpenLen = 7;   // strlen("<think>")
    static constexpr size_t kThinkCloseLen = 8;  // strlen("</think>")
};

}  // namespace ov::genai::modeling
