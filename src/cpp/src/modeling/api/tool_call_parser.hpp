// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace ov::genai::modeling {

/// A parsed tool call extracted from model output.
struct ToolCall {
    std::string id;         ///< Unique call ID (e.g., "call_0", "call_1")
    std::string name;       ///< Function name
    std::string arguments;  ///< JSON string of arguments
};

/// Result of processing text through the tool call parser.
struct ToolParseResult {
    std::string text;                 ///< Regular text (non-tool-call content)
    std::vector<ToolCall> tool_calls; ///< Completed tool calls found
    bool in_tool_call = false;        ///< Currently accumulating a tool call
};

/// Parses Qwen3.5 tool call XML format from streaming text.
///
/// Qwen3.5 emits tool calls in this format:
/// ```
/// <tool_call>
/// {"name": "func_name", "arguments": {"arg1": "val1"}}
/// </tool_call>
/// ```
///
/// This parser accumulates text between `<tool_call>` and `</tool_call>` tags,
/// then parses the JSON content into structured ToolCall objects.
/// Handles partial tags across chunk boundaries.
class ToolCallParser {
public:
    /// Process a text delta, extracting any tool calls.
    ToolParseResult process(const std::string& text);

    /// Flush: if we're mid-tool-call, emit what we have as regular text.
    ToolParseResult flush();

    /// Reset state for new generation.
    void reset();

    /// Number of tool calls parsed so far.
    int call_count() const { return call_count_; }

private:
    /// Parse a complete tool call body (JSON between the tags).
    ToolCall parse_tool_call_body(const std::string& body);

    bool in_tool_call_ = false;
    std::string buffer_;       // Partial tag or tool call content
    int call_count_ = 0;

    static constexpr const char* kToolOpen = "<tool_call>";
    static constexpr const char* kToolClose = "</tool_call>";
    static constexpr size_t kToolOpenLen = 11;   // strlen("<tool_call>")
    static constexpr size_t kToolCloseLen = 12;  // strlen("</tool_call>")
};

}  // namespace ov::genai::modeling
