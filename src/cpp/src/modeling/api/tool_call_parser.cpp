// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/tool_call_parser.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

// Minimal JSON parsing for tool call bodies.
// Qwen3.5 emits: {"name": "func", "arguments": {...}}
// We extract name and arguments as strings.

namespace ov::genai::modeling {

namespace {

/// Trim whitespace from both ends of a string.
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return {};
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

/// Find a JSON string value for the given key in a JSON object string.
/// Returns the value (without quotes) or empty string if not found.
std::string find_json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return {};

    // Find the colon after key
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return {};

    // Skip whitespace
    pos = json.find_first_not_of(" \t\n\r", pos + 1);
    if (pos == std::string::npos) return {};

    if (json[pos] == '"') {
        // String value — find closing quote (handle escaped quotes)
        size_t start = pos + 1;
        size_t end = start;
        while (end < json.size()) {
            if (json[end] == '\\') {
                end += 2;  // skip escaped char
                continue;
            }
            if (json[end] == '"') break;
            ++end;
        }
        return json.substr(start, end - start);
    }

    return {};
}

/// Extract the "arguments" field as a raw JSON string (object or string).
std::string find_json_arguments(const std::string& json) {
    std::string pattern = "\"arguments\"";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return "{}";

    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return "{}";

    pos = json.find_first_not_of(" \t\n\r", pos + 1);
    if (pos == std::string::npos) return "{}";

    if (json[pos] == '{') {
        // Find matching closing brace (handle nesting)
        int depth = 0;
        bool in_string = false;
        size_t start = pos;
        for (size_t i = pos; i < json.size(); ++i) {
            char c = json[i];
            if (in_string) {
                if (c == '\\') { ++i; continue; }
                if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') { in_string = true; continue; }
            if (c == '{') ++depth;
            if (c == '}') {
                --depth;
                if (depth == 0) {
                    return json.substr(start, i - start + 1);
                }
            }
        }
        return json.substr(start);  // Unbalanced — return what we have
    }

    if (json[pos] == '"') {
        // String argument — return as-is
        size_t start = pos;
        size_t end = pos + 1;
        while (end < json.size()) {
            if (json[end] == '\\') { end += 2; continue; }
            if (json[end] == '"') { ++end; break; }
            ++end;
        }
        return json.substr(start, end - start);
    }

    return "{}";
}

}  // namespace

ToolCall ToolCallParser::parse_tool_call_body(const std::string& body) {
    ToolCall call;
    call.id = "call_" + std::to_string(call_count_);

    std::string trimmed = trim(body);

    call.name = find_json_string(trimmed, "name");
    call.arguments = find_json_arguments(trimmed);

    return call;
}

ToolParseResult ToolCallParser::process(const std::string& text) {
    if (text.empty()) {
        return {{}, {}, in_tool_call_};
    }

    ToolParseResult result;
    std::string input = buffer_ + text;
    buffer_.clear();

    size_t pos = 0;

    while (pos < input.size()) {
        if (!in_tool_call_) {
            // Look for <tool_call> tag
            auto found = input.find(kToolOpen, pos);
            if (found == std::string::npos) {
                // Check for partial match at tail
                size_t tail_start = (input.size() >= kToolOpenLen - 1)
                                        ? input.size() - (kToolOpenLen - 1)
                                        : pos;
                bool partial = false;
                for (size_t i = tail_start; i < input.size(); ++i) {
                    size_t remain = input.size() - i;
                    if (remain < kToolOpenLen &&
                        input.compare(i, remain, kToolOpen, remain) == 0) {
                        buffer_ = input.substr(i);
                        result.text += input.substr(pos, i - pos);
                        partial = true;
                        break;
                    }
                }
                if (!partial) {
                    result.text += input.substr(pos);
                }
                result.in_tool_call = in_tool_call_;
                return result;
            }
            // Text before tag is regular output
            if (found > pos) {
                result.text += input.substr(pos, found - pos);
            }
            in_tool_call_ = true;
            pos = found + kToolOpenLen;

        } else {
            // Look for </tool_call> tag
            auto found = input.find(kToolClose, pos);
            if (found == std::string::npos) {
                // Check for partial match at tail
                size_t tail_start = (input.size() >= kToolCloseLen - 1)
                                        ? input.size() - (kToolCloseLen - 1)
                                        : pos;
                bool partial = false;
                for (size_t i = tail_start; i < input.size(); ++i) {
                    size_t remain = input.size() - i;
                    if (remain < kToolCloseLen &&
                        input.compare(i, remain, kToolClose, remain) == 0) {
                        buffer_ = input.substr(i);
                        // Everything before partial match is tool call content
                        // (accumulated in buffer for when we find closing tag)
                        buffer_ = input.substr(pos);
                        partial = true;
                        break;
                    }
                }
                if (!partial) {
                    // Still in tool call, buffer everything
                    buffer_ = input.substr(pos);
                }
                result.in_tool_call = true;
                return result;
            }
            // Found closing tag — parse the tool call
            std::string body = input.substr(pos, found - pos);
            ToolCall call = parse_tool_call_body(body);
            ++call_count_;
            result.tool_calls.push_back(std::move(call));
            in_tool_call_ = false;
            pos = found + kToolCloseLen;
        }
    }

    result.in_tool_call = in_tool_call_;
    return result;
}

ToolParseResult ToolCallParser::flush() {
    ToolParseResult result;
    if (!buffer_.empty()) {
        if (in_tool_call_) {
            // Incomplete tool call — emit as regular text
            result.text = "<tool_call>" + buffer_;
        } else {
            result.text = buffer_;
        }
        buffer_.clear();
    }
    result.in_tool_call = false;
    in_tool_call_ = false;
    return result;
}

void ToolCallParser::reset() {
    in_tool_call_ = false;
    buffer_.clear();
    call_count_ = 0;
}

}  // namespace ov::genai::modeling
