// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ov_serve: OpenAI-compatible HTTP server for Qwen3.5 Modeling API
//
// Endpoints:
//   POST /v1/chat/completions   — Chat (multi-turn, tools, streaming SSE)
//   POST /v1/completions        — Text completion
//   GET  /v1/models             — Model list
//   GET  /health                — Health check
//
// Usage:
//   ov_serve --model C:\models\Qwen3.5-4B --port 8080 --workers 1

// Include modeling API headers FIRST (before httplib which pulls in Windows headers
// that define macros like ERROR and TIMEOUT conflicting with our enums)
#include "modeling/api/model_loader.hpp"
#include "modeling/api/session.hpp"
#include "modeling/api/thinking_tracker.hpp"
#include "modeling/api/tool_call_parser.hpp"
#include "modeling/api/types.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "dflash_engine.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// cpp-httplib MUST come after modeling headers to avoid Windows macro conflicts
// (ERROR, TIMEOUT etc. are #defined by Windows headers that httplib includes)
#ifdef ERROR
#undef ERROR
#endif
#ifdef TIMEOUT
#undef TIMEOUT
#endif
#include "httplib.h"

using json = nlohmann::json;
using namespace ov::genai::modeling;

// stb_image for decoding base64 images in memory
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ═══════════════════════════════════════════════════════════════════
//  Base64 decode + image loading from data: URI
// ═══════════════════════════════════════════════════════════════════

static const uint8_t kBase64Table[256] = {
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
    52,53,54,55,56,57,58,59,60,61,64,64,64, 0,64,64,
    64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
    64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
};

static std::vector<uint8_t> base64_decode(const std::string& input) {
    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (char c : input) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        uint8_t val = kBase64Table[static_cast<uint8_t>(c)];
        if (val == 64) continue;  // skip invalid chars
        accum = (accum << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

/// Decode a data: URI (base64 image) to an ov::Tensor [H, W, 3] uint8.
/// Supports "data:image/...;base64,..." format.
static ov::Tensor decode_image_from_data_uri(const std::string& uri) {
    // Find base64 payload after "base64,"
    auto pos = uri.find("base64,");
    if (pos == std::string::npos) {
        throw std::runtime_error("Image URL must be a data: URI with base64 encoding");
    }
    std::string b64_data = uri.substr(pos + 7);

    auto raw_bytes = base64_decode(b64_data);
    if (raw_bytes.empty()) {
        throw std::runtime_error("Base64 decode produced empty data");
    }

    int w = 0, h = 0, channels = 0;
    constexpr int desired_channels = 3;
    unsigned char* pixels = stbi_load_from_memory(
        raw_bytes.data(), static_cast<int>(raw_bytes.size()),
        &w, &h, &channels, desired_channels);
    if (!pixels) {
        throw std::runtime_error(std::string("Failed to decode image: ") +
                                 stbi_failure_reason());
    }

    // Create ov::Tensor that owns the pixel data
    size_t byte_count = static_cast<size_t>(h) * w * desired_channels;
    ov::Tensor image(ov::element::u8, {static_cast<size_t>(h),
                                       static_cast<size_t>(w),
                                       static_cast<size_t>(desired_channels)});
    std::memcpy(image.data(), pixels, byte_count);
    stbi_image_free(pixels);
    return image;
}

// ═══════════════════════════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════════════════════════

struct ServerConfig {
    std::string model_path;
    std::string dflash_model_path;   // DFlash draft model dir (empty = disabled)
    std::string device = "GPU";
    int port = 8080;
    std::string host = "0.0.0.0";
    int workers = 1;
    bool enable_thinking = true;
    bool enable_vision = false;
    int max_tokens_default = 2048;
    float repetition_penalty = 1.0f;  // 1.0 = no penalty
    float presence_penalty = 0.0f;    // 0.0 = off
    float frequency_penalty = 0.0f;   // 0.0 = off
    float temperature = 0.7f;         // standard default
    float top_p = 0.95f;              // standard default
    size_t top_k = 0;                 // 0 = disabled (no top-k filtering)
    float min_temperature = 0.0f;     // 0 = no override; set via --min-temp
    int warmup_tokens = 0;            // 0 = disable warmup
    bool enable_logging = true;       // Log prompts and request params to stderr
};

/// Check if an exception is a GPU out-of-memory error (CL_OUT_OF_RESOURCES).
static bool is_gpu_oom(const std::exception& e) {
    return std::string(e.what()).find("CL_OUT_OF_RESOURCES") != std::string::npos;
}

// ═══════════════════════════════════════════════════════════════════
//  Worker Pool — owns N Sessions, thread-safe acquire/release
// ═══════════════════════════════════════════════════════════════════

class WorkerPool {
public:
    WorkerPool(ModelLoader& loader, int n) : loader_(loader) {
        for (int i = 0; i < n; ++i) {
            sessions_.push_back(std::make_unique<Session>(loader));
            available_.push(sessions_.back().get());
        }
    }

    Session* acquire() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !available_.empty(); });
        auto* s = available_.front();
        available_.pop();
        return s;
    }

    void release(Session* s) {
        // Don't call s->reset() — preserve KV cache for prefix reuse.
        std::lock_guard<std::mutex> lock(mu_);
        available_.push(s);
        cv_.notify_one();
    }

    /// Warmup all sessions to pre-allocate GPU memory.
    void warmup(int warmup_tokens) {
        for (auto& s : sessions_) {
            s->warmup(warmup_tokens);
        }
    }

    /// Recreate a session after GPU error (e.g., CL_OUT_OF_RESOURCES).
    /// Caller must hold the session (acquired, not yet released).
    void recreate_session(Session* s, int warmup_tokens) {
        std::cerr << "[ov_serve] Recreating session after GPU error\n";
        s->recreate();
        if (warmup_tokens > 0) {
            s->warmup(warmup_tokens);
        }
    }

    size_t size() const { return sessions_.size(); }

    /// RAII guard — ensures session is always released back to pool.
    class Guard {
    public:
        Guard(WorkerPool& pool) : pool_(pool), session_(pool.acquire()) {}
        ~Guard() { if (session_) pool_.release(session_); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        Session* get() { return session_; }
        Session* operator->() { return session_; }
    private:
        WorkerPool& pool_;
        Session* session_;
    };

private:
    ModelLoader& loader_;
    std::vector<std::unique_ptr<Session>> sessions_;
    std::queue<Session*> available_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// ═══════════════════════════════════════════════════════════════════
//  Utility: Generate unique request ID
// ═══════════════════════════════════════════════════════════════════

static std::atomic<uint64_t> g_request_counter{0};

static std::string make_request_id() {
    return "chatcmpl-" + std::to_string(g_request_counter.fetch_add(1));
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ═══════════════════════════════════════════════════════════════════
//  Parse OpenAI request → GenerateParams + prompt
// ═══════════════════════════════════════════════════════════════════

struct ParsedRequest {
    std::string prompt;              // rendered chat prompt
    GenerateParams params;
    bool stream = false;
    std::string model_name;
    std::vector<json> tools;         // tool definitions (passed to chat template)
    std::vector<ov::Tensor> images;  // decoded images (max 1 currently)
};

static ParsedRequest parse_chat_request(const json& body, ov::genai::Tokenizer& tokenizer,
                                        const ServerConfig& cfg) {
    ParsedRequest req;
    req.stream = body.value("stream", false);
    req.model_name = body.value("model", "qwen3.5");

    // Extract tool definitions
    if (body.contains("tools")) {
        req.tools = body["tools"].get<std::vector<json>>();
        if (cfg.enable_logging)
            std::cerr << "[ov_serve] tools: " << req.tools.size() << " tool(s) provided\n";
    }

    // Build prompt: use native chat template for text-only requests;
    // for VL (image) requests, construct ChatML manually so that
    // tokenize_vl can expand <|vision_start|><|vision_end|> markers
    // with the correct pad count based on the actual image grid_thw.
    auto messages = body.at("messages");

    // First pass: extract images from multimodal content arrays
    for (const auto& msg : messages) {
        if (!msg.contains("content") || !msg["content"].is_array()) continue;
        for (const auto& part : msg["content"]) {
            if (part.value("type", "") == "image_url") {
                auto url = part.at("image_url").at("url").get<std::string>();
                req.images.push_back(decode_image_from_data_uri(url));
            }
        }
    }

    // Multi-image: all images are kept and passed to the session.
    // Each image gets a <|vision_start|><|vision_end|> marker in the prompt.
    const size_t total_images = req.images.size();

    // Prepend thinking tags to historical assistant messages so that the
    // tokenized prompt matches what was originally generated (the model's
    // output always starts with <think>\n\n</think>\n\n when thinking is
    // disabled, or <think>\n...thoughts...</think>\n\n when enabled).
    // Without this, prefix cache would miss because the cached KV sequence
    // contains these thinking tokens but the rebuilt prompt does not.
    const std::string think_prefix = "<think>\n\n</think>\n\n";
    for (size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].value("role", "") == "assistant"
            && i + 1 < messages.size()) {  // not the last message (which is the new assistant turn)
            // Normalize null / missing content to empty string so that the
            // think-prefix is always present (the model expects <think>
            // tags in every assistant turn; omitting them confuses it at
            // long context lengths, e.g. Hermes agent interrupted turns).
            std::string c;
            if (messages[i].contains("content") && messages[i]["content"].is_string())
                c = messages[i]["content"].get<std::string>();
            if (c.find("<think>") == std::string::npos) {
                messages[i]["content"] = think_prefix + c;
            }
        }
    }

    if (req.images.empty()) {
        // Text-only: build ChatML manually (same as VL path but with
        // vision markers for historical images) so that we can control
        // thinking tag placement for prefix cache compatibility.
        // Historical image markers will be expanded by tokenize_text using
        // stored per-image pad counts.

         // Build tool definition block (Qwen3.5 format) — will be emitted as a
        // separate system message near the end of the context so the model attends
        // to it even when the primary system prompt is very long (e.g. Hermes agent).
        std::string tool_defs;
        if (!req.tools.empty()) {
            tool_defs = "# Tools\n\nYou may call one or more functions to assist "
                        "with the user query.\n\nYou are provided with function signatures "
                        "within <tools></tools> XML tags:\n<tools>";
            for (const auto& tool : req.tools) {
                tool_defs += "\n" + tool.dump();
            }
            tool_defs += "\n</tools>\n\nFor each function call, return a json object with "
                         "function name and arguments within <tool_call></tool_call> XML tags:\n"
                         "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                         "</tool_call>";
        }

        std::string chat_text;
        bool in_tool_group = false;  // track consecutive tool messages
        bool tool_defs_emitted = false;  // emit tool defs once, right after first system message
        for (size_t mi = 0; mi < messages.size(); ++mi) {
            const auto& msg = messages[mi];
            std::string role = msg.at("role").get<std::string>();

            // Handle tool response messages: wrap in <tool_response> inside user block
            if (role == "tool") {
                std::string tc = msg.contains("content") && msg["content"].is_string()
                    ? msg["content"].get<std::string>() : "";
                if (!in_tool_group) {
                    chat_text += "<|im_start|>user";
                    in_tool_group = true;
                }
                chat_text += "\n<tool_response>\n" + tc + "\n</tool_response>";
                bool next_is_tool = (mi + 1 < messages.size()
                    && messages[mi + 1].value("role", "") == "tool");
                if (!next_is_tool) {
                    chat_text += "<|im_end|>\n";
                    in_tool_group = false;
                }
                continue;
            }

            std::string content;
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    // Extract text and add vision markers for historical images
                    std::string text_parts;
                    std::string vision_markers;
                    for (const auto& part : msg["content"]) {
                        std::string pt = part.value("type", "");
                        if (pt == "text") {
                            if (!text_parts.empty()) text_parts += "\n";
                            text_parts += part.at("text").get<std::string>();
                        } else if (pt == "image_url") {
                            vision_markers += "<|vision_start|><|vision_end|>";
                        }
                    }
                    content = vision_markers + text_parts;
                } else if (msg["content"].is_null()) {
                    content = "";
                }
            }

            chat_text += "<|im_start|>" + role + "\n" + content;

            // Append tool_calls for assistant messages (Qwen3.5 format)
            if (role == "assistant" && msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
                for (size_t ti = 0; ti < msg["tool_calls"].size(); ++ti) {
                    const auto& tc = msg["tool_calls"][ti];
                    json func = tc.contains("function") ? tc["function"] : tc;
                    std::string tc_name = func.value("name", "");
                    std::string tc_args = func.contains("arguments")
                        ? (func["arguments"].is_string() ? func["arguments"].get<std::string>()
                                                         : func["arguments"].dump())
                        : "{}";
                    if ((ti == 0 && !content.empty() && content.back() != '\n') || ti > 0) {
                        chat_text += "\n";
                    }
                    chat_text += "<tool_call>\n{\"name\": \"" + tc_name
                              + "\", \"arguments\": " + tc_args + "}\n</tool_call>";
                }
            }

            chat_text += "<|im_end|>\n";

            // Emit tool definitions as a separate system message right after the
            // first system message.  This keeps them at a FIXED position in the
            // token sequence so that the prefix cache can reuse KV entries across
            // multi-turn conversations (the system prompt + tools block is the
            // stable prefix shared by all requests in a session).
            if (role == "system" && !tool_defs.empty() && !tool_defs_emitted) {
                chat_text += "<|im_start|>system\n" + tool_defs + "<|im_end|>\n";
                tool_defs_emitted = true;
            }
        }

        // Fallback: if no system message was seen, emit tool defs before assistant turn
        if (!tool_defs.empty() && !tool_defs_emitted) {
            chat_text += "<|im_start|>system\n" + tool_defs + "<|im_end|>\n";
        }

        chat_text += "<|im_start|>assistant\n";
        if (cfg.enable_thinking) {
            chat_text += "<think>\n";
        } else {
            chat_text += "<think>\n\n</think>\n\n";
        }
        req.prompt = chat_text;
    } else {
        // VL: manual ChatML with vision markers so tokenize_vl can
        // expand them to the correct number of <|image_pad|> tokens.
        // Emit a vision marker for EVERY image_url in the conversation.
        std::string chat_text;
        for (const auto& msg : messages) {
            std::string role = msg.at("role").get<std::string>();
            std::string content;
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    std::string text_parts;
                    std::string vision_markers;
                    for (const auto& part : msg["content"]) {
                        std::string pt = part.value("type", "");
                        if (pt == "text") {
                            if (!text_parts.empty()) text_parts += "\n";
                            text_parts += part.at("text").get<std::string>();
                        } else if (pt == "image_url") {
                            vision_markers += "<|vision_start|><|vision_end|>";
                        }
                    }
                    content = vision_markers + text_parts;
                } else if (msg["content"].is_null()) {
                    content = "";
                }
            }
            chat_text += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        chat_text += "<|im_start|>assistant\n";
        if (cfg.enable_thinking) {
            chat_text += "<think>\n";
        } else {
            chat_text += "<think>\n\n</think>\n\n";
        }
        req.prompt = chat_text;
    }

    // Log the constructed prompt (truncated) if logging enabled
    if (cfg.enable_logging) {
        // Count vision markers in prompt
        size_t vm_count = 0;
        {
            const std::string marker = "<|vision_start|>";
            size_t p = 0;
            while ((p = req.prompt.find(marker, p)) != std::string::npos) {
                ++vm_count;
                p += marker.size();
            }
        }
        if (vm_count > 0 || !req.images.empty()) {
            std::cerr << "[ov_serve] VL: " << req.images.size() << " images, "
                      << vm_count << " vision markers, prompt_len="
                      << req.prompt.size() << " chars\n";
        }
        std::string dbg = req.prompt;
        // Replace base64 image data with placeholder for readability
        auto pos = dbg.find("data:image");
        if (pos != std::string::npos) {
            auto end = dbg.find("\"", pos);
            if (end != std::string::npos && end - pos > 100) {
                dbg.replace(pos + 30, end - pos - 30, "...<base64_truncated>...");
            }
        }
        if (dbg.size() > 8000) dbg = dbg.substr(0, 4000) + "\n...(truncated " + std::to_string(dbg.size()) + " chars)...\n" + dbg.substr(dbg.size() - 4000);
        std::cerr << "[ov_serve] PROMPT: " << dbg << "\n";
    }

    // Generation params — accept both "max_tokens" (legacy) and "max_completion_tokens" (OpenAI v2)
    if (body.contains("max_completion_tokens")) {
        req.params.max_new_tokens = body["max_completion_tokens"].get<int>();
    } else {
        req.params.max_new_tokens = body.value("max_tokens", cfg.max_tokens_default);
    }
    req.params.enable_thinking = cfg.enable_thinking;
    req.params.raw_prompt = true;  // prompt is already ChatML-formatted

    float temperature = body.value("temperature", cfg.temperature);
    if (cfg.min_temperature > 0.0f && temperature < cfg.min_temperature) {
        temperature = cfg.min_temperature;
    }
    req.params.sampling.temperature = temperature;
    req.params.sampling.top_p = body.value("top_p", cfg.top_p);
    if (body.contains("top_k")) {
        req.params.sampling.top_k = body["top_k"].get<size_t>();
    } else {
        req.params.sampling.top_k = cfg.top_k;
    }
    if (body.contains("seed")) {
        req.params.sampling.rng_seed = body["seed"].get<size_t>();
    }
    if (body.contains("repetition_penalty")) {
        req.params.sampling.repetition_penalty = body["repetition_penalty"].get<float>();
    } else {
        req.params.sampling.repetition_penalty = cfg.repetition_penalty;
    }
    if (body.contains("frequency_penalty")) {
        req.params.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
    } else {
        req.params.sampling.frequency_penalty = cfg.frequency_penalty;
    }
    if (body.contains("presence_penalty")) {
        req.params.sampling.presence_penalty = body["presence_penalty"].get<float>();
    } else {
        req.params.sampling.presence_penalty = cfg.presence_penalty;
    }

    // Stop strings
    if (body.contains("stop")) {
        if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                req.params.stop_strings.push_back(s.get<std::string>());
            }
        } else if (body["stop"].is_string()) {
            req.params.stop_strings.push_back(body["stop"].get<std::string>());
        }
    }

    // Tools
    if (body.contains("tools")) {
        req.tools = body["tools"].get<std::vector<json>>();
    }

    return req;
}

// ═══════════════════════════════════════════════════════════════════
//  Build OpenAI response JSON
// ═══════════════════════════════════════════════════════════════════

static json build_chat_response(const std::string& id, const std::string& model,
                                const GenerateResult& result,
                                const std::vector<ToolCall>& tool_calls) {
    json choice;
    choice["index"] = 0;

    json message;
    message["role"] = "assistant";
    message["content"] = result.text;

    if (!result.thinking_text.empty()) {
        message["reasoning_content"] = result.thinking_text;
    }

    if (!tool_calls.empty()) {
        json tc_array = json::array();
        for (const auto& tc : tool_calls) {
            json tc_obj;
            tc_obj["id"] = tc.id;
            tc_obj["type"] = "function";
            json func;
            func["name"] = tc.name;
            func["arguments"] = tc.arguments;
            tc_obj["function"] = func;
            tc_array.push_back(tc_obj);
        }
        message["tool_calls"] = tc_array;
        choice["finish_reason"] = "tool_calls";
    } else {
        choice["finish_reason"] = (result.stop_reason == StopReason::MAX_TOKENS)
                                      ? "length"
                                      : "stop";
    }
    message["content"] = result.text;
    choice["message"] = message;

    json usage;
    usage["prompt_tokens"] = result.prompt_tokens;
    usage["completion_tokens"] = result.generated_tokens;
    usage["total_tokens"] = result.prompt_tokens + result.generated_tokens;

    json perf;
    perf["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
    perf["prefill_ms"] = std::round(result.prefill_ms * 100.0) / 100.0;
    perf["decode_ms"] = std::round(result.decode_ms * 100.0) / 100.0;
    perf["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
    if (result.prefix_cached_tokens > 0)
        perf["prefix_cached_tokens"] = result.prefix_cached_tokens;
    usage["performance"] = perf;

    json resp;
    resp["id"] = id;
    resp["object"] = "chat.completion";
    resp["created"] = unix_timestamp();
    resp["model"] = model;
    resp["choices"] = json::array({choice});
    resp["usage"] = usage;
    return resp;
}

static json build_canned_chat_response(const std::string& id, const std::string& model,
                                       const std::string& text) {
    json choice;
    choice["index"] = 0;
    choice["finish_reason"] = "stop";
    json message;
    message["role"] = "assistant";
    message["content"] = text;
    choice["message"] = message;

    json usage;
    usage["prompt_tokens"] = 0;
    usage["completion_tokens"] = 0;
    usage["total_tokens"] = 0;

    json resp;
    resp["id"] = id;
    resp["object"] = "chat.completion";
    resp["created"] = unix_timestamp();
    resp["model"] = model;
    resp["choices"] = json::array({choice});
    resp["usage"] = usage;
    return resp;
}

// ═══════════════════════════════════════════════════════════════════
//  SSE streaming helpers
// ═══════════════════════════════════════════════════════════════════

static std::string sse_chunk(const json& data) {
    return "data: " + data.dump() + "\n\n";
}

static std::string sse_done() {
    return "data: [DONE]\n\n";
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

static void print_usage() {
    std::cerr << "Usage: ov_serve --model <path> [--port 8080] [--host 0.0.0.0] "
                 "[--workers 1] [--device GPU] [--vl] [--no-thinking] [--temperature 0.7] "
                 "[--top-p 0.95] [--top-k 0] [--rep-penalty 1.0]\n"
                 "\n"
                 "  --vl              Enable vision-language (load vision encoder)\n"
                 "  --no-thinking     Disable thinking mode\n"
                 "  --temperature     Default temperature (default: 0.7)\n"
                 "  --top-p           Default top-p (default: 0.95)\n"
                 "  --top-k           Default top-k (default: 0, disabled)\n"
                 "  --rep-penalty     Repetition penalty (default: 1.0, no penalty)\n"
                 "  --pres-penalty    Presence penalty (default: 0.0)\n"
                 "  --freq-penalty    Frequency penalty (default: 0.0)\n"
                 "  --min-temp        Minimum temperature floor (default: 0, no override)\n"
                 "  --warmup-tokens   Max sequence length for GPU warmup (default: 0, disabled)\n"
                 "  --no-log          Disable request/prompt logging to stderr\n";
                 "  --dflash-dir DIR  DFlash draft model dir (enables speculative decoding)\n";
}

int main(int argc, char* argv[]) {
    ServerConfig cfg;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) cfg.model_path = argv[++i];
        else if (arg == "--port" && i + 1 < argc) cfg.port = std::stoi(argv[++i]);
        else if (arg == "--host" && i + 1 < argc) cfg.host = argv[++i];
        else if (arg == "--workers" && i + 1 < argc) cfg.workers = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) cfg.device = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) cfg.max_tokens_default = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) cfg.temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) cfg.top_p = std::stof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) cfg.top_k = static_cast<size_t>(std::stoul(argv[++i]));
        else if (arg == "--rep-penalty" && i + 1 < argc) cfg.repetition_penalty = std::stof(argv[++i]);
        else if (arg == "--pres-penalty" && i + 1 < argc) cfg.presence_penalty = std::stof(argv[++i]);
        else if (arg == "--freq-penalty" && i + 1 < argc) cfg.frequency_penalty = std::stof(argv[++i]);
        else if (arg == "--min-temp" && i + 1 < argc) cfg.min_temperature = std::stof(argv[++i]);
        else if (arg == "--warmup-tokens" && i + 1 < argc) cfg.warmup_tokens = std::stoi(argv[++i]);
        else if (arg == "--no-thinking") cfg.enable_thinking = false;
        else if (arg == "--vl") cfg.enable_vision = true;
        else if (arg == "--no-log") cfg.enable_logging = false;
        else if (arg == "--dflash-dir" && i + 1 < argc) cfg.dflash_model_path = argv[++i];
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage();
        return 1;
    }

    // ── Load model ──
    std::cerr << "[ov_serve] Loading model from " << cfg.model_path << " ...\n";
    auto t0 = std::chrono::steady_clock::now();

    bool dflash_enabled = !cfg.dflash_model_path.empty();

    LoadParams lp;
    lp.device = cfg.device;
    lp.cache_ir = true;
    lp.enable_vision = cfg.enable_vision;
    lp.skip_text_compile = dflash_enabled;  // DFlash has its own target model

    ModelLoader loader(cfg.model_path, lp);

    auto t1 = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[ov_serve] Model loaded in " << load_sec << "s\n";

    // ── Create worker pool (skip when DFlash handles all text) ──
    std::unique_ptr<WorkerPool> pool_ptr;
    if (!dflash_enabled) {
        std::cerr << "[ov_serve] Creating " << cfg.workers << " worker session(s)...\n";
        pool_ptr = std::make_unique<WorkerPool>(loader, cfg.workers);
        std::cerr << "[ov_serve] Workers ready\n";

        // ── GPU warmup ──
        if (cfg.warmup_tokens > 0) {
            std::cerr << "[ov_serve] Warming up " << pool_ptr->size()
                      << " session(s) with max_seq_len=" << cfg.warmup_tokens << "...\n";
            auto tw0 = std::chrono::steady_clock::now();
            pool_ptr->warmup(cfg.warmup_tokens);
            auto tw1 = std::chrono::steady_clock::now();
            double warmup_sec = std::chrono::duration<double>(tw1 - tw0).count();
            std::cerr << "[ov_serve] Warmup complete in " << warmup_sec << "s\n";
        }
    }

    auto* tokenizer = loader.tokenizer();
    std::string model_name = "qwen3.5";

    // ── DFlash engine (optional speculative decoding) ──
    std::unique_ptr<dflash::DFlashEngine> dflash_engine;
    if (dflash_enabled) {
        std::cerr << "[ov_serve] Loading DFlash engine (draft: " << cfg.dflash_model_path << ")...\n";
        auto td0 = std::chrono::steady_clock::now();
        dflash::DFlashConfig dcfg;
        dcfg.target_model_dir = cfg.model_path;
        dcfg.draft_model_dir = cfg.dflash_model_path;
        dcfg.device = cfg.device;
        dflash_engine = std::make_unique<dflash::DFlashEngine>(dcfg);
        dflash_engine->set_stop_token_ids(loader.stop_token_ids());
        auto td1 = std::chrono::steady_clock::now();
        double dflash_sec = std::chrono::duration<double>(td1 - td0).count();
        std::cerr << "[ov_serve] DFlash engine loaded in " << dflash_sec << "s\n";
        model_name = "qwen3.5-dflash";
    }
    // Mutex for DFlash engine (single-threaded speculative decode)
    std::mutex dflash_mu;

    // ── HTTP server ──
    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Model list
    svr.Get("/v1/models", [&model_name, &loader](const httplib::Request&, httplib::Response& res) {
        json resp;
        resp["object"] = "list";
        json model;
        model["id"] = model_name;
        model["object"] = "model";
        model["owned_by"] = "openvino";
        model["max_context_length"] = loader.config().text.max_position_embeddings;
        resp["data"] = json::array({model});
        res.set_content(resp.dump(), "application/json");
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
             [&pool_ptr, &cfg, &model_name, tokenizer,
              &dflash_engine, dflash_enabled, &dflash_mu, &loader](const httplib::Request& req,
                                                    httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"]["message"] = std::string("Invalid JSON: ") + e.what();
            err["error"]["type"] = "invalid_request_error";
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            json err;
            err["error"]["message"] = "messages field is required and must be an array";
            err["error"]["type"] = "invalid_request_error";
            res.set_content(err.dump(), "application/json");
            return;
        }

        ParsedRequest parsed;
        try {
            parsed = parse_chat_request(body, *tokenizer, cfg);
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"]["message"] = std::string("Request parse error: ") + e.what();
            err["error"]["type"] = "invalid_request_error";
            res.set_content(err.dump(), "application/json");
            return;
        }

        // VL validation
        if (!parsed.images.empty()) {
            if (!cfg.enable_vision) {
                res.status = 400;
                json err;
                err["error"]["message"] = "Image input requires --vl flag. "
                    "Restart server with: ov_serve --model <path> --vl";
                err["error"]["type"] = "invalid_request_error";
                res.set_content(err.dump(), "application/json");
                return;
            }
            if (!parsed.tools.empty()) {
                // Ignore tool definitions for image requests
                parsed.tools.clear();
            }
        }

        auto request_id = make_request_id();

        if (cfg.enable_logging) {
            std::cerr << "[ov_serve] " << request_id
                      << " temp=" << parsed.params.sampling.temperature
                      << " top_p=" << parsed.params.sampling.top_p
                      << " top_k=" << parsed.params.sampling.top_k
                      << " rep=" << parsed.params.sampling.repetition_penalty
                      << " pres=" << parsed.params.sampling.presence_penalty
                      << " freq=" << parsed.params.sampling.frequency_penalty
                      << " max_tokens=" << parsed.params.max_new_tokens
                      << " stream=" << parsed.stream << "\n";
        }

        // ── DFlash speculative decoding path (text + VL) ──
        if (dflash_enabled) {
            // VL preprocessing: run vision encoder and build VL inputs
            std::shared_ptr<dflash::DFlashVLInputs> vl_inputs;
            if (!parsed.images.empty()) {
                try {
                    using namespace ov::genai::modeling;

                    auto* compiled_vision = loader.compiled_vision();
                    if (!compiled_vision) {
                        res.status = 400;
                        json err;
                        err["error"]["message"] = "Vision model not available";
                        err["error"]["type"] = "invalid_request_error";
                        res.set_content(err.dump(), "application/json");
                        return;
                    }

                    const auto& model_cfg = loader.config();
                    models::Qwen3_5VisionPreprocessor preprocessor(
                        model_cfg.vision, loader.preprocess_config());

                    // Run vision encoder for each image
                    auto vision_req = compiled_vision->create_infer_request();
                    std::vector<ov::Tensor> per_image_embeds;
                    std::vector<std::array<int64_t, 3>> per_image_grid;
                    size_t total_vis_tokens = 0;

                    for (const auto& image : parsed.images) {
                        auto inputs = preprocessor.preprocess(image, loader.pos_embed_weight());
                        vision_req.set_tensor(models::Qwen3_5VisionIO::kPixelValues, inputs.pixel_values);
                        vision_req.set_tensor(models::Qwen3_5VisionIO::kGridThw, inputs.grid_thw);
                        vision_req.set_tensor(models::Qwen3_5VisionIO::kPosEmbeds, inputs.pos_embeds);
                        vision_req.set_tensor(models::Qwen3_5VisionIO::kRotaryCos, inputs.rotary_cos);
                        vision_req.set_tensor(models::Qwen3_5VisionIO::kRotarySin, inputs.rotary_sin);
                        vision_req.infer();

                        ov::Tensor ref = vision_req.get_tensor(models::Qwen3_5VisionIO::kVisualEmbeds);
                        ov::Tensor embeds(ref.get_element_type(), ref.get_shape());
                        std::memcpy(embeds.data(), ref.data(), ref.get_byte_size());

                        const auto* g = inputs.grid_thw.data<const int64_t>();
                        per_image_grid.push_back({g[0], g[1], g[2]});
                        total_vis_tokens += embeds.get_shape().at(0);
                        per_image_embeds.push_back(std::move(embeds));
                    }

                    // Build combined grid_thw [N, 3]
                    ov::Tensor grid_thw(ov::element::i64, {parsed.images.size(), 3});
                    auto* gdata = grid_thw.data<int64_t>();
                    for (size_t i = 0; i < per_image_grid.size(); ++i) {
                        gdata[i * 3 + 0] = per_image_grid[i][0];
                        gdata[i * 3 + 1] = per_image_grid[i][1];
                        gdata[i * 3 + 2] = per_image_grid[i][2];
                    }

                    // Concatenate visual embeds if multiple images
                    ov::Tensor visual_embeds;
                    if (per_image_embeds.size() == 1) {
                        visual_embeds = per_image_embeds[0];
                    } else {
                        const auto hidden = per_image_embeds[0].get_shape().at(1);
                        const auto elem_type = per_image_embeds[0].get_element_type();
                        visual_embeds = ov::Tensor(elem_type, {total_vis_tokens, hidden});
                        char* dst = static_cast<char*>(visual_embeds.data());
                        for (const auto& e : per_image_embeds) {
                            const size_t nb = e.get_byte_size();
                            std::memcpy(dst, e.data(), nb);
                            dst += nb;
                        }
                    }

                    // Tokenize VL prompt (expand vision markers with pad tokens)
                    const int32_t merge = model_cfg.vision.spatial_merge_size;
                    const int64_t total_tokens = models::Qwen3_5VisionPreprocessor::count_visual_tokens(
                        grid_thw, merge);

                    // Expand all <|vision_start|><|vision_end|> markers
                    std::string vl_prompt = parsed.prompt;
                    const std::string marker = "<|vision_start|><|vision_end|>";
                    size_t pos = 0, img_idx = 0;
                    while ((pos = vl_prompt.find(marker, pos)) != std::string::npos
                           && img_idx < per_image_grid.size()) {
                        const int64_t h = per_image_grid[img_idx][1];
                        const int64_t w = per_image_grid[img_idx][2];
                        const int64_t pad_count = per_image_grid[img_idx][0] * (h / merge) * (w / merge);
                        std::string expansion = "<|vision_start|>";
                        for (int64_t i = 0; i < pad_count; ++i)
                            expansion += "<|image_pad|>";
                        expansion += "<|vision_end|>";
                        vl_prompt.replace(pos, marker.size(), expansion);
                        pos += expansion.size();
                        ++img_idx;
                    }

                    auto encoded = tokenizer->encode(vl_prompt, ov::genai::add_special_tokens(false));

                    // Build VL plan (mRoPE position IDs + visual position mask)
                    models::Qwen3_5InputPlanner planner(model_cfg);
                    auto plan = planner.build_plan(encoded.input_ids, &encoded.attention_mask, &grid_thw);

                    // Scatter visual embeddings into padded sequence-length tensor
                    auto visual_padded = models::Qwen3_5InputPlanner::scatter_visual_embeds(
                        visual_embeds, plan.visual_pos_mask);

                    vl_inputs = std::make_shared<dflash::DFlashVLInputs>();
                    vl_inputs->input_ids = encoded.input_ids;
                    vl_inputs->attention_mask = encoded.attention_mask;
                    vl_inputs->position_ids = plan.position_ids;
                    vl_inputs->visual_embeds = visual_padded;
                    vl_inputs->visual_pos_mask = plan.visual_pos_mask;

                    if (cfg.enable_logging) {
                        std::cerr << "[ov_serve] " << request_id << " [dflash-vl] "
                                  << parsed.images.size() << " images, "
                                  << total_tokens << " vision tokens, "
                                  << encoded.input_ids.get_shape()[1] << " prompt tokens\n";
                    }
                } catch (const std::exception& e) {
                    res.status = 500;
                    json err;
                    err["error"]["message"] = std::string("Vision preprocessing error: ") + e.what();
                    err["error"]["type"] = "server_error";
                    res.set_content(err.dump(), "application/json");
                    return;
                }
            }

            auto sp = std::move(parsed);
            auto rid = request_id;
            auto mn = model_name;
            auto log = cfg.enable_logging;
            auto max_tok = sp.params.max_new_tokens;
            auto enable_thinking = cfg.enable_thinking;

            if (sp.stream) {
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&dflash_engine, &dflash_mu, sp = std::move(sp), rid, mn, log, max_tok,
                     vl_inputs, enable_thinking](
                        size_t, httplib::DataSink& sink) {
                        std::lock_guard<std::mutex> lock(dflash_mu);

                        // ThinkingTracker to strip <think>...</think> from output
                        ov::genai::modeling::ThinkingTracker think_tracker;
                        if (!enable_thinking) {
                            // When thinking is disabled, prompt has <think>\n\n</think>\n\n
                            // but model may still emit these tokens — tracker will strip them.
                            // Start in BEFORE_THINKING so it catches any <think> tags.
                        }

                        // Send initial role chunk
                        {
                            json rc;
                            rc["id"] = rid;
                            rc["object"] = "chat.completion.chunk";
                            rc["created"] = unix_timestamp();
                            rc["model"] = mn;
                            json d; d["role"] = "assistant";
                            json c; c["index"] = 0; c["delta"] = d;
                            rc["choices"] = json::array({c});
                            std::string s = sse_chunk(rc);
                            sink.write(s.c_str(), s.size());
                        }

                        ToolCallParser tool_parser;
                        std::vector<ToolCall> accumulated_tool_calls;

                        try {
                            auto result = dflash_engine->generate(
                                sp.prompt, max_tok,
                                [&](const std::string& text, bool is_eos) -> bool {
                                    if (!sink.is_writable()) return false;
                                    // Filter through ThinkingTracker to strip <think>...</think>
                                    auto tr = think_tracker.process(text);
                                    const std::string& content = tr.content_text;
                                    if (content.empty()) return true;
                                    auto pr = tool_parser.process(content);
                                    for (auto& tc : pr.tool_calls)
                                        accumulated_tool_calls.push_back(std::move(tc));
                                    if (!pr.text.empty()) {
                                        json tc;
                                        tc["id"] = rid;
                                        tc["object"] = "chat.completion.chunk";
                                        tc["created"] = unix_timestamp();
                                        tc["model"] = mn;
                                        json d; d["content"] = pr.text;
                                        json c; c["index"] = 0; c["delta"] = d;
                                        tc["choices"] = json::array({c});
                                        std::string s = sse_chunk(tc);
                                        if (!sink.write(s.c_str(), s.size())) return false;
                                    }
                                    return true;
                                },
                                vl_inputs.get());

                            // Flush tool parser
                            auto final_pr = tool_parser.flush();
                            for (auto& tc : final_pr.tool_calls)
                                accumulated_tool_calls.push_back(std::move(tc));
                            if (!final_pr.text.empty()) {
                                json tc;
                                tc["id"] = rid;
                                tc["object"] = "chat.completion.chunk";
                                tc["created"] = unix_timestamp();
                                tc["model"] = mn;
                                json d; d["content"] = final_pr.text;
                                json c; c["index"] = 0; c["delta"] = d;
                                tc["choices"] = json::array({c});
                                std::string s = sse_chunk(tc);
                                sink.write(s.c_str(), s.size());
                            }

                            // Send tool calls if any
                            if (!accumulated_tool_calls.empty()) {
                                json tc;
                                tc["id"] = rid;
                                tc["object"] = "chat.completion.chunk";
                                tc["created"] = unix_timestamp();
                                tc["model"] = mn;
                                json d;
                                json tool_calls_json = json::array();
                                for (size_t i = 0; i < accumulated_tool_calls.size(); ++i) {
                                    json tcj;
                                    tcj["index"] = static_cast<int>(i);
                                    tcj["id"] = "call_" + std::to_string(i);
                                    tcj["type"] = "function";
                                    json fn;
                                    fn["name"] = accumulated_tool_calls[i].name;
                                    fn["arguments"] = accumulated_tool_calls[i].arguments;
                                    tcj["function"] = fn;
                                    tool_calls_json.push_back(tcj);
                                }
                                d["tool_calls"] = tool_calls_json;
                                json c;
                                c["index"] = 0;
                                c["delta"] = d;
                                tc["choices"] = json::array({c});
                                std::string s = sse_chunk(tc);
                                sink.write(s.c_str(), s.size());
                            }

                            // Final chunk with finish_reason
                            {
                                std::string fr = accumulated_tool_calls.empty()
                                    ? result.finish_reason : "tool_calls";
                                json fc;
                                fc["id"] = rid;
                                fc["object"] = "chat.completion.chunk";
                                fc["created"] = unix_timestamp();
                                fc["model"] = mn;
                                json c;
                                c["index"] = 0;
                                c["delta"] = json::object();
                                c["finish_reason"] = fr;
                                fc["choices"] = json::array({c});
                                // Usage info
                                json usage;
                                usage["prompt_tokens"] = static_cast<int>(result.prompt_tokens);
                                usage["completion_tokens"] = static_cast<int>(result.generated_tokens);
                                usage["total_tokens"] = static_cast<int>(result.prompt_tokens + result.generated_tokens);
                                json perf;
                                perf["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
                                perf["decode_ms"] = std::round(result.decode_ms * 100.0) / 100.0;
                                perf["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
                                if (result.prefix_cached_tokens > 0)
                                    perf["prefix_cached_tokens"] = result.prefix_cached_tokens;
                                usage["performance"] = perf;
                                fc["usage"] = usage;
                                std::string s = sse_chunk(fc);
                                sink.write(s.c_str(), s.size());
                            }

                            std::string done = sse_done();
                            sink.write(done.c_str(), done.size());

                            if (log) {
                                double accept_rate = result.generated_tokens > 0
                                    ? static_cast<double>(result.accepted_tokens) / static_cast<double>(result.generated_tokens) : 0.0;
                                double avg_accept = result.draft_steps > 0
                                    ? static_cast<double>(result.accepted_tokens) / static_cast<double>(result.draft_steps) : 0.0;
                                std::cerr << std::fixed << std::setprecision(1)
                                          << "[ov_serve] " << rid << " [dflash] "
                                          << result.prompt_tokens << "p+"
                                          << result.generated_tokens << "g "
                                          << result.throughput << " t/s, ttft=" << result.ttft_ms << "ms"
                                          << (result.prefix_cached_tokens > 0
                                              ? ", cache=" + std::to_string(result.prefix_cached_tokens) : "")
                                          << " | "
                                          << result.draft_steps << " steps, "
                                          << std::setprecision(1) << (accept_rate * 100) << "% accept, "
                                          << std::setprecision(1) << avg_accept << " avg/step\n";
                            }
                        } catch (const std::exception& e) {
                            json err_chunk;
                            err_chunk["error"] = e.what();
                            std::string s = sse_chunk(err_chunk);
                            sink.write(s.c_str(), s.size());
                            std::string done = sse_done();
                            sink.write(done.c_str(), done.size());
                            std::cerr << "[ov_serve] " << rid << " [dflash] error: " << e.what() << "\n";
                        }
                        sink.done();
                        return true;
                    });
            } else {
                // Non-streaming DFlash
                res.set_chunked_content_provider(
                    "application/json",
                    [&dflash_engine, &dflash_mu, sp = std::move(sp), rid, mn, log, max_tok,
                     vl_inputs](
                        size_t, httplib::DataSink& sink) {
                        std::lock_guard<std::mutex> lock(dflash_mu);

                        ToolCallParser tool_parser;
                        std::vector<ToolCall> accumulated_tool_calls;
                        std::string full_text;

                        try {
                            auto result = dflash_engine->generate(
                                sp.prompt, max_tok,
                                [&](const std::string& text, bool) -> bool {
                                    if (!sink.is_writable()) return false;
                                    auto pr = tool_parser.process(text);
                                    for (auto& tc : pr.tool_calls)
                                        accumulated_tool_calls.push_back(std::move(tc));
                                    full_text += pr.text;
                                    return true;
                                },
                                vl_inputs.get());

                            auto final_pr = tool_parser.flush();
                            for (auto& tc : final_pr.tool_calls)
                                accumulated_tool_calls.push_back(std::move(tc));
                            full_text += final_pr.text;

                            std::string fr = accumulated_tool_calls.empty()
                                ? result.finish_reason : "tool_calls";

                            json resp;
                            resp["id"] = rid;
                            resp["object"] = "chat.completion";
                            resp["created"] = unix_timestamp();
                            resp["model"] = mn;
                            json msg;
                            msg["role"] = "assistant";
                            if (!accumulated_tool_calls.empty()) {
                                json tc_arr = json::array();
                                for (size_t i = 0; i < accumulated_tool_calls.size(); ++i) {
                                    json tcj;
                                    tcj["id"] = "call_" + std::to_string(i);
                                    tcj["type"] = "function";
                                    json fn;
                                    fn["name"] = accumulated_tool_calls[i].name;
                                    fn["arguments"] = accumulated_tool_calls[i].arguments;
                                    tcj["function"] = fn;
                                    tc_arr.push_back(tcj);
                                }
                                msg["tool_calls"] = tc_arr;
                                msg["content"] = nullptr;
                            } else {
                                msg["content"] = full_text;
                            }
                            json choice;
                            choice["index"] = 0;
                            choice["message"] = msg;
                            choice["finish_reason"] = fr;
                            resp["choices"] = json::array({choice});
                            json usage;
                            usage["prompt_tokens"] = static_cast<int>(result.prompt_tokens);
                            usage["completion_tokens"] = static_cast<int>(result.generated_tokens);
                            usage["total_tokens"] = static_cast<int>(result.prompt_tokens + result.generated_tokens);
                            json perf;
                            perf["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
                            perf["decode_ms"] = std::round(result.decode_ms * 100.0) / 100.0;
                            perf["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
                            if (result.prefix_cached_tokens > 0)
                                perf["prefix_cached_tokens"] = result.prefix_cached_tokens;
                            usage["performance"] = perf;
                            resp["usage"] = usage;
                            std::string body = resp.dump();
                            sink.write(body.c_str(), body.size());

                            if (log) {
                                double accept_rate = result.generated_tokens > 0
                                    ? static_cast<double>(result.accepted_tokens) / static_cast<double>(result.generated_tokens) : 0.0;
                                double avg_accept = result.draft_steps > 0
                                    ? static_cast<double>(result.accepted_tokens) / static_cast<double>(result.draft_steps) : 0.0;
                                std::cerr << std::fixed << std::setprecision(1)
                                          << "[ov_serve] " << rid << " [dflash] "
                                          << result.prompt_tokens << "p+"
                                          << result.generated_tokens << "g "
                                          << result.throughput << " t/s, ttft=" << result.ttft_ms << "ms"
                                          << (result.prefix_cached_tokens > 0
                                              ? ", cache=" + std::to_string(result.prefix_cached_tokens) : "")
                                          << " | "
                                          << result.draft_steps << " steps, "
                                          << std::setprecision(1) << (accept_rate * 100) << "% accept, "
                                          << std::setprecision(1) << avg_accept << " avg/step\n";
                            }
                        } catch (const std::exception& e) {
                            json err;
                            err["error"]["message"] = std::string("DFlash generation error: ") + e.what();
                            err["error"]["type"] = "server_error";
                            std::string body = err.dump();
                            sink.write(body.c_str(), body.size());
                            std::cerr << "[ov_serve] " << rid << " [dflash] error: " << e.what() << "\n";
                        }
                        sink.done();
                        return true;
                    });
            }
            return;
        }

        try {
            if (parsed.stream) {
                // ── Streaming SSE ──
                auto sp = std::move(parsed);
                auto rid = request_id;
                auto mn = model_name;
                auto warmup_tok = cfg.warmup_tokens;
                auto log = cfg.enable_logging;

                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&pool_ptr, sp = std::move(sp), rid, mn, warmup_tok, log](size_t /*offset*/,
                                                         httplib::DataSink& sink) {
                        // Acquire session inside content provider (lives until lambda ends)
                        WorkerPool::Guard worker(*pool_ptr);
                        auto* session = worker.get();

                        // Send initial role chunk
                        {
                            json role_chunk;
                            role_chunk["id"] = rid;
                            role_chunk["object"] = "chat.completion.chunk";
                            role_chunk["created"] = unix_timestamp();
                            role_chunk["model"] = mn;
                            json delta;
                            delta["role"] = "assistant";
                            json choice;
                            choice["index"] = 0;
                            choice["delta"] = delta;
                            role_chunk["choices"] = json::array({choice});
                            std::string role_sse = sse_chunk(role_chunk);
                            sink.write(role_sse.c_str(), role_sse.size());
                        }

                        // Session already processes through ThinkingTracker internally;
                        // sc.is_thinking tells us which category each token belongs to.
                        ToolCallParser tool_parser;
                        std::vector<ToolCall> accumulated_tool_calls;

                        auto callback = [&](const StreamChunk& sc) -> bool {
                            if (sc.event == StreamEvent::TOKEN && !sc.token_text.empty()) {
                                if (log)
                                    std::cerr << "[DEBUG-STREAM] TOKEN thinking=" << sc.is_thinking
                                              << " text='" << sc.token_text.substr(0, 30) << "'\n";
                                if (sc.is_thinking) {
                                    // Send as reasoning_content delta
                                    json tc;
                                    tc["id"] = rid;
                                    tc["object"] = "chat.completion.chunk";
                                    tc["created"] = unix_timestamp();
                                    tc["model"] = mn;
                                    json d;
                                    d["reasoning_content"] = sc.token_text;
                                    json c;
                                    c["index"] = 0;
                                    c["delta"] = d;
                                    tc["choices"] = json::array({c});
                                    std::string s = sse_chunk(tc);
                                    if (!sink.write(s.c_str(), s.size())) return false;
                                } else {
                                    // Content text — run through tool parser
                                    auto pr = tool_parser.process(sc.token_text);
                                    // Accumulate any completed tool calls
                                    for (auto& tc_item : pr.tool_calls) {
                                        accumulated_tool_calls.push_back(std::move(tc_item));
                                    }
                                    if (!pr.text.empty()) {
                                        json tc;
                                        tc["id"] = rid;
                                        tc["object"] = "chat.completion.chunk";
                                        tc["created"] = unix_timestamp();
                                        tc["model"] = mn;
                                        json d;
                                        d["content"] = pr.text;
                                        json c;
                                        c["index"] = 0;
                                        c["delta"] = d;
                                        tc["choices"] = json::array({c});
                                        std::string s = sse_chunk(tc);
                                        if (!sink.write(s.c_str(), s.size())) return false;
                                    }
                                }
                            } else if (sc.event == StreamEvent::FINISH) {
                                auto flush_result = tool_parser.flush();

                                // Emit any remaining text from flush
                                if (!flush_result.text.empty()) {
                                    json tc2;
                                    tc2["id"] = rid;
                                    tc2["object"] = "chat.completion.chunk";
                                    tc2["created"] = unix_timestamp();
                                    tc2["model"] = mn;
                                    json d2;
                                    d2["content"] = flush_result.text;
                                    json c2;
                                    c2["index"] = 0;
                                    c2["delta"] = d2;
                                    tc2["choices"] = json::array({c2});
                                    std::string s2 = sse_chunk(tc2);
                                    sink.write(s2.c_str(), s2.size());
                                }

                                // Merge any tool calls from flush into accumulated
                                for (auto& tc_item : flush_result.tool_calls) {
                                    accumulated_tool_calls.push_back(std::move(tc_item));
                                }

                                // Emit tool_call deltas in SSE
                                for (size_t ti = 0; ti < accumulated_tool_calls.size(); ++ti) {
                                    const auto& atc = accumulated_tool_calls[ti];
                                    json tc_chunk;
                                    tc_chunk["id"] = rid;
                                    tc_chunk["object"] = "chat.completion.chunk";
                                    tc_chunk["created"] = unix_timestamp();
                                    tc_chunk["model"] = mn;
                                    json d_tc;
                                    json fn;
                                    fn["name"] = atc.name;
                                    fn["arguments"] = atc.arguments;
                                    json tc_obj;
                                    tc_obj["index"] = static_cast<int>(ti);
                                    tc_obj["id"] = atc.id;
                                    tc_obj["type"] = "function";
                                    tc_obj["function"] = fn;
                                    d_tc["tool_calls"] = json::array({tc_obj});
                                    json c_tc;
                                    c_tc["index"] = 0;
                                    c_tc["delta"] = d_tc;
                                    tc_chunk["choices"] = json::array({c_tc});
                                    std::string s_tc = sse_chunk(tc_chunk);
                                    sink.write(s_tc.c_str(), s_tc.size());
                                }

                                std::string finish_reason;
                                if (!accumulated_tool_calls.empty()) {
                                    finish_reason = "tool_calls";
                                } else if (sc.stop_reason == StopReason::MAX_TOKENS) {
                                    finish_reason = "length";
                                } else {
                                    finish_reason = "stop";
                                }

                                json fc;
                                fc["id"] = rid;
                                fc["object"] = "chat.completion.chunk";
                                fc["created"] = unix_timestamp();
                                fc["model"] = mn;
                                json d = json::object();  // empty delta
                                json c;
                                c["index"] = 0;
                                c["delta"] = d;
                                c["finish_reason"] = finish_reason;
                                fc["choices"] = json::array({c});

                                json usage;
                                usage["prompt_tokens"] = sc.prompt_tokens;
                                usage["completion_tokens"] = sc.generated_tokens;
                                usage["total_tokens"] = sc.prompt_tokens + sc.generated_tokens;

                                json perf;
                                perf["ttft_ms"] = std::round(sc.ttft_ms * 100.0) / 100.0;
                                perf["prefill_ms"] = std::round(sc.prefill_ms * 100.0) / 100.0;
                                perf["decode_ms"] = std::round(sc.decode_ms * 100.0) / 100.0;
                                perf["throughput_tps"] = std::round(sc.throughput * 100.0) / 100.0;
                                if (sc.prefix_cached_tokens > 0)
                                    perf["prefix_cached_tokens"] = sc.prefix_cached_tokens;
                                usage["performance"] = perf;
                                fc["usage"] = usage;

                                std::string s = sse_chunk(fc);
                                sink.write(s.c_str(), s.size());

                                std::string done = sse_done();
                                sink.write(done.c_str(), done.size());

                                if (log) {
                                    std::cerr << "[ov_serve] " << rid
                                              << " done: " << sc.generated_tokens << " tokens"
                                              << ", ttft=" << std::round(sc.ttft_ms * 10.0) / 10.0 << "ms"
                                              << ", throughput=" << std::round(sc.throughput * 10.0) / 10.0 << " t/s"
                                              << ", prefill=" << std::round(sc.prefill_ms * 10.0) / 10.0 << "ms"
                                              << ", decode=" << std::round(sc.decode_ms * 10.0) / 10.0 << "ms"
                                              << (sc.prefix_cached_tokens > 0
                                                  ? ", cache_hit=" + std::to_string(sc.prefix_cached_tokens) + " tokens"
                                                  : "")
                                              << "\n";
                                }
                            }
                            return true;  // continue generating
                        };

                        try {
                            if (!sp.images.empty()) {
                                session->generate_vl(sp.prompt, sp.images,
                                                     sp.params, callback);
                            } else {
                                session->generate(sp.prompt, sp.params, callback);
                            }
                        } catch (const std::exception& e) {
                            if (is_gpu_oom(e)) {
                                std::cerr << "[ov_serve] streaming GPU OOM, recreating session...\n";
                                try {
                                    pool_ptr->recreate_session(session, warmup_tok);
                                } catch (...) {}
                            }
                            // Send error as SSE event before closing
                            json err_chunk;
                            err_chunk["error"] = e.what();
                            std::string s = sse_chunk(err_chunk);
                            sink.write(s.c_str(), s.size());
                            std::string done = sse_done();
                            sink.write(done.c_str(), done.size());
                        }
                        sink.done();
                        return true;
                    });
                // Handler returns here; content provider runs asynchronously
            } else {
                // ── Non-streaming ──
                // Use chunked content provider so we get a DataSink to detect
                // client disconnects during generation.
                auto sp = std::move(parsed);
                auto rid = request_id;
                auto mn = model_name;
                auto warmup_tok = cfg.warmup_tokens;
                auto log = cfg.enable_logging;

                res.set_chunked_content_provider(
                    "application/json",
                    [&pool_ptr, sp = std::move(sp), rid, mn, warmup_tok, log](
                        size_t, httplib::DataSink& sink) {
                        WorkerPool::Guard worker(*pool_ptr);
                        auto* session = worker.get();

                        // Lightweight callback: only checks connection liveness
                        auto callback = [&](const StreamChunk& sc) -> bool {
                            if (sc.event == StreamEvent::TOKEN) {
                                return sink.is_writable();
                            }
                            return true;
                        };

                        GenerateResult result;
                        auto do_generate = [&]() {
                            if (!sp.images.empty()) {
                                return session->generate_vl(sp.prompt, sp.images,
                                                            sp.params, callback);
                            } else {
                                return session->generate(sp.prompt, sp.params, callback);
                            }
                        };

                        try {
                            try {
                                result = do_generate();
                            } catch (const std::exception& e) {
                                if (is_gpu_oom(e)) {
                                    std::cerr << "[ov_serve] " << rid
                                              << " GPU OOM detected, recreating session...\n";
                                    pool_ptr->recreate_session(session, warmup_tok);
                                    result = do_generate();
                                } else {
                                    throw;
                                }
                            }

                            // Skip response if client disconnected during generation
                            if (!sink.is_writable()) {
                                if (log) {
                                    std::cerr << "[ov_serve] " << rid
                                              << " client disconnected, dropping "
                                              << result.generated_tokens << " tokens\n";
                                }
                                sink.done();
                                return true;
                            }

                            ToolCallParser tool_parser;
                            auto pr = tool_parser.process(result.text);
                            result.text = pr.text;

                            auto resp = build_chat_response(rid, mn, result, pr.tool_calls);
                            std::string body = resp.dump();
                            sink.write(body.c_str(), body.size());

                            if (log) {
                                std::cerr << "[ov_serve] " << rid
                                          << " done: " << result.generated_tokens << " tokens"
                                          << ", ttft=" << std::round(result.ttft_ms * 10.0) / 10.0 << "ms"
                                          << ", throughput=" << std::round(result.throughput * 10.0) / 10.0 << " t/s"
                                          << ", prefill=" << std::round(result.prefill_ms * 10.0) / 10.0 << "ms"
                                          << ", decode=" << std::round(result.decode_ms * 10.0) / 10.0 << "ms"
                                          << (result.prefix_cached_tokens > 0
                                              ? ", cache_hit=" + std::to_string(result.prefix_cached_tokens) + " tokens"
                                              : "")
                                          << "\n";
                            }
                        } catch (const std::exception& e) {
                            if (!sink.is_writable()) { sink.done(); return true; }
                            json err;
                            err["error"]["message"] = std::string("Generation error: ") + e.what();
                            err["error"]["type"] = "server_error";
                            std::string body = err.dump();
                            sink.write(body.c_str(), body.size());
                        }
                        sink.done();
                        return true;
                    });
            }
        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]["message"] = std::string("Generation error: ") + e.what();
            err["error"]["type"] = "server_error";
            res.set_content(err.dump(), "application/json");
        }
    });

    // Text completions
    svr.Post("/v1/completions",
             [&pool_ptr, &cfg, &model_name](const httplib::Request& req, httplib::Response& res) {
        if (!pool_ptr) {
            res.status = 503;
            json err;
            err["error"]["message"] = "Text completions endpoint is unavailable in DFlash mode";
            err["error"]["type"] = "service_unavailable";
            res.set_content(err.dump(), "application/json");
            return;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"]["message"] = std::string("Invalid JSON: ") + e.what();
            err["error"]["type"] = "invalid_request_error";
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string prompt = body.value("prompt", "");
        if (prompt.empty()) {
            res.status = 400;
            json err;
            err["error"]["message"] = "prompt is required";
            err["error"]["type"] = "invalid_request_error";
            res.set_content(err.dump(), "application/json");
            return;
        }

        GenerateParams params;
        if (body.contains("max_completion_tokens")) {
            params.max_new_tokens = body["max_completion_tokens"].get<int>();
        } else {
            params.max_new_tokens = body.value("max_tokens", cfg.max_tokens_default);
        }
        params.sampling.temperature = body.value("temperature", cfg.temperature);
        if (cfg.min_temperature > 0.0f && params.sampling.temperature < cfg.min_temperature) {
            params.sampling.temperature = cfg.min_temperature;
        }
        params.sampling.top_p = body.value("top_p", cfg.top_p);
        if (body.contains("top_k")) {
            params.sampling.top_k = body["top_k"].get<size_t>();
        } else {
            params.sampling.top_k = cfg.top_k;
        }
        if (body.contains("repetition_penalty")) {
            params.sampling.repetition_penalty = body["repetition_penalty"].get<float>();
        } else {
            params.sampling.repetition_penalty = cfg.repetition_penalty;
        }
        if (body.contains("frequency_penalty")) {
            params.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
        } else {
            params.sampling.frequency_penalty = cfg.frequency_penalty;
        }
        if (body.contains("presence_penalty")) {
            params.sampling.presence_penalty = body["presence_penalty"].get<float>();
        } else {
            params.sampling.presence_penalty = cfg.presence_penalty;
        }
        params.enable_thinking = false;  // No thinking in raw completions

        auto request_id = make_request_id();
        auto prompt_copy = prompt;
        auto params_copy = params;
        auto rid = request_id;
        auto mn = model_name;
        auto warmup_tok = cfg.warmup_tokens;

        res.set_chunked_content_provider(
            "application/json",
            [&pool_ptr, prompt_copy = std::move(prompt_copy), params_copy, rid, mn, warmup_tok](
                size_t, httplib::DataSink& sink) {
                WorkerPool::Guard worker(*pool_ptr);
                auto* session = worker.get();

                auto callback = [&](const StreamChunk& sc) -> bool {
                    if (sc.event == StreamEvent::TOKEN) {
                        return sink.is_writable();
                    }
                    return true;
                };

                try {
                    auto result = session->generate(prompt_copy, params_copy, callback);

                    if (!sink.is_writable()) { sink.done(); return true; }

                    json resp;
                    resp["id"] = rid;
                    resp["object"] = "text_completion";
                    resp["created"] = unix_timestamp();
                    resp["model"] = mn;

                    json choice;
                    choice["index"] = 0;
                    choice["text"] = result.text;
                    choice["finish_reason"] = (result.stop_reason == StopReason::MAX_TOKENS)
                                                  ? "length"
                                                  : "stop";
                    resp["choices"] = json::array({choice});

                    json usage;
                    usage["prompt_tokens"] = result.prompt_tokens;
                    usage["completion_tokens"] = result.generated_tokens;
                    usage["total_tokens"] = result.prompt_tokens + result.generated_tokens;

                    json perf;
                    perf["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
                    perf["prefill_ms"] = std::round(result.prefill_ms * 100.0) / 100.0;
                    perf["decode_ms"] = std::round(result.decode_ms * 100.0) / 100.0;
                    perf["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
                    if (result.prefix_cached_tokens > 0)
                        perf["prefix_cached_tokens"] = result.prefix_cached_tokens;
                    usage["performance"] = perf;
                    resp["usage"] = usage;

                    std::string body = resp.dump();
                    sink.write(body.c_str(), body.size());
                } catch (const std::exception& e) {
                    if (!sink.is_writable()) { sink.done(); return true; }
                    json err;
                    err["error"]["message"] = std::string("Generation error: ") + e.what();
                    err["error"]["type"] = "server_error";
                    std::string body = err.dump();
                    sink.write(body.c_str(), body.size());
                }
                sink.done();
                return true;
            });
        // RAII Guard handles session release automatically
    });

    // ═══════════════════════════════════════════════════════════════════
    //  Ollama-compatible API endpoints
    // ═══════════════════════════════════════════════════════════════════

    // Helper: ISO 8601 timestamp
    auto iso_timestamp = []() -> std::string {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
        return std::string(buf);
    };

    // Helper: parse Ollama "options" object into GenerateParams
    auto parse_ollama_options = [&cfg](const json& body, GenerateParams& params) {
        json opts = body.value("options", json::object());

        float temperature = opts.value("temperature", cfg.temperature);
        if (cfg.min_temperature > 0.0f && temperature < cfg.min_temperature) {
            temperature = cfg.min_temperature;
        }
        params.sampling.temperature = temperature;
        params.sampling.top_p = opts.value("top_p", cfg.top_p);
        if (opts.contains("top_k")) {
            params.sampling.top_k = opts["top_k"].get<size_t>();
        } else {
            params.sampling.top_k = cfg.top_k;
        }
        if (opts.contains("seed")) {
            params.sampling.rng_seed = opts["seed"].get<size_t>();
        }
        if (opts.contains("repeat_penalty")) {
            params.sampling.repetition_penalty = opts["repeat_penalty"].get<float>();
        } else {
            params.sampling.repetition_penalty = cfg.repetition_penalty;
        }
        if (opts.contains("frequency_penalty")) {
            params.sampling.frequency_penalty = opts["frequency_penalty"].get<float>();
        } else {
            params.sampling.frequency_penalty = cfg.frequency_penalty;
        }
        if (opts.contains("presence_penalty")) {
            params.sampling.presence_penalty = opts["presence_penalty"].get<float>();
        } else {
            params.sampling.presence_penalty = cfg.presence_penalty;
        }

        params.max_new_tokens = opts.value("num_predict", cfg.max_tokens_default);
        if (body.contains("num_predict")) {
            params.max_new_tokens = body["num_predict"].get<int>();
        }

        if (body.contains("keep_alive")) {
            // Ollama keep_alive: ignored (we always keep model loaded)
        }
    };

    // GET /api/tags — List models (Ollama format)
    svr.Get("/api/tags", [&model_name, &iso_timestamp](const httplib::Request&,
                                                        httplib::Response& res) {
        json model;
        model["name"] = model_name + ":latest";
        model["model"] = model_name + ":latest";
        model["modified_at"] = iso_timestamp();
        model["size"] = 0;
        model["digest"] = "openvino";
        json details;
        details["parent_model"] = "";
        details["format"] = "openvino";
        details["family"] = "qwen3.5";
        details["families"] = json::array({"qwen3.5"});
        details["parameter_size"] = "35B";
        details["quantization_level"] = "INT4";
        model["details"] = details;

        json resp;
        resp["models"] = json::array({model});
        res.set_content(resp.dump(), "application/json");
    });

    // POST /api/show — Show model info (Ollama format)
    svr.Post("/api/show", [&model_name, &iso_timestamp, &cfg, &loader](const httplib::Request&,
                                                              httplib::Response& res) {
        json resp;
        resp["modelfile"] = "# OpenVINO GenAI model";
        resp["parameters"] = "temperature " + std::to_string(cfg.temperature) +
                             "\ntop_p " + std::to_string(cfg.top_p) +
                             "\ntop_k " + std::to_string(cfg.top_k);
        resp["template"] = "ChatML";
        json details;
        details["parent_model"] = "";
        details["format"] = "openvino";
        details["family"] = "qwen3.5";
        details["families"] = json::array({"qwen3.5"});
        details["parameter_size"] = "35B";
        details["quantization_level"] = "INT4";
        resp["details"] = details;
        json model_info;
        model_info["max_context_length"] = loader.config().text.max_position_embeddings;
        resp["model_info"] = model_info;
        resp["modified_at"] = iso_timestamp();
        res.set_content(resp.dump(), "application/json");
    });

    // POST /api/chat — Ollama chat (NDJSON streaming)
    svr.Post("/api/chat",
             [&pool_ptr, &cfg, &model_name, &iso_timestamp, &parse_ollama_options, tokenizer](
                 const httplib::Request& req, httplib::Response& res) {
        if (!pool_ptr) {
            res.status = 503;
            json err;
            err["error"] = "Ollama chat endpoint is unavailable in DFlash mode";
            res.set_content(err.dump(), "application/json");
            return;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            json err;
            err["error"] = "Invalid JSON";
            res.set_content(err.dump(), "application/json");
            return;
        }

        bool do_stream = body.value("stream", true);  // Ollama defaults to stream=true

        auto messages = body.value("messages", json::array());

        // Prepend thinking tags to historical assistant messages (same as OpenAI path)
        {
            const std::string think_prefix = "<think>\n\n</think>\n\n";
            for (size_t i = 0; i < messages.size(); ++i) {
                if (messages[i].value("role", "") == "assistant"
                    && i + 1 < messages.size()) {
                    std::string c;
                    if (messages[i].contains("content") && messages[i]["content"].is_string())
                        c = messages[i]["content"].get<std::string>();
                    if (c.find("<think>") == std::string::npos) {
                        messages[i]["content"] = think_prefix + c;
                    }
                }
            }
        }

        std::string chat_text;
        if (tokenizer && !tokenizer->get_chat_template().empty()) {
            ov::genai::ChatHistory history(ov::genai::JsonContainer::from_json_string(messages.dump()));
            ov::genai::JsonContainer extra({{"enable_thinking", cfg.enable_thinking}});
            chat_text = tokenizer->apply_chat_template(history, true, {}, std::nullopt, extra);
        } else {
            // Fallback path if tokenizer chat template is unavailable.
            for (const auto& msg : messages) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                chat_text += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }

            chat_text += "<|im_start|>assistant\n";
            if (cfg.enable_thinking) {
                chat_text += "<think>\n";
            } else {
                chat_text += "<think>\n\n</think>\n\n";
            }
        }

        GenerateParams params;
        params.raw_prompt = true;
        params.enable_thinking = cfg.enable_thinking;
        parse_ollama_options(body, params);

        std::string mn = model_name;

        if (cfg.enable_logging) {
            std::cerr << "[ov_serve] ollama-chat"
                      << " temp=" << params.sampling.temperature
                      << " top_k=" << params.sampling.top_k
                      << " rep=" << params.sampling.repetition_penalty
                      << " pres=" << params.sampling.presence_penalty
                      << " max_tokens=" << params.max_new_tokens
                      << " stream=" << do_stream << "\n";
        }

        if (do_stream) {
            // NDJSON streaming (Ollama format: one JSON per line)
            auto prompt = std::move(chat_text);
            res.set_chunked_content_provider(
                "application/x-ndjson",
                [&pool_ptr, params, prompt, mn, &cfg, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(*pool_ptr);
                    auto* session = worker.get();

                    auto callback = [&](const StreamChunk& sc) -> bool {
                        if (sc.event == StreamEvent::TOKEN && !sc.token_text.empty()) {
                            if (!sc.is_thinking) {
                                json chunk;
                                chunk["model"] = mn;
                                chunk["created_at"] = iso_timestamp();
                                json msg;
                                msg["role"] = "assistant";
                                msg["content"] = sc.token_text;
                                chunk["message"] = msg;
                                chunk["done"] = false;
                                std::string line = chunk.dump() + "\n";
                                if (!sink.write(line.c_str(), line.size())) return false;
                            }
                        } else if (sc.event == StreamEvent::FINISH) {
                            json done_chunk;
                            done_chunk["model"] = mn;
                            done_chunk["created_at"] = iso_timestamp();
                            json msg;
                            msg["role"] = "assistant";
                            msg["content"] = "";
                            done_chunk["message"] = msg;
                            done_chunk["done"] = true;
                            done_chunk["done_reason"] =
                                (sc.stop_reason == StopReason::MAX_TOKENS)
                                    ? "length" : "stop";
                            done_chunk["eval_count"] =
                                static_cast<int>(sc.generated_tokens);
                            done_chunk["prompt_eval_count"] =
                                static_cast<int>(sc.prompt_tokens);
                            done_chunk["eval_duration"] =
                                static_cast<int64_t>(sc.decode_ms * 1e6);
                            done_chunk["prompt_eval_duration"] =
                                static_cast<int64_t>(sc.prefill_ms * 1e6);
                            done_chunk["ttft_ms"] = std::round(sc.ttft_ms * 100.0) / 100.0;
                            done_chunk["throughput_tps"] = std::round(sc.throughput * 100.0) / 100.0;
                            std::string line = done_chunk.dump() + "\n";
                            sink.write(line.c_str(), line.size());
                        }
                        return true;
                    };

                    try {
                        session->generate(prompt, params, callback);
                    } catch (const std::exception& e) {
                        json err;
                        err["error"] = e.what();
                        std::string line = err.dump() + "\n";
                        sink.write(line.c_str(), line.size());
                    }
                    sink.done();
                    return true;
                });
        } else {
            // Non-streaming with disconnect detection
            auto prompt_copy = chat_text;
            auto params_copy = params;

            res.set_chunked_content_provider(
                "application/json",
                [&pool_ptr, prompt_copy = std::move(prompt_copy), params_copy, mn, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(*pool_ptr);
                    auto* session = worker.get();

                    auto callback = [&](const StreamChunk& sc) -> bool {
                        if (sc.event == StreamEvent::TOKEN) {
                            return sink.is_writable();
                        }
                        return true;
                    };

                    try {
                        auto result = session->generate(prompt_copy, params_copy, callback);

                        if (!sink.is_writable()) { sink.done(); return true; }

                        json resp;
                        resp["model"] = mn;
                        resp["created_at"] = iso_timestamp();
                        json msg;
                        msg["role"] = "assistant";
                        msg["content"] = result.text;
                        resp["message"] = msg;
                        resp["done"] = true;
                        resp["done_reason"] = (result.stop_reason == StopReason::MAX_TOKENS)
                                                  ? "length" : "stop";
                        resp["eval_count"] = static_cast<int>(result.generated_tokens);
                        resp["prompt_eval_count"] = static_cast<int>(result.prompt_tokens);
                        resp["eval_duration"] = static_cast<int64_t>(result.decode_ms * 1e6);
                        resp["prompt_eval_duration"] = static_cast<int64_t>(result.prefill_ms * 1e6);
                        resp["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
                        resp["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
                        std::string body = resp.dump();
                        sink.write(body.c_str(), body.size());
                    } catch (const std::exception& e) {
                        if (!sink.is_writable()) { sink.done(); return true; }
                        json err;
                        err["error"] = std::string("Generation error: ") + e.what();
                        std::string body = err.dump();
                        sink.write(body.c_str(), body.size());
                    }
                    sink.done();
                    return true;
                });
        }
    });

    // POST /api/generate — Ollama raw generate (NDJSON streaming)
    svr.Post("/api/generate",
             [&pool_ptr, &cfg, &model_name, &iso_timestamp, &parse_ollama_options, tokenizer](
                 const httplib::Request& req, httplib::Response& res) {
        if (!pool_ptr) {
            res.status = 503;
            json err;
            err["error"] = "Ollama generate endpoint is unavailable in DFlash mode";
            res.set_content(err.dump(), "application/json");
            return;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            json err;
            err["error"] = "Invalid JSON";
            res.set_content(err.dump(), "application/json");
            return;
        }

        bool do_stream = body.value("stream", true);
        std::string prompt = body.value("prompt", "");
        bool raw = body.value("raw", false);

        // If not raw mode, wrap in chat template / ChatML
        if (!raw) {
            std::string system = body.value("system", "");
            if (tokenizer && !tokenizer->get_chat_template().empty()) {
                ov::genai::ChatHistory history;
                if (!system.empty()) {
                    history.push_back({{"role", "system"}, {"content", system}});
                }
                history.push_back({{"role", "user"}, {"content", prompt}});
                ov::genai::JsonContainer extra({{"enable_thinking", cfg.enable_thinking}});
                prompt = tokenizer->apply_chat_template(history, true, {}, std::nullopt, extra);
            } else {
                std::string wrapped;
                if (!system.empty()) {
                    wrapped = "<|im_start|>system\n" + system + "<|im_end|>\n";
                }
                wrapped += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
                wrapped += "<|im_start|>assistant\n";
                if (cfg.enable_thinking) {
                    wrapped += "<think>\n";
                } else {
                    wrapped += "<think>\n\n</think>\n\n";
                }
                prompt = wrapped;
            }
        }

        GenerateParams params;
        params.raw_prompt = true;
        params.enable_thinking = cfg.enable_thinking;
        parse_ollama_options(body, params);

        std::string mn = model_name;

        if (do_stream) {
            res.set_chunked_content_provider(
                "application/x-ndjson",
                [&pool_ptr, params, prompt, mn, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(*pool_ptr);
                    auto* session = worker.get();

                    auto callback = [&](const StreamChunk& sc) -> bool {
                        if (sc.event == StreamEvent::TOKEN && !sc.token_text.empty()) {
                            if (!sc.is_thinking) {
                                json chunk;
                                chunk["model"] = mn;
                                chunk["created_at"] = iso_timestamp();
                                chunk["response"] = sc.token_text;
                                chunk["done"] = false;
                                std::string line = chunk.dump() + "\n";
                                if (!sink.write(line.c_str(), line.size())) return false;
                            }
                        } else if (sc.event == StreamEvent::FINISH) {
                            json done_chunk;
                            done_chunk["model"] = mn;
                            done_chunk["created_at"] = iso_timestamp();
                            done_chunk["response"] = "";
                            done_chunk["done"] = true;
                            done_chunk["done_reason"] =
                                (sc.stop_reason == StopReason::MAX_TOKENS)
                                    ? "length" : "stop";
                            done_chunk["eval_count"] =
                                static_cast<int>(sc.generated_tokens);
                            done_chunk["prompt_eval_count"] =
                                static_cast<int>(sc.prompt_tokens);
                            done_chunk["eval_duration"] =
                                static_cast<int64_t>(sc.decode_ms * 1e6);
                            done_chunk["prompt_eval_duration"] =
                                static_cast<int64_t>(sc.prefill_ms * 1e6);
                            done_chunk["ttft_ms"] = std::round(sc.ttft_ms * 100.0) / 100.0;
                            done_chunk["throughput_tps"] = std::round(sc.throughput * 100.0) / 100.0;
                            std::string line = done_chunk.dump() + "\n";
                            sink.write(line.c_str(), line.size());
                        }
                        return true;
                    };

                    try {
                        session->generate(prompt, params, callback);
                    } catch (const std::exception& e) {
                        json err;
                        err["error"] = e.what();
                        std::string line = err.dump() + "\n";
                        sink.write(line.c_str(), line.size());
                    }
                    sink.done();
                    return true;
                });
        } else {
            auto prompt_copy = prompt;
            auto params_copy = params;

            res.set_chunked_content_provider(
                "application/json",
                [&pool_ptr, prompt_copy = std::move(prompt_copy), params_copy, mn, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(*pool_ptr);
                    auto* session = worker.get();

                    auto callback = [&](const StreamChunk& sc) -> bool {
                        if (sc.event == StreamEvent::TOKEN) {
                            return sink.is_writable();
                        }
                        return true;
                    };

                    try {
                        auto result = session->generate(prompt_copy, params_copy, callback);

                        if (!sink.is_writable()) { sink.done(); return true; }

                        json resp;
                        resp["model"] = mn;
                        resp["created_at"] = iso_timestamp();
                        resp["response"] = result.text;
                        resp["done"] = true;
                        resp["done_reason"] = (result.stop_reason == StopReason::MAX_TOKENS)
                                                  ? "length" : "stop";
                        resp["eval_count"] = static_cast<int>(result.generated_tokens);
                        resp["prompt_eval_count"] = static_cast<int>(result.prompt_tokens);
                        resp["eval_duration"] = static_cast<int64_t>(result.decode_ms * 1e6);
                        resp["prompt_eval_duration"] = static_cast<int64_t>(result.prefill_ms * 1e6);
                        resp["ttft_ms"] = std::round(result.ttft_ms * 100.0) / 100.0;
                        resp["throughput_tps"] = std::round(result.throughput * 100.0) / 100.0;
                        std::string body = resp.dump();
                        sink.write(body.c_str(), body.size());
                    } catch (const std::exception& e) {
                        if (!sink.is_writable()) { sink.done(); return true; }
                        json err;
                        err["error"] = std::string("Generation error: ") + e.what();
                        std::string body = err.dump();
                        sink.write(body.c_str(), body.size());
                    }
                    sink.done();
                    return true;
                });
        }
    });

    // ── Start server ──
    std::cerr << "[ov_serve] Starting server on " << cfg.host << ":" << cfg.port << "\n";
    std::cerr << "[ov_serve] Workers: " << cfg.workers
              << ", Device: " << cfg.device
              << ", Thinking: " << (cfg.enable_thinking ? "on" : "off")
              << ", Vision: " << (cfg.enable_vision ? "on" : "off")
              << ", Rep.Penalty: " << cfg.repetition_penalty
              << ", Pres.Penalty: " << cfg.presence_penalty
              << ", Freq.Penalty: " << cfg.frequency_penalty
              << ", Logging: " << (cfg.enable_logging ? "on" : "off") << "\n";
    std::cerr << "[ov_serve] Endpoints:\n"
              << "  POST /v1/chat/completions  (OpenAI)\n"
              << "  POST /v1/completions       (OpenAI)\n"
              << "  GET  /v1/models            (OpenAI)\n"
              << "  POST /api/chat             (Ollama)\n"
              << "  POST /api/generate         (Ollama)\n"
              << "  GET  /api/tags             (Ollama)\n"
              << "  POST /api/show             (Ollama)\n"
              << "  GET  /health\n";

    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "[ov_serve] Failed to start server on " << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }

    return 0;
}
