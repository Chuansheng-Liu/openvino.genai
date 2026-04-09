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

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
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
    std::string device = "GPU";
    int port = 8080;
    std::string host = "0.0.0.0";
    int workers = 1;
    bool enable_thinking = true;
    bool enable_vision = false;
    int max_tokens_default = 2048;
    float repetition_penalty = 1.1f;  // Default to prevent degeneration
    float presence_penalty = 1.5f;    // Qwen3.5 official recommendation
    float min_temperature = 0.0f;     // 0 = no override; set via --min-temp
    int warmup_tokens = 4096;         // 0 = disable warmup
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
        s->reset();
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
    }

    // Build chat prompt using Qwen3.5 ChatML format
    auto messages = body.at("messages");
    std::string chat_text;

    // If tools are present, inject tool schema into system message (Qwen3.5 format)
    bool has_system = false;
    for (const auto& msg : messages) {
        std::string role = msg.at("role").get<std::string>();
        std::string content;

        // Handle both string content and multimodal content array
        if (msg.contains("content")) {
            if (msg["content"].is_string()) {
                content = msg["content"].get<std::string>();
            } else if (msg["content"].is_array()) {
                // OpenAI multimodal content array: [{type: "text"}, {type: "image_url"}]
                for (const auto& part : msg["content"]) {
                    std::string part_type = part.value("type", "");
                    if (part_type == "text") {
                        if (!content.empty()) content += "\n";
                        content += part.at("text").get<std::string>();
                    } else if (part_type == "image_url") {
                        auto url = part.at("image_url").at("url").get<std::string>();
                        auto image = decode_image_from_data_uri(url);
                        req.images.push_back(std::move(image));
                        // Insert vision marker into prompt at image position
                        content += "<|vision_start|><|vision_end|>";
                    }
                }
            } else if (msg["content"].is_null()) {
                content = "";
            }
        }

        if (role == "system" && !req.tools.empty() && !has_system) {
            // Inject tool definitions into system message
            content += "\n\n# Tools\n\nYou may call one or more functions to assist "
                       "with the user query.\n\nYou are provided with function signatures "
                       "within <tools></tools> XML tags:\n<tools>\n";
            for (const auto& tool : req.tools) {
                if (tool.contains("function")) {
                    content += tool["function"].dump() + "\n";
                }
            }
            content += "</tools>\n\nFor each function call, return a json object with "
                       "function name and arguments within <tool_call></tool_call> XML tags:\n"
                       "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                       "</tool_call>";
            has_system = true;
        } else if (role == "system") {
            has_system = true;
        } else if (role == "tool") {
            // Tool response messages
            std::string tool_call_id;
            if (msg.contains("tool_call_id")) {
                tool_call_id = msg["tool_call_id"].get<std::string>();
            }
            content = "[Tool Response: " + tool_call_id + "]\n" + content;
            role = "user";  // Map tool role to user for Qwen3.5
        }

        chat_text += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }

    // If tools but no system message, prepend one
    if (!req.tools.empty() && !has_system) {
        std::string sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                          "# Tools\n\nYou may call one or more functions.\n<tools>\n";
        for (const auto& tool : req.tools) {
            if (tool.contains("function")) {
                sys += tool["function"].dump() + "\n";
            }
        }
        sys += "</tools>\n\nFor each function call, return a json object with "
               "function name and arguments within <tool_call></tool_call> XML tags:\n"
               "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
               "</tool_call>";
        chat_text = "<|im_start|>system\n" + sys + "<|im_end|>\n" + chat_text;
    }

    chat_text += "<|im_start|>assistant\n";
    if (cfg.enable_thinking) {
        // Trigger structured thinking with <think> tag
        chat_text += "<think>\n";
    } else {
        // Suppress thinking: empty think block tells model to skip thinking
        chat_text += "<think>\n</think>\n\n";
    }
    req.prompt = chat_text;

    // Generation params
    req.params.max_new_tokens = body.value("max_tokens", cfg.max_tokens_default);
    req.params.enable_thinking = cfg.enable_thinking;
    req.params.raw_prompt = true;  // prompt is already ChatML-formatted

    float temperature = body.value("temperature", 0.7f);
    if (cfg.min_temperature > 0.0f && temperature < cfg.min_temperature) {
        temperature = cfg.min_temperature;
    }
    req.params.sampling.temperature = temperature;
    req.params.sampling.top_p = body.value("top_p", 0.95f);
    if (body.contains("top_k")) {
        req.params.sampling.top_k = body["top_k"].get<size_t>();
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
                 "[--workers 1] [--device GPU] [--vl] [--no-thinking] [--rep-penalty 1.1]\n"
                 "\n"
                 "  --vl              Enable vision-language (load vision encoder)\n"
                 "  --no-thinking     Disable thinking mode\n"
                 "  --rep-penalty     Repetition penalty (default: 1.1)\n"
                 "  --min-temp        Minimum temperature floor (default: 0, no override)\n"
                 "  --warmup-tokens   Max sequence length for GPU warmup (default: 4096, 0=disable)\n";
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
        else if (arg == "--rep-penalty" && i + 1 < argc) cfg.repetition_penalty = std::stof(argv[++i]);
        else if (arg == "--min-temp" && i + 1 < argc) cfg.min_temperature = std::stof(argv[++i]);
        else if (arg == "--warmup-tokens" && i + 1 < argc) cfg.warmup_tokens = std::stoi(argv[++i]);
        else if (arg == "--no-thinking") cfg.enable_thinking = false;
        else if (arg == "--vl") cfg.enable_vision = true;
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

    LoadParams lp;
    lp.device = cfg.device;
    lp.cache_ir = true;
    lp.enable_vision = cfg.enable_vision;

    ModelLoader loader(cfg.model_path, lp);

    auto t1 = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[ov_serve] Model loaded in " << load_sec << "s\n";

    // ── Create worker pool ──
    std::cerr << "[ov_serve] Creating " << cfg.workers << " worker session(s)...\n";
    WorkerPool pool(loader, cfg.workers);
    std::cerr << "[ov_serve] Workers ready\n";

    // ── GPU warmup ──
    if (cfg.warmup_tokens > 0) {
        std::cerr << "[ov_serve] Warming up " << pool.size()
                  << " session(s) with max_seq_len=" << cfg.warmup_tokens << "...\n";
        auto tw0 = std::chrono::steady_clock::now();
        pool.warmup(cfg.warmup_tokens);
        auto tw1 = std::chrono::steady_clock::now();
        double warmup_sec = std::chrono::duration<double>(tw1 - tw0).count();
        std::cerr << "[ov_serve] Warmup complete in " << warmup_sec << "s\n";
    }

    auto* tokenizer = loader.tokenizer();
    std::string model_name = "qwen3.5";

    // ── HTTP server ──
    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Model list
    svr.Get("/v1/models", [&model_name](const httplib::Request&, httplib::Response& res) {
        json resp;
        resp["object"] = "list";
        json model;
        model["id"] = model_name;
        model["object"] = "model";
        model["owned_by"] = "openvino";
        resp["data"] = json::array({model});
        res.set_content(resp.dump(), "application/json");
    });

    // Chat completions
    svr.Post("/v1/chat/completions",
             [&pool, &cfg, &model_name, tokenizer](const httplib::Request& req,
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
            if (parsed.images.size() > 1) {
                res.status = 400;
                json err;
                err["error"]["message"] = "Only one image per request is currently supported";
                err["error"]["type"] = "invalid_request_error";
                res.set_content(err.dump(), "application/json");
                return;
            }
            if (!parsed.tools.empty()) {
                res.status = 400;
                json err;
                err["error"]["message"] = "Tool calling with images is not supported";
                err["error"]["type"] = "invalid_request_error";
                res.set_content(err.dump(), "application/json");
                return;
            }
        }

        auto request_id = make_request_id();

        std::cerr << "[ov_serve] " << request_id
                  << " temp=" << parsed.params.sampling.temperature
                  << " top_p=" << parsed.params.sampling.top_p
                  << " top_k=" << parsed.params.sampling.top_k
                  << " rep=" << parsed.params.sampling.repetition_penalty
                  << " pres=" << parsed.params.sampling.presence_penalty
                  << " freq=" << parsed.params.sampling.frequency_penalty
                  << " max_tokens=" << parsed.params.max_new_tokens
                  << " stream=" << parsed.stream << "\n";

        try {
            if (parsed.stream) {
                // ── Streaming SSE ──
                // IMPORTANT: For streaming, session lifecycle must live inside the
                // content provider lambda. set_chunked_content_provider returns
                // immediately; the lambda runs after this handler exits.
                // Capturing local variables by reference would be use-after-free.
                auto sp = std::move(parsed);  // move into value captures
                auto rid = request_id;
                auto mn = model_name;
                auto warmup_tok = cfg.warmup_tokens;

                res.set_chunked_content_provider(
                    "text/event-stream",
                    [&pool, sp = std::move(sp), rid, mn, warmup_tok](size_t /*offset*/,
                                                         httplib::DataSink& sink) {
                        // Acquire session inside content provider (lives until lambda ends)
                        WorkerPool::Guard worker(pool);
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

                        auto callback = [&](const StreamChunk& sc) -> bool {
                            if (sc.event == StreamEvent::TOKEN && !sc.token_text.empty()) {
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
                                    sink.write(s.c_str(), s.size());
                                } else {
                                    // Content text — run through tool parser
                                    auto pr = tool_parser.process(sc.token_text);
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
                                        sink.write(s.c_str(), s.size());
                                    }
                                }
                            } else if (sc.event == StreamEvent::FINISH) {
                                auto flush_result = tool_parser.flush();
                                std::string finish_reason =
                                    (sc.stop_reason == StopReason::MAX_TOKENS)
                                        ? "length"
                                        : "stop";

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
                                fc["usage"] = usage;

                                std::string s = sse_chunk(fc);
                                sink.write(s.c_str(), s.size());

                                std::string done = sse_done();
                                sink.write(done.c_str(), done.size());
                            }
                            return true;  // continue generating
                        };

                        try {
                            if (!sp.images.empty()) {
                                session->generate_vl(sp.prompt, sp.images[0],
                                                     sp.params, callback);
                            } else {
                                session->generate(sp.prompt, sp.params, callback);
                            }
                        } catch (const std::exception& e) {
                            if (is_gpu_oom(e)) {
                                std::cerr << "[ov_serve] streaming GPU OOM, recreating session...\n";
                                try {
                                    pool.recreate_session(session, warmup_tok);
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
                WorkerPool::Guard worker(pool);
                auto* session = worker.get();

                GenerateResult result;
                auto do_generate = [&]() {
                    if (!parsed.images.empty()) {
                        return session->generate_vl(parsed.prompt, parsed.images[0],
                                                    parsed.params);
                    } else {
                        return session->generate(parsed.prompt, parsed.params);
                    }
                };

                try {
                    result = do_generate();
                } catch (const std::exception& e) {
                    if (is_gpu_oom(e)) {
                        std::cerr << "[ov_serve] " << request_id
                                  << " GPU OOM detected, recreating session...\n";
                        pool.recreate_session(session, cfg.warmup_tokens);
                        result = do_generate();  // retry once
                    } else {
                        throw;
                    }
                }

                // Parse tool calls from output
                ToolCallParser tool_parser;
                auto pr = tool_parser.process(result.text);
                result.text = pr.text;  // Remove tool call XML from text

                auto resp = build_chat_response(request_id, model_name, result, pr.tool_calls);
                res.set_content(resp.dump(), "application/json");
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
             [&pool, &cfg, &model_name](const httplib::Request& req, httplib::Response& res) {
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
        params.max_new_tokens = body.value("max_tokens", cfg.max_tokens_default);
        params.sampling.temperature = body.value("temperature", 0.7f);
        if (cfg.min_temperature > 0.0f && params.sampling.temperature < cfg.min_temperature) {
            params.sampling.temperature = cfg.min_temperature;
        }
        params.sampling.top_p = body.value("top_p", 0.95f);
        if (body.contains("top_k")) {
            params.sampling.top_k = body["top_k"].get<size_t>();
        }
        if (body.contains("repetition_penalty")) {
            params.sampling.repetition_penalty = body["repetition_penalty"].get<float>();
        } else {
            params.sampling.repetition_penalty = cfg.repetition_penalty;
        }
        if (body.contains("frequency_penalty")) {
            params.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
        }
        if (body.contains("presence_penalty")) {
            params.sampling.presence_penalty = body["presence_penalty"].get<float>();
        } else {
            params.sampling.presence_penalty = cfg.presence_penalty;
        }
        params.enable_thinking = false;  // No thinking in raw completions

        auto request_id = make_request_id();
        WorkerPool::Guard worker(pool);
        auto* session = worker.get();

        try {
            auto result = session->generate(prompt, params);

            json resp;
            resp["id"] = request_id;
            resp["object"] = "text_completion";
            resp["created"] = unix_timestamp();
            resp["model"] = model_name;

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
            resp["usage"] = usage;

            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]["message"] = std::string("Generation error: ") + e.what();
            err["error"]["type"] = "server_error";
            res.set_content(err.dump(), "application/json");
        }
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

        float temperature = opts.value("temperature", 0.7f);
        if (cfg.min_temperature > 0.0f && temperature < cfg.min_temperature) {
            temperature = cfg.min_temperature;
        }
        params.sampling.temperature = temperature;
        params.sampling.top_p = opts.value("top_p", 0.95f);
        if (opts.contains("top_k")) {
            params.sampling.top_k = opts["top_k"].get<size_t>();
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
    svr.Post("/api/show", [&model_name, &iso_timestamp](const httplib::Request&,
                                                         httplib::Response& res) {
        json resp;
        resp["modelfile"] = "# OpenVINO GenAI model";
        resp["parameters"] = "temperature 0.7\ntop_p 0.95\ntop_k 20";
        resp["template"] = "ChatML";
        json details;
        details["parent_model"] = "";
        details["format"] = "openvino";
        details["family"] = "qwen3.5";
        details["families"] = json::array({"qwen3.5"});
        details["parameter_size"] = "35B";
        details["quantization_level"] = "INT4";
        resp["details"] = details;
        resp["model_info"] = json::object();
        resp["modified_at"] = iso_timestamp();
        res.set_content(resp.dump(), "application/json");
    });

    // POST /api/chat — Ollama chat (NDJSON streaming)
    svr.Post("/api/chat",
             [&pool, &cfg, &model_name, &iso_timestamp, &parse_ollama_options](
                 const httplib::Request& req, httplib::Response& res) {
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

        // Build ChatML prompt from Ollama messages
        auto messages = body.value("messages", json::array());
        std::string chat_text;

        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            chat_text += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }

        chat_text += "<|im_start|>assistant\n";
        if (cfg.enable_thinking) {
            chat_text += "<think>\n";
        } else {
            chat_text += "<think>\n</think>\n\n";
        }

        GenerateParams params;
        params.raw_prompt = true;
        params.enable_thinking = cfg.enable_thinking;
        parse_ollama_options(body, params);

        std::string mn = model_name;

        std::cerr << "[ov_serve] ollama-chat"
                  << " temp=" << params.sampling.temperature
                  << " top_k=" << params.sampling.top_k
                  << " rep=" << params.sampling.repetition_penalty
                  << " pres=" << params.sampling.presence_penalty
                  << " max_tokens=" << params.max_new_tokens
                  << " stream=" << do_stream << "\n";

        if (do_stream) {
            // NDJSON streaming (Ollama format: one JSON per line)
            auto prompt = std::move(chat_text);
            res.set_chunked_content_provider(
                "application/x-ndjson",
                [&pool, params, prompt, mn, &cfg, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(pool);
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
                                sink.write(line.c_str(), line.size());
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
            // Non-streaming: return full response
            WorkerPool::Guard worker(pool);
            auto* session = worker.get();

            try {
                auto result = session->generate(chat_text, params);

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
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err;
                err["error"] = std::string("Generation error: ") + e.what();
                res.set_content(err.dump(), "application/json");
            }
        }
    });

    // POST /api/generate — Ollama raw generate (NDJSON streaming)
    svr.Post("/api/generate",
             [&pool, &cfg, &model_name, &iso_timestamp, &parse_ollama_options](
                 const httplib::Request& req, httplib::Response& res) {
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

        // If not raw mode, wrap in ChatML
        if (!raw) {
            std::string system = body.value("system", "");
            std::string wrapped;
            if (!system.empty()) {
                wrapped = "<|im_start|>system\n" + system + "<|im_end|>\n";
            }
            wrapped += "<|im_start|>user\n" + prompt + "<|im_end|>\n";
            wrapped += "<|im_start|>assistant\n";
            if (cfg.enable_thinking) {
                wrapped += "<think>\n";
            } else {
                wrapped += "<think>\n</think>\n\n";
            }
            prompt = wrapped;
        }

        GenerateParams params;
        params.raw_prompt = true;
        params.enable_thinking = cfg.enable_thinking;
        parse_ollama_options(body, params);

        std::string mn = model_name;

        if (do_stream) {
            res.set_chunked_content_provider(
                "application/x-ndjson",
                [&pool, params, prompt, mn, &iso_timestamp](
                    size_t, httplib::DataSink& sink) {
                    WorkerPool::Guard worker(pool);
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
                                sink.write(line.c_str(), line.size());
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
            WorkerPool::Guard worker(pool);
            auto* session = worker.get();

            try {
                auto result = session->generate(prompt, params);

                json resp;
                resp["model"] = mn;
                resp["created_at"] = iso_timestamp();
                resp["response"] = result.text;
                resp["done"] = true;
                resp["done_reason"] = (result.stop_reason == StopReason::MAX_TOKENS)
                                          ? "length" : "stop";
                resp["eval_count"] = static_cast<int>(result.generated_tokens);
                resp["prompt_eval_count"] = static_cast<int>(result.prompt_tokens);
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err;
                err["error"] = std::string("Generation error: ") + e.what();
                res.set_content(err.dump(), "application/json");
            }
        }
    });

    // ── Start server ──
    std::cerr << "[ov_serve] Starting server on " << cfg.host << ":" << cfg.port << "\n";
    std::cerr << "[ov_serve] Workers: " << cfg.workers
              << ", Device: " << cfg.device
              << ", Thinking: " << (cfg.enable_thinking ? "on" : "off")
              << ", Vision: " << (cfg.enable_vision ? "on" : "off")
              << ", Rep.Penalty: " << cfg.repetition_penalty
              << ", Pres.Penalty: " << cfg.presence_penalty << "\n";
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
