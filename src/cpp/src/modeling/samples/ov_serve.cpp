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
    int max_tokens_default = 2048;
};

// ═══════════════════════════════════════════════════════════════════
//  Worker Pool — owns N Sessions, thread-safe acquire/release
// ═══════════════════════════════════════════════════════════════════

class WorkerPool {
public:
    WorkerPool(ModelLoader& loader, int n) {
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
        if (msg.contains("content") && msg["content"].is_string()) {
            content = msg["content"].get<std::string>();
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
    req.prompt = chat_text;

    // Generation params
    req.params.max_new_tokens = body.value("max_tokens", cfg.max_tokens_default);
    req.params.enable_thinking = cfg.enable_thinking;

    float temperature = body.value("temperature", 0.7f);
    req.params.sampling.temperature = temperature;
    req.params.sampling.top_p = body.value("top_p", 0.95f);
    if (body.contains("top_k")) {
        req.params.sampling.top_k = body["top_k"].get<size_t>();
    }
    if (body.contains("seed")) {
        req.params.sampling.rng_seed = body["seed"].get<size_t>();
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
                 "[--workers 1] [--device GPU]\n";
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
        else if (arg == "--no-thinking") cfg.enable_thinking = false;
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
    lp.enable_vision = false;  // text-only for now

    ModelLoader loader(cfg.model_path, lp);

    auto t1 = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[ov_serve] Model loaded in " << load_sec << "s\n";

    // ── Create worker pool ──
    std::cerr << "[ov_serve] Creating " << cfg.workers << " worker session(s)...\n";
    WorkerPool pool(loader, cfg.workers);
    std::cerr << "[ov_serve] Workers ready\n";

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

        auto request_id = make_request_id();
        WorkerPool::Guard worker(pool);
        auto* session = worker.get();

        try {
            if (parsed.stream) {
                // ── Streaming SSE ──
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");

                // Send initial role chunk
                {
                    json chunk;
                    chunk["id"] = request_id;
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = unix_timestamp();
                    chunk["model"] = model_name;
                    json delta;
                    delta["role"] = "assistant";
                    json choice;
                    choice["index"] = 0;
                    choice["delta"] = delta;
                    chunk["choices"] = json::array({choice});
                    res.set_chunked_content_provider(
                        "text/event-stream",
                        [&](size_t /*offset*/, httplib::DataSink& sink) {
                            // Send role chunk
                            std::string role_sse = sse_chunk(chunk);
                            sink.write(role_sse.c_str(), role_sse.size());

                            ThinkingTracker thinking_tracker;
                            ToolCallParser tool_parser;
                            bool is_in_thinking = false;

                            auto callback = [&](const StreamChunk& sc) -> bool {
                                if (sc.event == StreamEvent::TOKEN && !sc.token_text.empty()) {
                                    // Process through thinking tracker
                                    auto tr = thinking_tracker.process(sc.token_text);

                                    // Send thinking content
                                    if (!tr.thinking_text.empty()) {
                                        if (!is_in_thinking) {
                                            is_in_thinking = true;
                                        }
                                        json tc;
                                        tc["id"] = request_id;
                                        tc["object"] = "chat.completion.chunk";
                                        tc["created"] = unix_timestamp();
                                        tc["model"] = model_name;
                                        json d;
                                        d["reasoning_content"] = tr.thinking_text;
                                        json c;
                                        c["index"] = 0;
                                        c["delta"] = d;
                                        tc["choices"] = json::array({c});
                                        std::string s = sse_chunk(tc);
                                        sink.write(s.c_str(), s.size());
                                    }

                                    // Send content text
                                    if (!tr.content_text.empty()) {
                                        // Run through tool parser
                                        auto pr = tool_parser.process(tr.content_text);
                                        if (!pr.text.empty()) {
                                            json tc;
                                            tc["id"] = request_id;
                                            tc["object"] = "chat.completion.chunk";
                                            tc["created"] = unix_timestamp();
                                            tc["model"] = model_name;
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
                                    // Final chunk with finish_reason
                                    auto flush_result = tool_parser.flush();
                                    std::string finish_reason =
                                        (sc.stop_reason == StopReason::MAX_TOKENS)
                                            ? "length"
                                            : "stop";

                                    json fc;
                                    fc["id"] = request_id;
                                    fc["object"] = "chat.completion.chunk";
                                    fc["created"] = unix_timestamp();
                                    fc["model"] = model_name;
                                    json d;
                                    d = json::object();  // empty delta
                                    json c;
                                    c["index"] = 0;
                                    c["delta"] = d;
                                    c["finish_reason"] = finish_reason;
                                    fc["choices"] = json::array({c});

                                    // Add usage if available
                                    json usage;
                                    usage["prompt_tokens"] = sc.prompt_tokens;
                                    usage["completion_tokens"] = sc.generated_tokens;
                                    usage["total_tokens"] = sc.prompt_tokens + sc.generated_tokens;
                                    fc["usage"] = usage;

                                    std::string s = sse_chunk(fc);
                                    sink.write(s.c_str(), s.size());

                                    // Send [DONE]
                                    std::string done = sse_done();
                                    sink.write(done.c_str(), done.size());
                                }
                                return true;  // continue generating
                            };

                            session->generate(parsed.prompt, parsed.params, callback);
                            sink.done();
                            return true;
                        });
                }
            } else {
                // ── Non-streaming ──
                auto result = session->generate(parsed.prompt, parsed.params);

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
        // RAII Guard handles session release automatically
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
        params.sampling.top_p = body.value("top_p", 0.95f);
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

    // ── Start server ──
    std::cerr << "[ov_serve] Starting server on " << cfg.host << ":" << cfg.port << "\n";
    std::cerr << "[ov_serve] Workers: " << cfg.workers
              << ", Device: " << cfg.device
              << ", Thinking: " << (cfg.enable_thinking ? "on" : "off") << "\n";
    std::cerr << "[ov_serve] Endpoints:\n"
              << "  POST /v1/chat/completions\n"
              << "  POST /v1/completions\n"
              << "  GET  /v1/models\n"
              << "  GET  /health\n";

    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "[ov_serve] Failed to start server on " << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }

    return 0;
}
