// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/ov_modeling_qwen3_5.h"

#include <cstring>
#include <memory>
#include <string>

#include "modeling/api/model_loader.hpp"
#include "modeling/api/session.hpp"
#include "modeling/api/types.hpp"

// Thread-local last error message.
static thread_local std::string g_last_error;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

// ─── Opaque handle structs ───

struct ov_model_t {
    std::unique_ptr<ov::genai::modeling::ModelLoader> loader;
};

struct ov_session_t {
    std::unique_ptr<ov::genai::modeling::Session> session;
    ov_model_t* model;  // non-owning back-pointer
};

// ─── Helpers ───

static char* strdup_safe(const std::string& s) {
    if (s.empty()) return nullptr;
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) {
        std::memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

static ov::genai::modeling::GenerateParams to_cpp_params(const ov_gen_params_t* p) {
    ov::genai::modeling::GenerateParams params;
    if (!p) return params;

    params.max_new_tokens = p->max_new_tokens > 0 ? p->max_new_tokens : 256;
    params.enable_thinking = (p->enable_thinking != 0);
    params.sampling.temperature = p->temperature;
    params.sampling.top_p = p->top_p > 0.0f ? p->top_p : 0.95f;
    params.sampling.top_k = p->top_k > 0 ? static_cast<size_t>(p->top_k) : 20;
    params.sampling.repetition_penalty = p->repetition_penalty > 0.0f ? p->repetition_penalty : 1.0f;
    params.sampling.frequency_penalty = p->frequency_penalty;
    params.sampling.presence_penalty = p->presence_penalty;
    params.sampling.rng_seed = p->rng_seed;
    return params;
}

static int stop_reason_to_int(ov::genai::modeling::StopReason r) {
    switch (r) {
        case ov::genai::modeling::StopReason::EOS: return 0;
        case ov::genai::modeling::StopReason::MAX_TOKENS: return 1;
        case ov::genai::modeling::StopReason::USER_STOP: return 2;
        case ov::genai::modeling::StopReason::STOP_STRING: return 3;
        case ov::genai::modeling::StopReason::TIMEOUT: return 4;
        case ov::genai::modeling::StopReason::ERROR: return 5;
        default: return -1;
    }
}

// ─── Default params ───

extern "C" ov_load_params_t ov_default_load_params(void) {
    ov_load_params_t p{};
    p.device = "GPU";
    p.cache_ir = 1;
    p.enable_vision = 0;
    p.quant_mode = nullptr;
    p.quant_group_size = 0;
    p.num_layers = 0;
    return p;
}

extern "C" ov_gen_params_t ov_default_gen_params(void) {
    ov_gen_params_t p{};
    p.max_new_tokens = 256;
    p.temperature = 0.0f;
    p.top_p = 0.95f;
    p.top_k = 20;
    p.repetition_penalty = 1.0f;
    p.frequency_penalty = 0.0f;
    p.presence_penalty = 0.0f;
    p.enable_thinking = 1;
    p.rng_seed = 0;
    return p;
}

// ─── Model lifecycle ───

extern "C" ov_model_t* ov_model_load(const char* model_path,
                                      const ov_load_params_t* params) {
    if (!model_path) {
        set_error("model_path is NULL");
        return nullptr;
    }

    try {
        ov::genai::modeling::LoadParams lp;
        if (params) {
            if (params->device) lp.device = params->device;
            lp.cache_ir = (params->cache_ir != 0);
            lp.enable_vision = (params->enable_vision != 0);
            if (params->num_layers > 0) lp.num_layers = params->num_layers;
        }

        auto model = std::make_unique<ov_model_t>();
        model->loader = std::make_unique<ov::genai::modeling::ModelLoader>(
            model_path, lp);
        return model.release();

    } catch (const std::exception& e) {
        set_error(std::string("ov_model_load failed: ") + e.what());
        return nullptr;
    }
}

extern "C" void ov_model_free(ov_model_t* model) {
    delete model;
}

extern "C" ov_status_t ov_get_model_info(const ov_model_t* model,
                                          ov_model_info_t* out) {
    if (!model || !out) {
        set_error("NULL parameter");
        return OV_ERROR_INVALID_PARAM;
    }

    try {
        std::memset(out, 0, sizeof(*out));
        const auto& cfg = model->loader->config();

        std::strncpy(out->model_name, "qwen3.5", sizeof(out->model_name) - 1);
        out->vocab_size = cfg.text.vocab_size;
        out->hidden_size = cfg.text.hidden_size;
        out->num_layers = cfg.text.num_hidden_layers;
        out->has_vision = model->loader->compiled_vision() ? 1 : 0;
        return OV_OK;

    } catch (const std::exception& e) {
        set_error(std::string("ov_get_model_info failed: ") + e.what());
        return OV_ERROR_UNKNOWN;
    }
}

// ─── Session lifecycle ───

extern "C" ov_session_t* ov_session_create(ov_model_t* model) {
    if (!model) {
        set_error("model is NULL");
        return nullptr;
    }

    try {
        auto session = std::make_unique<ov_session_t>();
        session->session = std::make_unique<ov::genai::modeling::Session>(
            *model->loader);
        session->model = model;
        return session.release();

    } catch (const std::exception& e) {
        set_error(std::string("ov_session_create failed: ") + e.what());
        return nullptr;
    }
}

extern "C" void ov_session_free(ov_session_t* session) {
    delete session;
}

extern "C" void ov_session_reset(ov_session_t* session) {
    if (session && session->session) {
        session->session->reset();
    }
}

// ─── Text generation ───

extern "C" ov_status_t ov_generate(ov_session_t* session,
                                    const char* prompt,
                                    const ov_gen_params_t* params,
                                    ov_gen_result_t* result) {
    if (!session || !prompt || !result) {
        set_error("NULL parameter");
        return OV_ERROR_INVALID_PARAM;
    }

    try {
        auto cpp_params = to_cpp_params(params);
        auto gen_result = session->session->generate(prompt, cpp_params);

        std::memset(result, 0, sizeof(*result));
        result->text = strdup_safe(gen_result.text);
        result->thinking_text = strdup_safe(gen_result.thinking_text);
        result->prompt_tokens = gen_result.prompt_tokens;
        result->generated_tokens = gen_result.generated_tokens;
        result->thinking_tokens = gen_result.thinking_tokens;
        result->prefill_ms = gen_result.prefill_ms;
        result->decode_ms = gen_result.decode_ms;
        result->ttft_ms = gen_result.ttft_ms;
        result->throughput = gen_result.throughput;
        result->stop_reason = stop_reason_to_int(gen_result.stop_reason);
        return OV_OK;

    } catch (const std::exception& e) {
        set_error(std::string("ov_generate failed: ") + e.what());
        return OV_ERROR_INFERENCE;
    }
}

extern "C" ov_status_t ov_generate_stream(ov_session_t* session,
                                           const char* prompt,
                                           const ov_gen_params_t* params,
                                           ov_stream_callback_t callback,
                                           void* user_data) {
    if (!session || !prompt || !callback) {
        set_error("NULL parameter");
        return OV_ERROR_INVALID_PARAM;
    }

    try {
        auto cpp_params = to_cpp_params(params);

        // Wrap C callback into C++ StreamCallback
        auto cpp_callback = [callback, user_data](
            const ov::genai::modeling::StreamChunk& chunk) -> bool {

            ov_stream_chunk_t c{};
            switch (chunk.event) {
                case ov::genai::modeling::StreamEvent::PREFILL_DONE:
                    c.event = OV_EVENT_PREFILL_DONE;
                    break;
                case ov::genai::modeling::StreamEvent::TOKEN:
                    c.event = OV_EVENT_TOKEN;
                    break;
                case ov::genai::modeling::StreamEvent::THINKING_START:
                    c.event = OV_EVENT_THINKING_START;
                    break;
                case ov::genai::modeling::StreamEvent::THINKING_END:
                    c.event = OV_EVENT_THINKING_END;
                    break;
                case ov::genai::modeling::StreamEvent::FINISH:
                    c.event = OV_EVENT_FINISH;
                    break;
            }

            c.token_id = chunk.token_id;
            c.token_text = chunk.token_text.empty() ? nullptr : chunk.token_text.c_str();
            c.is_thinking = chunk.is_thinking ? 1 : 0;
            c.stop_reason = static_cast<int>(chunk.stop_reason);
            c.prompt_tokens = chunk.prompt_tokens;
            c.generated_tokens = chunk.generated_tokens;
            c.prefill_ms = chunk.prefill_ms;
            c.decode_ms = chunk.decode_ms;
            c.ttft_ms = chunk.ttft_ms;
            c.throughput = chunk.throughput;

            return callback(&c, user_data) == 0;  // 0 = continue
        };

        session->session->generate(prompt, cpp_params, std::move(cpp_callback));
        return OV_OK;

    } catch (const std::exception& e) {
        set_error(std::string("ov_generate_stream failed: ") + e.what());
        return OV_ERROR_INFERENCE;
    }
}

// ─── Control ───

extern "C" void ov_generate_stop(ov_session_t* session) {
    if (session && session->session) {
        session->session->stop();
    }
}

// ─── Error handling ───

extern "C" const char* ov_get_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

// ─── Memory management ───

extern "C" void ov_free(void* ptr) {
    std::free(ptr);
}
