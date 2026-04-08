/* Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Qwen3.5 Modeling Library — C API
 *
 * Opaque handle + error code pattern for FFI consumers (Go, Python, Rust).
 * Thread safety: model handle can be shared across sessions;
 *                single session is NOT reentrant.
 */

#ifndef OV_MODELING_QWEN3_5_H
#define OV_MODELING_QWEN3_5_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* When compiled as part of openvino_genai shared library, openvino_genai_EXPORTS
 * is defined by CMake. Modeling samples also define it. */
#ifdef _WIN32
#  ifdef openvino_genai_EXPORTS
#    define OV_MODELING_API __declspec(dllexport)
#  else
#    define OV_MODELING_API __declspec(dllimport)
#  endif
#else
#  define OV_MODELING_API __attribute__((visibility("default")))
#endif

/* ── Opaque handles ── */
typedef struct ov_model_t ov_model_t;
typedef struct ov_session_t ov_session_t;

/* ── Error codes ── */
typedef enum {
    OV_OK = 0,
    OV_ERROR_INVALID_PARAM = -1,
    OV_ERROR_MODEL_LOAD = -2,
    OV_ERROR_OUT_OF_MEMORY = -3,
    OV_ERROR_DEVICE_NOT_FOUND = -4,
    OV_ERROR_INFERENCE = -5,
    OV_ERROR_STOPPED = -6,
    OV_ERROR_BUSY = -7,
    OV_ERROR_UNKNOWN = -99,
} ov_status_t;

/* ── Load parameters ── */
typedef struct {
    const char* device;           /* "GPU", "CPU", NULL = "GPU" */
    int cache_ir;                 /* 1 = cache IR to disk */
    int enable_vision;            /* 1 = load vision model */
    const char* quant_mode;       /* "int4_asym", "int8_sym", NULL = from env */
    int quant_group_size;         /* 128, 0 = from env */
    int num_layers;               /* 0 = all layers, >0 = override */
} ov_load_params_t;

/* ── Generation parameters ── */
typedef struct {
    int max_new_tokens;
    float temperature;            /* 0.0 = greedy */
    float top_p;
    int top_k;
    float repetition_penalty;
    float frequency_penalty;
    float presence_penalty;
    int enable_thinking;          /* 1 = thinking mode */
    uint64_t rng_seed;            /* 0 = random */
} ov_gen_params_t;

/* ── Stream event types ── */
typedef enum {
    OV_EVENT_PREFILL_DONE = 0,
    OV_EVENT_TOKEN = 1,
    OV_EVENT_THINKING_START = 2,
    OV_EVENT_THINKING_END = 3,
    OV_EVENT_FINISH = 4,
} ov_stream_event_t;

/* ── Stream chunk passed to callback ── */
typedef struct {
    ov_stream_event_t event;
    int64_t token_id;             /* -1 if not applicable */
    const char* token_text;       /* NULL if not applicable */
    int is_thinking;              /* 1 = inside <think> block */
    /* FINISH fields */
    int stop_reason;              /* 0=EOS, 1=MAX_TOKENS, 2=USER_STOP, 3=ERROR */
    int prompt_tokens;
    int generated_tokens;
    double prefill_ms;
    double decode_ms;
    double ttft_ms;
    double throughput;
} ov_stream_chunk_t;

/* Stream callback. Return 0 to continue, non-zero to stop. */
typedef int (*ov_stream_callback_t)(const ov_stream_chunk_t* chunk, void* user_data);

/* ── Generation result ── */
typedef struct {
    char* text;                   /* Decoded content text (ov_free to release) */
    char* thinking_text;          /* Thinking block text (ov_free to release, may be NULL) */
    int prompt_tokens;
    int generated_tokens;
    int thinking_tokens;
    double prefill_ms;
    double decode_ms;
    double ttft_ms;
    double throughput;
    int stop_reason;              /* 0=EOS, 1=MAX_TOKENS, 2=USER_STOP, 3=ERROR */
} ov_gen_result_t;

/* ── Model info ── */
typedef struct {
    char model_name[256];
    int vocab_size;
    int hidden_size;
    int num_layers;
    int has_vision;
} ov_model_info_t;

/* ═══════════════════════════════════════════════
 *                  CORE API
 * ═══════════════════════════════════════════════ */

/* ── Default parameters ── */
OV_MODELING_API ov_load_params_t  ov_default_load_params(void);
OV_MODELING_API ov_gen_params_t   ov_default_gen_params(void);

/* ── Model lifecycle ── */
OV_MODELING_API ov_model_t*   ov_model_load(const char* model_path,
                                             const ov_load_params_t* params);
OV_MODELING_API void          ov_model_free(ov_model_t* model);
OV_MODELING_API ov_status_t   ov_get_model_info(const ov_model_t* model,
                                                 ov_model_info_t* out);

/* ── Session lifecycle ── */
OV_MODELING_API ov_session_t* ov_session_create(ov_model_t* model);
OV_MODELING_API void          ov_session_free(ov_session_t* session);
OV_MODELING_API void          ov_session_reset(ov_session_t* session);

/* ── Text generation ── */
OV_MODELING_API ov_status_t   ov_generate(ov_session_t* session,
                                           const char* prompt,
                                           const ov_gen_params_t* params,
                                           ov_gen_result_t* result);

OV_MODELING_API ov_status_t   ov_generate_stream(ov_session_t* session,
                                                  const char* prompt,
                                                  const ov_gen_params_t* params,
                                                  ov_stream_callback_t callback,
                                                  void* user_data);

/* ── Control ── */
OV_MODELING_API void          ov_generate_stop(ov_session_t* session);

/* ── Error handling ── */
OV_MODELING_API const char*   ov_get_last_error(void);

/* ── Memory management ── */
OV_MODELING_API void          ov_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* OV_MODELING_QWEN3_5_H */
