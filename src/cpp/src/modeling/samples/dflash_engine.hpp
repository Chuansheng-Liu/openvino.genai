// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// DFlashEngine: Speculative decoding engine for ov_serve integration.
// Wraps 3 DFlash sub-models (target, context_fc, combined_draft) and provides
// a generate() method with streaming callback support.
//
// Usage:
//   DFlashEngine engine(config);
//   engine.generate(token_ids, max_tokens, [](int64_t token, bool eos) { ... });

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "loaders/model_config.hpp"
#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "utils.hpp"

namespace dflash {

// ═══════════════════════════════════════════════════════════════════
//  Utility functions (adapted from modeling_qwen3_5_dflash.cpp)
// ═══════════════════════════════════════════════════════════════════

namespace detail {

inline ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor t(ov::element::i32, {batch});
    auto* d = t.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) d[i] = static_cast<int32_t>(i);
    return t;
}

inline ov::Tensor make_state_update_mode_tensor(int32_t mode) {
    ov::Tensor t(ov::element::i32, {1});
    t.data<int32_t>()[0] = mode;
    return t;
}

inline int64_t resolve_mask_token_id(ov::genai::Tokenizer& tokenizer) {
    const auto vocab = tokenizer.get_vocab();
    auto it = vocab.find("<|MASK|>");
    if (it != vocab.end()) return it->second;
    auto try_single = [&](const std::string& tok) -> std::optional<int64_t> {
        try {
            auto enc = tokenizer.encode(tok, {ov::genai::add_special_tokens(false)}).input_ids;
            if (enc.get_shape() == ov::Shape{1, 1}) return enc.data<const int64_t>()[0];
        } catch (...) {}
        return std::nullopt;
    };
    if (auto v = try_single("<|fim_pad|>")) return *v;
    if (auto v = try_single("<|vision_pad|>")) return *v;
    int64_t pad = tokenizer.get_pad_token_id();
    if (pad != -1) return pad;
    int64_t eos = tokenizer.get_eos_token_id();
    if (eos != -1) return eos;
    throw std::runtime_error("No suitable mask token found");
}

template <typename T>
int64_t argmax_row(const T* data, size_t vocab) {
    T mx = data[0]; size_t mi = 0;
    for (size_t i = 1; i < vocab; ++i)
        if (data[i] > mx) { mx = data[i]; mi = i; }
    return static_cast<int64_t>(mi);
}

inline int64_t argmax_row_f16_fast(const uint16_t* data, size_t vocab) {
    auto to_sortable = [](uint16_t v) -> uint16_t {
        return (v & 0x8000) ? static_cast<uint16_t>(~v) : static_cast<uint16_t>(v ^ 0x8000);
    };
    uint16_t mx = to_sortable(data[0]); size_t mi = 0;
    for (size_t i = 1; i < vocab; ++i) {
        uint16_t sv = to_sortable(data[i]);
        if (sv > mx) { mx = sv; mi = i; }
    }
    return static_cast<int64_t>(mi);
}

inline std::vector<int64_t> argmax_logits_slice(const ov::Tensor& logits, size_t start, size_t count) {
    const auto shape = logits.get_shape();
    const size_t vocab = shape[2];
    std::vector<int64_t> tokens;
    tokens.reserve(count);
    if (logits.get_element_type() == ov::element::f16) {
        const auto* d = reinterpret_cast<const uint16_t*>(logits.data<const ov::float16>());
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row_f16_fast(d + (start + i) * vocab, vocab));
    } else if (logits.get_element_type() == ov::element::f32) {
        const auto* d = logits.data<const float>();
        for (size_t i = 0; i < count; ++i)
            tokens.push_back(argmax_row(d + (start + i) * vocab, vocab));
    } else {
        throw std::runtime_error("Unsupported logits dtype");
    }
    return tokens;
}

inline int64_t argmax_last_token(const ov::Tensor& logits) {
    return argmax_logits_slice(logits, logits.get_shape()[1] - 1, 1).front();
}

inline ov::Tensor copy_to_host(const ov::Tensor& t) {
    ov::Tensor out(t.get_element_type(), t.get_shape());
    t.copy_to(out);
    return out;
}

inline bool deferred_state_commit_enabled() {
    const char* raw = std::getenv("OV_GENAI_DISABLE_DFLASH_DEFERRED_STATE_COMMIT");
    return !(raw && std::string(raw) == "1");
}

inline std::string quant_mode_token(ov::genai::modeling::weights::QuantizationConfig::Mode m) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    switch (m) {
        case Mode::INT4_SYM:  return "4s";
        case Mode::INT4_ASYM: return "4a";
        case Mode::INT8_SYM:  return "8s";
        case Mode::INT8_ASYM: return "8a";
        default:              return "n";
    }
}

inline std::string quant_cache_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) return "";
    return "_q" + quant_mode_token(cfg.mode) + "_b" + quant_mode_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

inline std::optional<std::pair<std::filesystem::path, std::filesystem::path>>
find_ir(const std::filesystem::path& dir, const std::string& stem) {
    auto xml = dir / (stem + ".xml");
    auto bin = dir / (stem + ".bin");
    if (std::filesystem::exists(xml) && std::filesystem::exists(bin))
        return std::make_pair(xml, bin);
    return std::nullopt;
}

inline std::vector<std::pair<std::string, ov::Tensor>>
save_linear_states(ov::InferRequest& req) {
    std::vector<std::pair<std::string, ov::Tensor>> saved;
    for (auto& state : req.query_state()) {
        const auto& name = state.get_name();
        if (name.find("linear_states.") != std::string::npos) {
            auto src = state.get_state();
            ov::Tensor copy(src.get_element_type(), src.get_shape());
            src.copy_to(copy);
            saved.emplace_back(name, std::move(copy));
        }
    }
    return saved;
}

inline void restore_linear_states(ov::InferRequest& req,
                                  const std::vector<std::pair<std::string, ov::Tensor>>& saved) {
    for (auto& state : req.query_state()) {
        const auto& name = state.get_name();
        for (const auto& [sn, st] : saved) {
            if (name == sn) { state.set_state(st); break; }
        }
    }
}

}  // namespace detail

// ═══════════════════════════════════════════════════════════════════
//  DFlashEngine — speculative decoding engine
// ═══════════════════════════════════════════════════════════════════

struct DFlashConfig {
    std::filesystem::path target_model_dir;  // Where HF model + IR files live
    std::filesystem::path draft_model_dir;   // Where draft config.json lives
    std::string device = "GPU";
};

struct DFlashGenerateResult {
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double throughput = 0.0;  // tokens/s
    size_t draft_steps = 0;
    size_t accepted_tokens = 0;
    std::string finish_reason;  // "stop" or "length"
};

// Callback: (token_text, is_eos) → continue?
using DFlashStreamCallback = std::function<bool(const std::string& token_text, bool is_eos)>;

/// UTF-8-safe streaming text buffer for detokenization.
/// Accumulates token IDs, re-decodes the full sequence each time, and emits
/// only the stable prefix that forms valid UTF-8.  This prevents emitting
/// partial multi-byte characters (e.g. half a CJK codepoint → U+FFFD).
struct StreamTextBuffer {
    ov::genai::Tokenizer& tokenizer;
    std::vector<int64_t> tokens;
    size_t printed_len = 0;

    explicit StreamTextBuffer(ov::genai::Tokenizer& tok) : tokenizer(tok) {}

    static bool ends_with_replacement(const std::string& s) {
        // U+FFFD in UTF-8: EF BF BD
        return s.size() >= 3 &&
               s[s.size()-3] == '\xef' &&
               s[s.size()-2] == '\xbf' &&
               s[s.size()-1] == '\xbd';
    }

    /// Add new token(s) and return the safe-to-emit delta text.
    std::string push(const std::vector<int64_t>& new_ids) {
        tokens.insert(tokens.end(), new_ids.begin(), new_ids.end());
        auto full = tokenizer.decode(tokens, {ov::genai::skip_special_tokens(true)});
        if (ends_with_replacement(full))
            return {};  // trailing incomplete UTF-8 — wait
        if (full.size() > printed_len) {
            std::string delta = full.substr(printed_len);
            printed_len = full.size();
            return delta;
        }
        return {};
    }

    std::string push(int64_t id) { return push(std::vector<int64_t>{id}); }

    /// Flush any remaining buffered text.
    std::string flush() {
        if (tokens.empty()) return {};
        auto full = tokenizer.decode(tokens, {ov::genai::skip_special_tokens(true)});
        if (full.size() > printed_len) {
            std::string delta = full.substr(printed_len);
            printed_len = full.size();
            return delta;
        }
        return {};
    }
};

/// Pre-processed VL inputs for DFlash generate.
struct DFlashVLInputs {
    ov::Tensor input_ids;       // [1, seq_len]
    ov::Tensor attention_mask;  // [1, seq_len]
    ov::Tensor position_ids;    // [3, 1, seq_len] mRoPE
    ov::Tensor visual_embeds;   // [1, seq_len, hidden_size] scattered
    ov::Tensor visual_pos_mask; // [1, seq_len] boolean
};

class DFlashEngine {
public:
    explicit DFlashEngine(const DFlashConfig& cfg)
        : target_dir_(cfg.target_model_dir)
        , draft_dir_(cfg.draft_model_dir)
        , device_(cfg.device)
    {
        using namespace ov::genai::modeling;
        namespace fs = std::filesystem;

        // Load configs
        auto target_qwen35_cfg = models::Qwen3_5Config::from_json_file(target_dir_);
        auto draft_cfg_raw = ov::genai::loaders::ModelConfig::from_hf_json(draft_dir_ / "config.json");

        models::DFlashDraftConfig dcfg;
        dcfg.hidden_size = draft_cfg_raw.hidden_size;
        dcfg.intermediate_size = draft_cfg_raw.intermediate_size;
        dcfg.num_hidden_layers = draft_cfg_raw.num_hidden_layers;
        dcfg.num_target_layers = (draft_cfg_raw.num_target_layers > 0)
            ? draft_cfg_raw.num_target_layers
            : target_qwen35_cfg.text.num_hidden_layers;
        dcfg.num_attention_heads = draft_cfg_raw.num_attention_heads;
        dcfg.num_key_value_heads = draft_cfg_raw.num_key_value_heads > 0
            ? draft_cfg_raw.num_key_value_heads : draft_cfg_raw.num_attention_heads;
        dcfg.head_dim = draft_cfg_raw.head_dim > 0
            ? draft_cfg_raw.head_dim : (draft_cfg_raw.hidden_size / draft_cfg_raw.num_attention_heads);
        dcfg.block_size = draft_cfg_raw.block_size > 0 ? draft_cfg_raw.block_size : 16;
        dcfg.rms_norm_eps = draft_cfg_raw.rms_norm_eps;
        dcfg.rope_theta = draft_cfg_raw.rope_theta;
        dcfg.hidden_act = draft_cfg_raw.hidden_act;
        dcfg.attention_bias = draft_cfg_raw.attention_bias;
        dcfg.target_layer_ids = draft_cfg_raw.target_layer_ids;
        dcfg.mask_token_id = draft_cfg_raw.mask_token_id;
        if (dcfg.target_layer_ids.empty()) {
            dcfg.target_layer_ids = models::build_target_layer_ids(
                dcfg.num_target_layers, dcfg.num_hidden_layers);
        }
        dflash_cfg_ = dcfg;
        block_size_ = static_cast<size_t>(dcfg.block_size);
        hidden_dim_ = 0;  // will be set from actual target_hidden output after first prefill
        model_hidden_size_ = static_cast<size_t>(target_qwen35_cfg.text.hidden_size);
        ctx_hidden_dim_ = static_cast<size_t>(dcfg.hidden_size);

        std::cerr << "[DFlashEngine] block_size=" << block_size_
                  << " target_layers=" << dcfg.target_layer_ids.size()
                  << " draft_layers=" << dcfg.num_hidden_layers << "\n";

        // Find IR files
        auto quant_cfg = weights::parse_quantization_config_from_env();
        std::string qsuffix = detail::quant_cache_suffix(quant_cfg);
        // Try non-VL first, then VL fallback
        auto cached_target = detail::find_ir(target_dir_, "qwen3_5_dflash_target" + qsuffix);
        if (!cached_target.has_value()) {
            cached_target = detail::find_ir(target_dir_, "qwen3_5_dflash_target_vl" + qsuffix);
            if (cached_target.has_value()) target_ir_is_vl_ = true;
        }
        auto cached_ctx_fc = detail::find_ir(target_dir_, "qwen3_5_dflash_context_fc");
        auto cached_draft = detail::find_ir(target_dir_, "qwen3_5_dflash_combined_draft_v2" + qsuffix);

        if (!cached_target || !cached_ctx_fc || !cached_draft) {
            throw std::runtime_error(
                "DFlash IR files not found in " + target_dir_.string() +
                ". Run convert_ir --dflash to generate them.");
        }

        // Load IR models
        ov::Core core;
        std::cerr << "[DFlashEngine] Loading target IR: " << cached_target->first.filename() << "\n";
        auto target_model = core.read_model(cached_target->first.string(), cached_target->second.string());
        std::cerr << "[DFlashEngine] Loading context_fc IR: " << cached_ctx_fc->first.filename() << "\n";
        auto context_fc_model = core.read_model(cached_ctx_fc->first.string(), cached_ctx_fc->second.string());
        std::cerr << "[DFlashEngine] Loading draft IR: " << cached_draft->first.filename() << "\n";
        auto combined_draft_model = core.read_model(cached_draft->first.string(), cached_draft->second.string());

        // Detect VL inputs
        if (!target_ir_is_vl_) {
            for (const auto& input : target_model->inputs()) {
                for (const auto& name : input.get_names()) {
                    if (name == "visual_embeds") { target_ir_is_vl_ = true; break; }
                }
                if (target_ir_is_vl_) break;
            }
        }
        if (target_ir_is_vl_)
            std::cerr << "[DFlashEngine] VL target IR loaded — will supply zero visual inputs\n";

        // Apply f16 preprocessing for draft context_hidden input
        try {
            ov::preprocess::PrePostProcessor ppp(combined_draft_model);
            ppp.input("context_hidden").tensor().set_element_type(ov::element::f16);
            ppp.input("context_hidden").preprocess().convert_element_type(ov::element::f32);
            combined_draft_model = ppp.build();
            use_f16_ctx_ = true;
        } catch (const std::exception&) {}

        // Detect and convert snapshot outputs to f16
        {
            bool has_snapshots_in_model = false;
            for (auto& out : target_model->get_results()) {
                for (auto& name : out->get_output_tensor(0).get_names()) {
                    if (name.find("snapshot.") == 0) { has_snapshots_in_model = true; break; }
                }
                if (has_snapshots_in_model) break;
            }
            if (has_snapshots_in_model) {
                ov::preprocess::PrePostProcessor ppp(target_model);
                int cnt = 0;
                for (size_t idx = 0; idx < target_model->outputs().size(); ++idx) {
                    for (auto& name : target_model->output(idx).get_names()) {
                        if (name.find("snapshot.") == 0 &&
                            target_model->output(idx).get_element_type() == ov::element::f32) {
                            ppp.output(idx).postprocess().convert_element_type(ov::element::f16);
                            ++cnt;
                            break;
                        }
                    }
                }
                if (cnt > 0) {
                    target_model = ppp.build();
                    target_model->validate_nodes_and_infer_types();
                }
            }
        }

        // Compile models
        ov::AnyMap compile_opts = {
            {ov::hint::inference_precision.name(), ov::element::f16},
            {ov::hint::kv_cache_precision.name(), ov::element::f16},
            {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
            {ov::hint::dynamic_quantization_group_size.name(), uint64_t{128}},
            {ov::intel_gpu::hint::enable_kernels_reuse.name(), true},
        };

        std::cerr << "[DFlashEngine] Compiling target on " << device_ << "...\n";
        compiled_target_ = core.compile_model(target_model, device_, compile_opts);
        std::cerr << "[DFlashEngine] Compiling context_fc on CPU...\n";
        compiled_ctx_fc_ = core.compile_model(context_fc_model, "CPU", {});
        std::cerr << "[DFlashEngine] Compiling draft on " << device_ << "...\n";
        compiled_draft_ = core.compile_model(combined_draft_model, device_, compile_opts);
        std::cerr << "[DFlashEngine] All models compiled.\n";

        // Create infer requests
        target_req_ = compiled_target_.create_infer_request();
        ctx_fc_req_ = compiled_ctx_fc_.create_infer_request();
        draft_req_ = compiled_draft_.create_infer_request();

        // Release model graphs
        target_model.reset();
        context_fc_model.reset();
        combined_draft_model.reset();

        // Setup KV state tracking
        target_req_.reset_state();
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target_.get_runtime_model());
        kv_state_.seq_length_axis = kv_pos.seq_len;

        // Detect state_update_mode + snapshots
        for (const auto& input : compiled_target_.inputs()) {
            if (input.get_names().count("state_update_mode")) {
                has_state_update_mode_ = true; break;
            }
        }
        for (auto& output : compiled_target_.outputs()) {
            for (auto& name : output.get_names()) {
                if (name.find("snapshot.") == 0) { has_snapshots_ = true; break; }
            }
            if (has_snapshots_) break;
        }
        use_deferred_commit_ = has_snapshots_ && has_state_update_mode_ &&
                               detail::deferred_state_commit_enabled();

        // GPU context for USM
        try { remote_ctx_ = compiled_target_.get_context(); has_gpu_ctx_ = true; }
        catch (...) { has_gpu_ctx_ = false; }

        // Tokenizer
        tokenizer_ = std::make_unique<ov::genai::Tokenizer>(target_dir_);
        // Use mask_token_id from draft config if available (must match training);
        // fall back to tokenizer heuristic otherwise.
        mask_token_id_ = (dflash_cfg_.mask_token_id > 0)
                             ? dflash_cfg_.mask_token_id
                             : detail::resolve_mask_token_id(*tokenizer_);
        eos_token_id_ = tokenizer_->get_eos_token_id();

        // Pre-allocate reusable tensors
        beam_idx_ = detail::make_beam_idx(1);

        std::cerr << "[DFlashEngine] Ready. mask_token_id=" << mask_token_id_
                  << " has_snapshots=" << has_snapshots_
                  << " deferred_commit=" << use_deferred_commit_ << "\n";
    }

    ov::genai::Tokenizer& tokenizer() { return *tokenizer_; }

    /// Generate tokens from pre-rendered prompt text.
    /// Callback receives (decoded_text, is_eos) and returns true to continue.
    DFlashGenerateResult generate(const std::string& prompt, int max_new_tokens,
                                  DFlashStreamCallback callback,
                                  const DFlashVLInputs* vl = nullptr) {
        using Clock = std::chrono::steady_clock;
        auto duration_ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        DFlashGenerateResult result;

        // Reset state from previous request
        reset_state();

        // Tokenize / prepare input tensors
        ov::Tensor input_ids_tensor;
        ov::Tensor position_ids_tensor;
        size_t prompt_len;

        if (vl) {
            // VL path: use pre-processed inputs
            input_ids_tensor = vl->input_ids;
            position_ids_tensor = vl->position_ids;
            prompt_len = input_ids_tensor.get_shape()[1];
        } else {
            // Text path: tokenize and build simple mRoPE
            auto encoded = tokenizer_->encode(prompt, {ov::genai::add_special_tokens(false)});
            input_ids_tensor = encoded.input_ids;
            prompt_len = input_ids_tensor.get_shape()[1];

            ov::Tensor pos(ov::element::i64, {3, 1, prompt_len});
            auto* pd = pos.data<int64_t>();
            for (size_t dim = 0; dim < 3; ++dim)
                for (size_t i = 0; i < prompt_len; ++i)
                    pd[dim * prompt_len + i] = static_cast<int64_t>(i);
            position_ids_tensor = pos;
        }
        result.prompt_tokens = prompt_len;

        const size_t max_length = prompt_len + static_cast<size_t>(max_new_tokens);

        // ── Prefill ──
        auto prefill_start = Clock::now();

        target_req_.set_tensor("input_ids", input_ids_tensor);
        {
            ov::Tensor mask(ov::element::i64, {1, prompt_len});
            std::fill_n(mask.data<int64_t>(), prompt_len, 1LL);
            target_req_.set_tensor("attention_mask", mask);
        }
        target_req_.set_tensor("position_ids", position_ids_tensor);
        target_req_.set_tensor("beam_idx", beam_idx_);
        if (has_state_update_mode_)
            target_req_.set_tensor("state_update_mode", detail::make_state_update_mode_tensor(1));

        // VL inputs for prefill
        if (vl) {
            // Real visual embeddings from vision encoder
            target_req_.set_tensor("visual_embeds", vl->visual_embeds);
            target_req_.set_tensor("visual_pos_mask", vl->visual_pos_mask);
        } else if (target_ir_is_vl_) {
            // Text-only on VL IR: zero visual inputs
            ov::Tensor zve(ov::element::f32, {1, prompt_len, model_hidden_size_});
            std::memset(zve.data(), 0, zve.get_byte_size());
            target_req_.set_tensor("visual_embeds", zve);
            ov::Tensor zpm(ov::element::boolean, {1, prompt_len});
            std::memset(zpm.data(), 0, zpm.get_byte_size());
            target_req_.set_tensor("visual_pos_mask", zpm);
        }

        target_req_.infer();

        auto logits = target_req_.get_tensor("logits");
        auto target_hidden_block = get_target_hidden();
        const auto hidden_elem_type = target_hidden_block.get_element_type();

        // Resolve hidden_dim from actual model output (may differ from config)
        hidden_dim_ = target_hidden_block.get_shape()[2];

        // Allocate hidden storage
        const size_t storage_len = max_length + block_size_;
        if (has_gpu_ctx_) {
            try {
                target_hidden_storage_ = remote_ctx_.create_host_tensor(
                    hidden_elem_type, {1, storage_len, hidden_dim_});
                using_usm_storage_ = true;
            } catch (...) { using_usm_storage_ = false; }
        }
        if (!using_usm_storage_)
            target_hidden_storage_ = ov::Tensor(hidden_elem_type, {1, storage_len, hidden_dim_});

        // Copy prefill hidden to storage
        {
            ov::Tensor dst(target_hidden_storage_, {0, 0, 0}, {1, prompt_len, hidden_dim_});
            target_hidden_block.copy_to(dst);
        }
        target_hidden_len_ = prompt_len;

        // Bind USM output for target_hidden
        if (using_usm_storage_) {
            try {
                const size_t max_verify = std::max(prompt_len, block_size_ + 1);
                auto usm_out = remote_ctx_.create_host_tensor(
                    hidden_elem_type, {1, max_verify, hidden_dim_});
                target_req_.set_tensor("target_hidden", usm_out);
                using_usm_output_ = true;
            } catch (...) {}
        }

        // Context hidden cache (f16 for draft model)
        const ov::element::Type ctx_elem = use_f16_ctx_ ? ov::element::f16 : ov::element::f32;
        if (has_gpu_ctx_) {
            try {
                ctx_hidden_storage_ = remote_ctx_.create_host_tensor(
                    ctx_elem, {1, storage_len, ctx_hidden_dim_});
                using_usm_ctx_ = true;
            } catch (...) {}
        }
        if (!using_usm_ctx_)
            ctx_hidden_storage_ = ov::Tensor(ctx_elem, {1, storage_len, ctx_hidden_dim_});

        // Compute initial context_hidden from prefill
        {
            ov::Tensor init_th(target_hidden_storage_, {0, 0, 0}, {1, prompt_len, hidden_dim_});
            ctx_fc_req_.set_tensor("target_hidden", init_th);
            ctx_fc_req_.infer();
            auto init_ctx = ctx_fc_req_.get_tensor("context_hidden");
            ov::Tensor dst(ctx_hidden_storage_, {0, 0, 0}, {1, prompt_len, ctx_hidden_dim_});
            copy_ctx_to_cache(init_ctx, dst);
        }
        ctx_hidden_len_ = prompt_len;

        // USM logits binding
        const auto logits_elem = logits.get_element_type();
        const size_t vocab_size = logits.get_shape().back();
        if (has_gpu_ctx_) {
            try {
                auto dl = remote_ctx_.create_host_tensor(logits_elem, {1, block_size_, vocab_size});
                draft_req_.set_tensor("logits", dl);
            } catch (...) {}
            try {
                auto tl = remote_ctx_.create_host_tensor(logits_elem, {1, block_size_ + 1, vocab_size});
                target_req_.set_tensor("logits", tl);
            } catch (...) {}
        }

        int64_t next_token = detail::argmax_last_token(logits);
        auto prefill_end = Clock::now();
        result.ttft_ms = duration_ms(prefill_start, prefill_end);

        std::vector<int64_t> output_ids;
        output_ids.reserve(max_new_tokens);
        output_ids.push_back(next_token);

        // UTF-8-safe streaming buffer
        StreamTextBuffer strbuf(*tokenizer_);

        // Stream first token
        if (is_stop_token(next_token)) {
            auto text = strbuf.push(next_token);
            auto remaining = strbuf.flush();
            text += remaining;
            if (!text.empty()) callback(text, true);
            result.generated_tokens = 1;
            result.finish_reason = "stop";
            result.decode_ms = 0;
            result.throughput = 0;
            return result;
        }
        {
            auto text = strbuf.push(next_token);
            if (!text.empty()) {
                if (!callback(text, false)) {
                    result.generated_tokens = output_ids.size();
                    result.finish_reason = "stop";
                    return result;
                }
            }
        }

        // Pre-allocate zero visual tensors for decode steps
        ov::Tensor zero_ve, zero_pm;
        if (target_ir_is_vl_) {
            zero_ve = ov::Tensor(ov::element::f32, {1, block_size_, model_hidden_size_});
            std::memset(zero_ve.data(), 0, zero_ve.get_byte_size());
            zero_pm = ov::Tensor(ov::element::boolean, {1, block_size_});
            std::memset(zero_pm.data(), 0, zero_pm.get_byte_size());
        }

        // Pre-allocate reusable tensors
        ov::Tensor draft_ids(ov::element::i64, {1, block_size_});
        ov::Tensor reuse_draft_pos(ov::element::i64, {1, storage_len + block_size_});
        ov::Tensor reuse_verify_ids(ov::element::i64, {1, block_size_});
        ov::Tensor reuse_verify_mask(ov::element::i64, {1, max_length + block_size_});
        std::fill_n(reuse_verify_mask.data<int64_t>(),
                    static_cast<ptrdiff_t>(max_length + block_size_), 1LL);
        ov::Tensor reuse_verify_pos(ov::element::i64, {3, 1, block_size_});
        std::vector<int64_t> block_output_ids;
        block_output_ids.reserve(block_size_);

        int32_t pending_snapshot_commit = -1;
        bool stopped_by_eos = false;

        // ── Speculative decode loop ──
        auto gen_start = Clock::now();

        while (output_ids.size() < static_cast<size_t>(max_new_tokens)) {
            if (is_stop_token(next_token)) {
                stopped_by_eos = true;
                break;
            }

            // Build draft input: [last_token, MASK, MASK, ...]
            {
                auto* ids = draft_ids.data<int64_t>();
                ids[0] = output_ids.back();
                for (size_t i = 1; i < block_size_; ++i) ids[i] = mask_token_id_;
            }

            // Pre-prepare target verify tensors
            {
                reuse_verify_ids.set_shape({1, block_size_});
                reuse_verify_ids.data<int64_t>()[0] = output_ids.back();

                reuse_verify_mask.set_shape({1, target_hidden_len_ + block_size_});
                target_req_.set_tensor("attention_mask", reuse_verify_mask);

                reuse_verify_pos.set_shape({3, 1, block_size_});
                auto* pd = reuse_verify_pos.data<int64_t>();
                for (size_t dim = 0; dim < 3; ++dim)
                    for (size_t i = 0; i < block_size_; ++i)
                        pd[dim * block_size_ + i] = static_cast<int64_t>(target_hidden_len_ + i);
                target_req_.set_tensor("position_ids", reuse_verify_pos);
                target_req_.set_tensor("beam_idx", beam_idx_);

                if (target_ir_is_vl_) {
                    target_req_.set_tensor("visual_embeds", zero_ve);
                    target_req_.set_tensor("visual_pos_mask", zero_pm);
                }

                if (use_deferred_commit_ && pending_snapshot_commit >= 0) {
                    set_state_update_mode(-(pending_snapshot_commit + 1));
                    pending_snapshot_commit = -1;
                } else if (use_deferred_commit_) {
                    set_state_update_mode(0);
                } else {
                    set_state_update_mode(1);
                }
            }

            // Draft inference
            ov::Tensor draft_logits;
            {
                ov::Tensor ctx_view(ctx_hidden_storage_, {0, 0, 0},
                                    {1, ctx_hidden_len_, ctx_hidden_dim_});
                const size_t total_pos = ctx_hidden_len_ + block_size_;
                reuse_draft_pos.set_shape({1, total_pos});
                auto* pd = reuse_draft_pos.data<int64_t>();
                for (size_t i = 0; i < total_pos; ++i) pd[i] = static_cast<int64_t>(i);

                draft_req_.set_tensor("context_hidden", ctx_view);
                draft_req_.set_tensor("input_ids", draft_ids);
                draft_req_.set_tensor("position_ids", reuse_draft_pos);
                draft_req_.infer();
                draft_logits = draft_req_.get_tensor("logits");
            }

            // Argmax draft tokens (skip position 0)
            auto draft_tokens = detail::argmax_logits_slice(draft_logits, 1, block_size_ - 1);

            // Build verify block
            block_output_ids.clear();
            block_output_ids.push_back(output_ids.back());
            block_output_ids.insert(block_output_ids.end(),
                                    draft_tokens.begin(), draft_tokens.end());
            const size_t verify_len = block_output_ids.size();

            // Save linear states for fallback
            std::vector<std::pair<std::string, ov::Tensor>> saved_linear;
            if (!has_snapshots_) saved_linear = detail::save_linear_states(target_req_);

            // Verify
            std::memcpy(reuse_verify_ids.data<int64_t>(), block_output_ids.data(),
                        verify_len * sizeof(int64_t));
            target_req_.set_tensor("input_ids", reuse_verify_ids);
            target_req_.infer();

            logits = target_req_.get_tensor("logits");

            // Lazy argmax verification
            const size_t vocab = logits.get_shape()[2];
            size_t accepted = 0;
            int64_t posterior_next = 0;

            if (logits.get_element_type() == ov::element::f16) {
                const auto* ld = reinterpret_cast<const uint16_t*>(logits.data<const ov::float16>());
                for (size_t i = 0; i < draft_tokens.size(); ++i) {
                    int64_t tok = detail::argmax_row_f16_fast(ld + i * vocab, vocab);
                    if (tok != draft_tokens[i]) { posterior_next = tok; break; }
                    ++accepted;
                }
                if (accepted == draft_tokens.size())
                    posterior_next = detail::argmax_row_f16_fast(ld + accepted * vocab, vocab);
            } else {
                const auto* ld = logits.data<const float>();
                for (size_t i = 0; i < draft_tokens.size(); ++i) {
                    int64_t tok = detail::argmax_row(ld + i * vocab, vocab);
                    if (tok != draft_tokens[i]) { posterior_next = tok; break; }
                    ++accepted;
                }
                if (accepted == draft_tokens.size())
                    posterior_next = detail::argmax_row(ld + accepted * vocab, vocab);
            }

            const size_t num_accepted = accepted + 1;
            const bool all_accepted = (accepted == draft_tokens.size());
            const bool has_linear = has_snapshots_ || !saved_linear.empty();
            const bool use_snap = has_snapshots_ && has_linear && num_accepted > 0 &&
                                  (!all_accepted || use_deferred_commit_);

            // State update based on acceptance
            if (use_snap) {
                pending_snapshot_commit = static_cast<int32_t>(num_accepted - 1);
                size_t trim = verify_len - num_accepted;
                kv_state_.num_tokens_to_trim = trim;
                ov::genai::utils::trim_kv_cache(target_req_, kv_state_, std::nullopt);
                kv_state_.num_tokens_to_trim = 0;
                target_hidden_block = get_target_hidden();
            } else if (all_accepted) {
                target_hidden_block = get_target_hidden();
            } else if (!has_linear) {
                size_t trim = verify_len - num_accepted;
                kv_state_.num_tokens_to_trim = trim;
                ov::genai::utils::trim_kv_cache(target_req_, kv_state_, std::nullopt);
                kv_state_.num_tokens_to_trim = 0;
                target_hidden_block = get_target_hidden();
            } else {
                // Restore + replay
                detail::restore_linear_states(target_req_, saved_linear);
                kv_state_.num_tokens_to_trim = verify_len;
                ov::genai::utils::trim_kv_cache(target_req_, kv_state_, std::nullopt);
                kv_state_.num_tokens_to_trim = 0;

                reuse_verify_ids.set_shape({1, num_accepted});
                std::memcpy(reuse_verify_ids.data<int64_t>(), block_output_ids.data(),
                            num_accepted * sizeof(int64_t));
                target_req_.set_tensor("input_ids", reuse_verify_ids);
                reuse_verify_mask.set_shape({1, target_hidden_len_ + num_accepted});
                target_req_.set_tensor("attention_mask", reuse_verify_mask);
                reuse_verify_pos.set_shape({3, 1, num_accepted});
                auto* pd = reuse_verify_pos.data<int64_t>();
                for (size_t dim = 0; dim < 3; ++dim)
                    for (size_t i = 0; i < num_accepted; ++i)
                        pd[dim * num_accepted + i] = static_cast<int64_t>(target_hidden_len_ + i);
                target_req_.set_tensor("position_ids", reuse_verify_pos);
                target_req_.set_tensor("beam_idx", beam_idx_);
                if (target_ir_is_vl_) {
                    target_req_.set_tensor("visual_embeds", zero_ve);
                    target_req_.set_tensor("visual_pos_mask", zero_pm);
                }
                set_state_update_mode(1);
                target_req_.infer();
                target_hidden_block = get_target_hidden();
            }

            // Update target_hidden storage
            if (num_accepted > 0 && target_hidden_len_ + num_accepted <= storage_len) {
                ov::Tensor src(target_hidden_block, {0, 0, 0}, {1, num_accepted, hidden_dim_});
                ov::Tensor dst(target_hidden_storage_,
                               {0, target_hidden_len_, 0},
                               {1, target_hidden_len_ + num_accepted, hidden_dim_});
                std::memcpy(dst.data<float>(), src.data<const float>(),
                            num_accepted * hidden_dim_ * sizeof(float));

                // Async context_fc
                ov::Tensor new_th(target_hidden_storage_,
                                  {0, target_hidden_len_, 0},
                                  {1, target_hidden_len_ + num_accepted, hidden_dim_});
                ctx_fc_req_.set_tensor("target_hidden", new_th);
                ctx_fc_req_.start_async();
                target_hidden_len_ += num_accepted;
            }

            // Collect accepted tokens and stream them
            std::vector<int64_t> new_tokens;
            bool draft_hit_stop = false;
            for (size_t i = 0; i < accepted && output_ids.size() < static_cast<size_t>(max_new_tokens); ++i) {
                if (is_stop_token(draft_tokens[i])) {
                    draft_hit_stop = true;
                    break;
                }
                output_ids.push_back(draft_tokens[i]);
                new_tokens.push_back(draft_tokens[i]);
            }

            result.draft_steps++;
            result.accepted_tokens += accepted;

            // Wait for context_fc
            if (num_accepted > 0) {
                ctx_fc_req_.wait();
                auto new_ctx = ctx_fc_req_.get_tensor("context_hidden");
                ov::Tensor ctx_dst(ctx_hidden_storage_,
                                   {0, ctx_hidden_len_, 0},
                                   {1, ctx_hidden_len_ + num_accepted, ctx_hidden_dim_});
                copy_ctx_to_cache(new_ctx, ctx_dst);
                ctx_hidden_len_ += num_accepted;
            }

            // Stream accepted draft tokens
            if (!new_tokens.empty()) {
                auto text = strbuf.push(new_tokens);
                if (!text.empty()) {
                    if (!callback(text, false)) {
                        result.generated_tokens = output_ids.size();
                        result.finish_reason = "stop";
                        break;
                    }
                }
            }

            if (draft_hit_stop) {
                stopped_by_eos = true;
                break;
            }

            if (output_ids.size() >= static_cast<size_t>(max_new_tokens)) break;

            // Posterior token
            next_token = posterior_next;
            output_ids.push_back(next_token);

            if (is_stop_token(next_token)) {
                stopped_by_eos = true;
                auto text = strbuf.push(next_token);
                auto remaining = strbuf.flush();
                text += remaining;
                if (!text.empty()) callback(text, true);
                break;
            }

            // Stream posterior token
            {
                auto text = strbuf.push(next_token);
                if (!text.empty()) {
                    if (!callback(text, false)) {
                        result.generated_tokens = output_ids.size();
                        result.finish_reason = "stop";
                        break;
                    }
                }
            }
        }

        // Flush any remaining buffered text from the streaming buffer
        {
            auto remaining = strbuf.flush();
            if (!remaining.empty()) {
                callback(remaining, false);
            }
        }

        auto gen_end = Clock::now();
        result.generated_tokens = output_ids.size();
        result.decode_ms = duration_ms(gen_start, gen_end);
        result.throughput = result.decode_ms > 0
            ? (static_cast<double>(result.generated_tokens) * 1000.0 / result.decode_ms) : 0;
        result.finish_reason = stopped_by_eos ? "stop" : "length";
        return result;
    }

private:
    void reset_state() {
        target_req_.reset_state();
        target_req_.get_tensor("attention_mask").set_shape({1, 0});
        kv_state_ = {};
        // Re-detect kv axes (safe to repeat)
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(compiled_target_.get_runtime_model());
        kv_state_.seq_length_axis = kv_pos.seq_len;

        target_hidden_len_ = 0;
        ctx_hidden_len_ = 0;
        using_usm_storage_ = false;
        using_usm_output_ = false;
        using_usm_ctx_ = false;
    }

    void set_state_update_mode(int32_t mode) {
        if (!has_state_update_mode_) return;
        target_req_.set_tensor("state_update_mode", detail::make_state_update_mode_tensor(mode));
    }

    ov::Tensor get_target_hidden() {
        auto t = target_req_.get_tensor("target_hidden");
        if (using_usm_output_) return t;
        return detail::copy_to_host(t);
    }

    void copy_ctx_to_cache(const ov::Tensor& src, ov::Tensor& dst) {
        if (use_f16_ctx_) {
            const float* s = src.data<float>();
            ov::float16* d = dst.data<ov::float16>();
            const size_t n = src.get_size();
            for (size_t i = 0; i < n; ++i) d[i] = ov::float16(s[i]);
        } else {
            src.copy_to(dst);
        }
    }

    // Config
    std::filesystem::path target_dir_;
    std::filesystem::path draft_dir_;
    std::string device_;
    ov::genai::modeling::models::DFlashDraftConfig dflash_cfg_;
    size_t block_size_ = 16;
    size_t hidden_dim_ = 0;
    size_t model_hidden_size_ = 0;  // config hidden_size for visual_embeds
    size_t ctx_hidden_dim_ = 0;
    bool target_ir_is_vl_ = false;
    bool use_f16_ctx_ = false;

    // Compiled models
    ov::CompiledModel compiled_target_;
    ov::CompiledModel compiled_ctx_fc_;
    ov::CompiledModel compiled_draft_;

    // Infer requests
    ov::InferRequest target_req_;
    ov::InferRequest ctx_fc_req_;
    ov::InferRequest draft_req_;

    // State tracking
    ov::genai::utils::KVCacheState kv_state_;
    bool has_state_update_mode_ = false;
    bool has_snapshots_ = false;
    bool use_deferred_commit_ = false;

    // GPU / USM
    ov::RemoteContext remote_ctx_;
    bool has_gpu_ctx_ = false;
    bool using_usm_storage_ = false;
    bool using_usm_output_ = false;
    bool using_usm_ctx_ = false;

    // Storage tensors (allocated per-generate, freed on reset)
    ov::Tensor target_hidden_storage_;
    ov::Tensor ctx_hidden_storage_;
    size_t target_hidden_len_ = 0;
    size_t ctx_hidden_len_ = 0;

    // Tokenizer
    std::unique_ptr<ov::genai::Tokenizer> tokenizer_;
    int64_t mask_token_id_ = 0;
    int64_t eos_token_id_ = 0;
    std::set<int64_t> stop_token_ids_;

    bool is_stop_token(int64_t token) const {
        return token == eos_token_id_ || stop_token_ids_.count(token) > 0;
    }

    // Reusable tensors
    ov::Tensor beam_idx_;

public:
    /// Set additional stop token IDs (e.g. <|im_end|>).
    void set_stop_token_ids(const std::set<int64_t>& ids) { stop_token_ids_ = ids; }
};

}  // namespace dflash
