// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/session.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include <openvino/openvino.hpp>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "modeling/api/sampler.hpp"
#include "modeling/api/types.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "sampling/logit_processor.hpp"

namespace ov::genai::modeling {

namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/// Create a USM-host tensor via GPU context, or fall back to regular tensor.
ov::Tensor make_usm_host_tensor(std::optional<ov::RemoteContext>& ctx,
                                const ov::element::Type& type,
                                const ov::Shape& shape) {
    if (ctx.has_value()) {
        try {
            return ctx->create_host_tensor(type, shape);
        } catch (...) {}
    }
    return ov::Tensor(type, shape);
}

/// Clone a tensor as USM-host for iGPU zero-copy.
ov::Tensor clone_as_usm_host(std::optional<ov::RemoteContext>& ctx, const ov::Tensor& src) {
    auto dst = make_usm_host_tensor(ctx, src.get_element_type(), src.get_shape());
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
}

/// Try to get the GPU RemoteContext from a CompiledModel.
std::optional<ov::RemoteContext> try_get_gpu_context(ov::CompiledModel& compiled) {
    try {
        return compiled.get_context();
    } catch (...) {
        return std::nullopt;
    }
}

/// Build VL prompt string with image token placeholders.
std::string build_vl_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

}  // namespace

// ─── Impl ───

struct Session::Impl {
    ModelLoader& model_;

    // Text inference
    ov::InferRequest text_request_;
    std::optional<ov::RemoteContext> gpu_ctx_;

    // Pre-allocated decode tensors (USM-host for zero-copy)
    ov::Tensor step_ids_;           // [B, 1]
    ov::Tensor step_mask_;          // [B, 1]
    ov::Tensor decode_pos_;         // [3, B, 1]
    ov::Tensor beam_idx_;           // [B]

    // VL decode placeholders
    ov::Tensor decode_visual_;      // [B, 1, hidden_size] zeros
    ov::Tensor decode_visual_mask_; // [B, 1] zeros

    // State tracking
    int64_t past_len_ = 0;
    ov::Tensor rope_deltas_;        // [B, 1] from InputPlanner
    std::vector<int64_t> generated_ids_;

    // Sampling scratch
    std::vector<float> logit_buf_;
    SamplingContext sampling_ctx_;

    // Control
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> is_generating_{false};
    std::mutex generate_mutex_;

    // Vision (stateless, created once)
    std::optional<ov::InferRequest> vision_request_;

    static constexpr size_t kBatch = 1;

    explicit Impl(ModelLoader& model) : model_(model) {
        // Create text InferRequest
        text_request_ = model_.compiled_text().create_infer_request();
        gpu_ctx_ = try_get_gpu_context(model_.compiled_text());

        // Pre-allocate decode tensors
        step_ids_ = make_usm_host_tensor(gpu_ctx_, ov::element::i64, {kBatch, 1});
        step_mask_ = make_usm_host_tensor(gpu_ctx_, ov::element::i64, {kBatch, 1});
        step_mask_.data<int64_t>()[0] = 1;
        decode_pos_ = make_usm_host_tensor(gpu_ctx_, ov::element::i64, {3, kBatch, 1});

        beam_idx_ = make_usm_host_tensor(gpu_ctx_, ov::element::i32, {kBatch});
        beam_idx_.data<int32_t>()[0] = 0;

        // VL decode placeholders
        if (model_.compiled_vision()) {
            const auto hidden = static_cast<size_t>(model_.config().text.hidden_size);
            decode_visual_ = make_usm_host_tensor(gpu_ctx_, ov::element::f32, {kBatch, 1, hidden});
            std::memset(decode_visual_.data(), 0, decode_visual_.get_byte_size());
            decode_visual_mask_ = make_usm_host_tensor(gpu_ctx_, ov::element::boolean, {kBatch, 1});
            std::memset(decode_visual_mask_.data(), 0, decode_visual_mask_.get_byte_size());

            vision_request_ = model_.compiled_vision()->create_infer_request();
        }
    }

    /// Compute past_len from attention mask (counts active tokens).
    int64_t compute_past_len(const ov::Tensor& attention_mask) {
        if (attention_mask.get_element_type() != ov::element::i64) {
            return static_cast<int64_t>(attention_mask.get_shape().at(1));
        }
        const int64_t* mask = attention_mask.data<const int64_t>();
        const size_t seq = attention_mask.get_shape().at(1);
        int64_t active = 0;
        for (size_t s = 0; s < seq; ++s) {
            if (mask[s] != 0) active++;
        }
        return active;
    }

    /// Run vision encoder to produce visual embeddings.
    std::pair<ov::Tensor, ov::Tensor> encode_vision(const ov::Tensor& image) {
        if (!vision_request_.has_value()) {
            throw std::runtime_error("Vision model not available");
        }

        const auto& cfg = model_.config();
        models::Qwen3_5VisionPreprocessor preprocessor(cfg.vision, model_.preprocess_config());
        auto inputs = preprocessor.preprocess(image, model_.pos_embed_weight());

        auto& req = *vision_request_;
        req.set_tensor(models::Qwen3_5VisionIO::kPixelValues, inputs.pixel_values);
        req.set_tensor(models::Qwen3_5VisionIO::kGridThw, inputs.grid_thw);
        req.set_tensor(models::Qwen3_5VisionIO::kPosEmbeds, inputs.pos_embeds);
        req.set_tensor(models::Qwen3_5VisionIO::kRotaryCos, inputs.rotary_cos);
        req.set_tensor(models::Qwen3_5VisionIO::kRotarySin, inputs.rotary_sin);
        req.infer();

        ov::Tensor visual_embeds = req.get_tensor(models::Qwen3_5VisionIO::kVisualEmbeds);
        return {visual_embeds, inputs.grid_thw};
    }

    /// Tokenize a text prompt using chat template.
    std::pair<ov::Tensor, ov::Tensor> tokenize_text(const std::string& prompt,
                                                     bool enable_thinking) {
        auto* tok = model_.tokenizer();
        if (!tok) {
            throw std::runtime_error("Tokenizer not available");
        }

        std::string final_prompt = prompt;
        bool add_special = true;

        if (!tok->get_chat_template().empty()) {
            ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            ov::genai::JsonContainer extra({{"enable_thinking", enable_thinking}});
            final_prompt = tok->apply_chat_template(history, true, {}, std::nullopt, extra);
            add_special = false;
        }

        auto result = tok->encode(final_prompt, ov::genai::add_special_tokens(add_special));
        return {result.input_ids, result.attention_mask};
    }

    /// Tokenize a VL prompt with image token placeholders.
    std::pair<ov::Tensor, ov::Tensor> tokenize_vl(const std::string& prompt,
                                                    const ov::Tensor& grid_thw) {
        auto* tok = model_.tokenizer();
        if (!tok) {
            throw std::runtime_error("Tokenizer not available");
        }

        const auto& cfg = model_.config();
        const int64_t image_tokens = models::Qwen3_5VisionPreprocessor::count_visual_tokens(
            grid_thw, cfg.vision.spatial_merge_size);

        std::string vl_prompt = build_vl_prompt(prompt, image_tokens);
        auto result = tok->encode(vl_prompt, ov::genai::add_special_tokens(false));
        return {result.input_ids, result.attention_mask};
    }

    /// Core generation loop: prefill + decode.
    GenerateResult run_generate(const ov::Tensor& input_ids,
                                const ov::Tensor& attention_mask,
                                const ov::Tensor& position_ids,
                                const ov::Tensor* visual_embeds,
                                const ov::Tensor* visual_pos_mask,
                                const ov::Tensor& rope_deltas,
                                const GenerateParams& params,
                                StreamCallback callback) {
        const bool use_vl = (visual_embeds != nullptr);
        const bool use_sampling = params.sampling.temperature > 0.0f;
        const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

        // Collect prompt token IDs for LogitProcessor
        std::vector<int64_t> prompt_token_ids(
            input_ids.data<const int64_t>(),
            input_ids.data<const int64_t>() + input_ids.get_size());

        // Build penalty-only LogitProcessor
        ov::genai::GenerationConfig penalty_config;
        penalty_config.do_sample = false;
        penalty_config.repetition_penalty = params.sampling.repetition_penalty;
        penalty_config.frequency_penalty = params.sampling.frequency_penalty;
        penalty_config.presence_penalty = params.sampling.presence_penalty;
        ov::genai::LogitProcessor penalty_processor(penalty_config, prompt_token_ids);

        // RNG for multinomial sampling
        std::mt19937 rng(params.sampling.rng_seed != 0
                         ? static_cast<std::mt19937::result_type>(params.sampling.rng_seed)
                         : std::random_device{}());

        // ─── Prefill ───
        auto usm_ids = clone_as_usm_host(gpu_ctx_, input_ids);
        auto usm_mask = clone_as_usm_host(gpu_ctx_, attention_mask);
        auto usm_pos = clone_as_usm_host(gpu_ctx_, position_ids);

        text_request_.reset_state();
        text_request_.set_tensor(models::Qwen3_5TextIO::kInputIds, usm_ids);
        text_request_.set_tensor(models::Qwen3_5TextIO::kAttentionMask, usm_mask);
        text_request_.set_tensor(models::Qwen3_5TextIO::kPositionIds, usm_pos);
        text_request_.set_tensor(models::Qwen3_5TextIO::kBeamIdx, beam_idx_);

        if (use_vl) {
            auto usm_vis = clone_as_usm_host(gpu_ctx_, *visual_embeds);
            auto usm_vis_mask = clone_as_usm_host(gpu_ctx_, *visual_pos_mask);
            text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, usm_vis);
            text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, usm_vis_mask);
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        text_request_.infer();
        const auto prefill_end = std::chrono::steady_clock::now();

        ov::Tensor logits = text_request_.get_tensor(models::Qwen3_5TextIO::kLogits);
        extract_last_logits_f32(logits, logit_buf_);

        int64_t next_id;
        {
            ov::genai::Logits lw(logit_buf_.data(), logit_buf_.size());
            penalty_processor.apply(lw);
            next_id = use_sampling
                          ? sample_fast(logit_buf_.data(), logit_buf_.size(),
                                        params.sampling.temperature, params.sampling.top_p,
                                        params.sampling.top_k, rng, sampling_ctx_)
                          : argmax_f32(logit_buf_);
        }
        penalty_processor.register_new_generated_token(next_id);
        penalty_processor.update_generated_len(1);

        std::vector<int64_t> generated;
        generated.reserve(static_cast<size_t>(params.max_new_tokens));
        generated.push_back(next_id);

        const auto& stop_ids = model_.stop_token_ids();
        past_len_ = compute_past_len(attention_mask);
        rope_deltas_ = rope_deltas;
        const int64_t* rope_data = rope_deltas_.data<const int64_t>();

        // Notify PREFILL_DONE
        if (callback) {
            StreamChunk chunk;
            chunk.event = StreamEvent::PREFILL_DONE;
            chunk.prompt_tokens = static_cast<int>(prompt_len);
            chunk.prefill_ms = elapsed_ms(prefill_start, prefill_end);
            if (!callback(chunk)) {
                stop_requested_.store(true);
            }
        }

        // Notify first token
        if (callback && !stop_requested_.load()) {
            StreamChunk chunk;
            chunk.event = StreamEvent::TOKEN;
            chunk.token_id = next_id;
            if (model_.tokenizer()) {
                chunk.token_text = model_.tokenizer()->decode({next_id}, ov::genai::skip_special_tokens(true));
            }
            if (!callback(chunk)) {
                stop_requested_.store(true);
            }
        }

        // ─── Decode loop ───
        size_t decode_steps = 0;
        const auto decode_start = std::chrono::steady_clock::now();

        for (int step = 1; step < params.max_new_tokens && !stop_requested_.load(); ++step) {
            if (!stop_ids.empty() && stop_ids.count(next_id) > 0) {
                break;
            }

            // Fill step inputs
            step_ids_.data<int64_t>()[0] = next_id;

            auto* pos_data = decode_pos_.data<int64_t>();
            const int64_t value = past_len_ + rope_data[0];
            pos_data[0] = value;
            pos_data[kBatch] = value;
            pos_data[2 * kBatch] = value;

            text_request_.set_tensor(models::Qwen3_5TextIO::kInputIds, step_ids_);
            text_request_.set_tensor(models::Qwen3_5TextIO::kAttentionMask, step_mask_);
            text_request_.set_tensor(models::Qwen3_5TextIO::kPositionIds, decode_pos_);
            text_request_.set_tensor(models::Qwen3_5TextIO::kBeamIdx, beam_idx_);

            if (use_vl) {
                text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, decode_visual_);
                text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, decode_visual_mask_);
            }

            text_request_.infer();
            logits = text_request_.get_tensor(models::Qwen3_5TextIO::kLogits);
            extract_last_logits_f32(logits, logit_buf_);

            {
                ov::genai::Logits lw(logit_buf_.data(), logit_buf_.size());
                penalty_processor.apply(lw);
                next_id = use_sampling
                              ? sample_fast(logit_buf_.data(), logit_buf_.size(),
                                            params.sampling.temperature, params.sampling.top_p,
                                            params.sampling.top_k, rng, sampling_ctx_)
                              : argmax_f32(logit_buf_);
            }
            penalty_processor.register_new_generated_token(next_id);
            generated.push_back(next_id);
            penalty_processor.update_generated_len(generated.size());
            decode_steps += 1;
            past_len_ += 1;

            // Stream token
            if (callback) {
                StreamChunk chunk;
                chunk.event = StreamEvent::TOKEN;
                chunk.token_id = next_id;
                if (model_.tokenizer()) {
                    chunk.token_text = model_.tokenizer()->decode({next_id}, ov::genai::skip_special_tokens(true));
                }
                if (!callback(chunk)) {
                    stop_requested_.store(true);
                }
            }
        }
        const auto decode_end = std::chrono::steady_clock::now();

        // ─── Build result ───
        GenerateResult result;
        result.token_ids = std::move(generated);
        result.prompt_tokens = static_cast<int>(prompt_len);
        result.generated_tokens = static_cast<int>(result.token_ids.size());
        result.prefill_ms = elapsed_ms(prefill_start, prefill_end);
        result.decode_ms = elapsed_ms(decode_start, decode_end);
        result.ttft_ms = result.prefill_ms;
        result.throughput = decode_steps > 0 && result.decode_ms > 0.0
                                ? (static_cast<double>(decode_steps) * 1000.0 / result.decode_ms)
                                : 0.0;

        if (stop_requested_.load()) {
            result.stop_reason = StopReason::USER_STOP;
        } else if (static_cast<int>(result.token_ids.size()) >= params.max_new_tokens) {
            result.stop_reason = StopReason::MAX_TOKENS;
        } else {
            result.stop_reason = StopReason::EOS;
        }

        // Decode text
        if (model_.tokenizer()) {
            result.text = model_.tokenizer()->decode(result.token_ids, ov::genai::skip_special_tokens(true));
        }

        // Notify FINISH
        if (callback) {
            StreamChunk chunk;
            chunk.event = StreamEvent::FINISH;
            chunk.stop_reason = result.stop_reason;
            chunk.prompt_tokens = result.prompt_tokens;
            chunk.generated_tokens = result.generated_tokens;
            chunk.prefill_ms = result.prefill_ms;
            chunk.decode_ms = result.decode_ms;
            chunk.ttft_ms = result.ttft_ms;
            chunk.throughput = result.throughput;
            callback(chunk);  // FINISH callback return value ignored
        }

        return result;
    }
};

// ─── Public interface ───

Session::Session(ModelLoader& model)
    : impl_(std::make_unique<Impl>(model)) {}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

GenerateResult Session::generate(const std::string& prompt,
                                  const GenerateParams& params,
                                  StreamCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->generate_mutex_);
    impl_->is_generating_.store(true);
    impl_->stop_requested_.store(false);

    try {
        // Tokenize
        auto [input_ids, attention_mask] = impl_->tokenize_text(prompt, params.enable_thinking);

        // Plan inputs
        const auto& cfg = impl_->model_.config();
        models::Qwen3_5InputPlanner planner(cfg);
        auto plan = planner.build_plan(input_ids, &attention_mask, nullptr);

        // Run
        auto result = impl_->run_generate(input_ids, attention_mask, plan.position_ids,
                                           nullptr, nullptr, plan.rope_deltas,
                                           params, std::move(callback));
        impl_->is_generating_.store(false);
        return result;
    } catch (...) {
        impl_->is_generating_.store(false);
        throw;
    }
}

GenerateResult Session::generate_vl(const std::string& prompt,
                                     const ov::Tensor& image,
                                     const GenerateParams& params,
                                     StreamCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->generate_mutex_);
    impl_->is_generating_.store(true);
    impl_->stop_requested_.store(false);

    try {
        // Vision encode
        auto [visual_embeds, grid_thw] = impl_->encode_vision(image);

        // Tokenize VL prompt
        auto [input_ids, attention_mask] = impl_->tokenize_vl(prompt, grid_thw);

        // Plan with VL
        const auto& cfg = impl_->model_.config();
        models::Qwen3_5InputPlanner planner(cfg);
        auto plan = planner.build_plan(input_ids, &attention_mask, &grid_thw);

        // Scatter visual embeddings
        auto visual_padded = models::Qwen3_5InputPlanner::scatter_visual_embeds(
            visual_embeds, plan.visual_pos_mask);

        // Run
        auto result = impl_->run_generate(input_ids, attention_mask, plan.position_ids,
                                           &visual_padded, &plan.visual_pos_mask,
                                           plan.rope_deltas, params, std::move(callback));
        impl_->is_generating_.store(false);
        return result;
    } catch (...) {
        impl_->is_generating_.store(false);
        throw;
    }
}

void Session::stop() {
    impl_->stop_requested_.store(true);
}

void Session::reset() {
    impl_->text_request_.reset_state();
    impl_->past_len_ = 0;
    impl_->generated_ids_.clear();
}

bool Session::is_generating() const {
    return impl_->is_generating_.load();
}

}  // namespace ov::genai::modeling
