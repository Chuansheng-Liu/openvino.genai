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
#include "modeling/api/thinking_tracker.hpp"
#include "modeling/api/token_processor.hpp"
#include "modeling/api/tool_call_parser.hpp"
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

    // Streaming pipeline components
    std::optional<TokenProcessor> token_processor_;
    ThinkingTracker thinking_tracker_;
    ToolCallParser tool_call_parser_;

    static constexpr size_t kBatch = 1;

    /// Create/recreate InferRequest and all GPU-context-backed tensors.
    void init_request_and_tensors() {
        text_request_ = model_.compiled_text().create_infer_request();
        gpu_ctx_ = try_get_gpu_context(model_.compiled_text());

        // Pre-allocate decode tensors (USM-host for zero-copy on iGPU)
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

        past_len_ = 0;
        generated_ids_.clear();
    }

    explicit Impl(ModelLoader& model) : model_(model) {
        init_request_and_tensors();

        // Initialize token processor if tokenizer available
        if (model_.tokenizer()) {
            token_processor_.emplace(*model_.tokenizer());
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

    /// Tokenize a text prompt using chat template (or raw if pre-formatted).
    std::pair<ov::Tensor, ov::Tensor> tokenize_text(const std::string& prompt,
                                                     bool enable_thinking,
                                                     bool raw_prompt = false) {
        auto* tok = model_.tokenizer();
        if (!tok) {
            throw std::runtime_error("Tokenizer not available");
        }

        std::string final_prompt = prompt;
        bool add_special = true;

        if (!raw_prompt && !tok->get_chat_template().empty()) {
            ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            ov::genai::JsonContainer extra({{"enable_thinking", enable_thinking}});
            final_prompt = tok->apply_chat_template(history, true, {}, std::nullopt, extra);
            add_special = false;
        } else if (raw_prompt) {
            // Prompt is already fully formatted (ChatML etc.), just tokenize
            add_special = false;
        }

        auto result = tok->encode(final_prompt, ov::genai::add_special_tokens(add_special));
        return {result.input_ids, result.attention_mask};
    }

    /// Tokenize a VL prompt with image token placeholders.
    /// When raw_prompt=true, the prompt is already ChatML-formatted and contains
    /// <|vision_start|><|vision_end|> as a marker; we expand it with image_pad tokens.
    /// When raw_prompt=false, build_vl_prompt wraps a plain user message.
    std::pair<ov::Tensor, ov::Tensor> tokenize_vl(const std::string& prompt,
                                                    const ov::Tensor& grid_thw,
                                                    bool raw_prompt = false) {
        auto* tok = model_.tokenizer();
        if (!tok) {
            throw std::runtime_error("Tokenizer not available");
        }

        const auto& cfg = model_.config();
        const int64_t image_tokens = models::Qwen3_5VisionPreprocessor::count_visual_tokens(
            grid_thw, cfg.vision.spatial_merge_size);

        std::string vl_prompt;
        if (raw_prompt) {
            // Expand <|vision_start|><|vision_end|> marker with image_pad tokens
            vl_prompt = prompt;
            const std::string marker = "<|vision_start|><|vision_end|>";
            auto pos = vl_prompt.find(marker);
            if (pos != std::string::npos) {
                std::string expansion = "<|vision_start|>";
                expansion.reserve(expansion.size() +
                                  static_cast<size_t>(image_tokens) * 13 + 16);
                for (int64_t i = 0; i < image_tokens; ++i) {
                    expansion += "<|image_pad|>";
                }
                expansion += "<|vision_end|>";
                vl_prompt.replace(pos, marker.size(), expansion);
            }
        } else {
            vl_prompt = build_vl_prompt(prompt, image_tokens);
        }
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
        } else if (model_.compiled_vision()) {
            // VL model loaded but text-only request: provide zero visual tensors
            // matching prompt sequence length so Select nodes don't get shape mismatch.
            const auto hidden = static_cast<size_t>(model_.config().text.hidden_size);
            const auto seq = static_cast<size_t>(prompt_len);
            auto zero_vis = make_usm_host_tensor(gpu_ctx_, ov::element::f32,
                                                  {kBatch, seq, hidden});
            std::memset(zero_vis.data(), 0, zero_vis.get_byte_size());
            auto zero_mask = make_usm_host_tensor(gpu_ctx_, ov::element::boolean,
                                                   {kBatch, seq});
            std::memset(zero_mask.data(), 0, zero_mask.get_byte_size());
            text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, zero_vis);
            text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, zero_mask);
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

        // Reset streaming pipeline for this generation
        if (token_processor_) token_processor_->reset();
        thinking_tracker_.reset();
        tool_call_parser_.reset();

        // When enable_thinking=true, the prompt already contains <think> (either
        // from chat template or raw prompt), so the model output starts inside the
        // thinking block. Initialize tracker accordingly.
        if (params.enable_thinking) {
            thinking_tracker_.start_in_thinking();
        }

        // Accumulated text for the final result
        std::string accumulated_thinking;
        std::string accumulated_content;
        int thinking_tokens = 0;
        bool sent_thinking_start = false;

        // Helper: run text through thinking tracker + tool parser + callback
        auto emit_text = [&](const std::string& text, int64_t token_id) -> bool {
            if (text.empty() || !callback) return true;

            // Step 1: Thinking tracker splits into thinking vs content
            auto tr = thinking_tracker_.process(text);

            // Send THINKING_START event on first entry
            if (thinking_tracker_.is_thinking() && !sent_thinking_start) {
                StreamChunk start_chunk;
                start_chunk.event = StreamEvent::THINKING_START;
                if (!callback(start_chunk)) return false;
                sent_thinking_start = true;
            }

            // Emit thinking text
            if (!tr.thinking_text.empty()) {
                accumulated_thinking += tr.thinking_text;
                StreamChunk chunk;
                chunk.event = StreamEvent::TOKEN;
                chunk.token_id = token_id;
                chunk.token_text = tr.thinking_text;
                chunk.is_thinking = true;
                if (!callback(chunk)) return false;
            }

            // Send THINKING_END when tracker transitions out
            if (sent_thinking_start &&
                thinking_tracker_.state() == ThinkingState::AFTER_THINKING &&
                !tr.content_text.empty()) {
                StreamChunk end_chunk;
                end_chunk.event = StreamEvent::THINKING_END;
                if (!callback(end_chunk)) return false;
            }

            // Step 2: Tool call parser on content text
            if (!tr.content_text.empty()) {
                auto tp = tool_call_parser_.process(tr.content_text);
                if (!tp.text.empty()) {
                    accumulated_content += tp.text;
                    StreamChunk chunk;
                    chunk.event = StreamEvent::TOKEN;
                    chunk.token_id = token_id;
                    chunk.token_text = tp.text;
                    chunk.is_thinking = false;
                    if (!callback(chunk)) return false;
                }
                // Tool calls are accumulated and reported in GenerateResult
            }

            return true;
        };

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

        // Process first token through streaming pipeline
        if (!stop_requested_.load()) {
            if (token_processor_) {
                std::string delta = token_processor_->process(next_id);
                if (!emit_text(delta, next_id)) {
                    stop_requested_.store(true);
                }
            } else if (callback) {
                // No tokenizer — emit token ID only
                StreamChunk chunk;
                chunk.event = StreamEvent::TOKEN;
                chunk.token_id = next_id;
                if (!callback(chunk)) {
                    stop_requested_.store(true);
                }
            }
        }

        // ─── Decode loop ───
        size_t decode_steps = 0;
        size_t tokens_since_punct = 0;
        size_t consecutive_symbol_tokens = 0;  // emoji/symbol flood detection
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
            } else if (model_.compiled_vision()) {
                // VL model, text-only: use pre-allocated [B, 1] zero tensors
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

            // Detect degenerate output and force stop:
            // 1) N consecutive identical tokens (e.g. "!!!!!!!")
            // 2) Low token diversity in sliding window (e.g. nonsense word salad)
            // 3) Too many tokens without sentence-ending punctuation (word list degeneration)
            // 4) Symbol/emoji flood (e.g. "🚩🔴🟠⛈️❄️➤♣♥♦♂♀" or "ΩΔΣΠΛΞ")
            {
                constexpr size_t kMaxRepeatTokens = 5;
                constexpr size_t kDiversityWindow = 40;
                constexpr size_t kMinUniqueTokens = 10;

                const size_t n = generated.size();

                // Check consecutive identical tokens
                if (n >= kMaxRepeatTokens) {
                    bool all_same = true;
                    for (size_t i = n - kMaxRepeatTokens; i < n - 1; ++i) {
                        if (generated[i] != generated[i + 1]) { all_same = false; break; }
                    }
                    if (all_same) stop_requested_.store(true);
                }

                // Check sliding window diversity
                if (n >= kDiversityWindow) {
                    std::set<int64_t> unique_in_window(
                        generated.end() - kDiversityWindow, generated.end());
                    if (unique_in_window.size() < kMinUniqueTokens)
                        stop_requested_.store(true);
                }
            }

            // Count thinking tokens
            if (thinking_tracker_.is_thinking()) {
                thinking_tokens++;
            }

            // Stream token through pipeline + check for punctuation absence
            if (token_processor_) {
                std::string delta = token_processor_->process(next_id);

                // Track tokens since last sentence-ending punctuation.
                // Normal text has punctuation every ~20-50 tokens; word salad has none.
                bool has_punct = false;
                for (unsigned char c : delta) {
                    // ASCII punctuation
                    if (c == '.' || c == '!' || c == '?' || c == '\n') { has_punct = true; break; }
                }
                // Check for CJK punctuation (UTF-8 encoded)
                if (!has_punct) {
                    // 。= E3 80 82, ！= EF BC 81, ？= EF BC 9F, ，= EF BC 8C
                    for (size_t i = 0; i + 2 < delta.size(); ++i) {
                        auto b0 = static_cast<unsigned char>(delta[i]);
                        auto b1 = static_cast<unsigned char>(delta[i+1]);
                        auto b2 = static_cast<unsigned char>(delta[i+2]);
                        if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) { has_punct = true; break; } // 。
                        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x81) { has_punct = true; break; } // ！
                        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F) { has_punct = true; break; } // ？
                        if (b0 == 0xEF && b1 == 0xBC && b2 == 0x8C) { has_punct = true; break; } // ，
                    }
                }
                if (has_punct) {
                    tokens_since_punct = 0;
                } else if (!delta.empty()) {
                    tokens_since_punct++;
                }
                constexpr size_t kMaxTokensWithoutPunct = 50;
                if (tokens_since_punct > kMaxTokensWithoutPunct && generated.size() > 100) {
                    stop_requested_.store(true);
                }

                // 4) Symbol/emoji flood detection: if most chars in the token are
                //    non-text (not CJK, not Latin letters/digits, not common punct),
                //    increment counter. Normal text resets it.
                {
                    size_t text_chars = 0, total_chars = 0;
                    for (size_t ci = 0; ci < delta.size(); ) {
                        auto b0 = static_cast<unsigned char>(delta[ci]);
                        size_t char_len = 1;
                        if (b0 < 0x80) {
                            char_len = 1;
                            // ASCII letters, digits, common punct, space
                            if ((b0 >= 'A' && b0 <= 'Z') || (b0 >= 'a' && b0 <= 'z') ||
                                (b0 >= '0' && b0 <= '9') || b0 == ' ' || b0 == '\n' ||
                                b0 == '.' || b0 == ',' || b0 == '!' || b0 == '?' ||
                                b0 == ':' || b0 == ';' || b0 == '-' || b0 == '\'') {
                                text_chars++;
                            }
                        } else if (b0 < 0xE0) {
                            char_len = 2;
                        } else if (b0 < 0xF0) {
                            char_len = 3;
                            // CJK Unified Ideographs (U+4E00-U+9FFF) = E4 B8 80 - E9 BF BF
                            // CJK common punctuation (U+3000-U+303F) = E3 80 80 - E3 80 BF
                            // CJK fullwidth forms (U+FF00-U+FF5E) = EF BC 80 - EF BD 9E
                            if (ci + 2 < delta.size()) {
                                auto b1 = static_cast<unsigned char>(delta[ci+1]);
                                auto b2 = static_cast<unsigned char>(delta[ci+2]);
                                uint32_t cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                                if ((cp >= 0x4E00 && cp <= 0x9FFF) ||  // CJK ideographs
                                    (cp >= 0x3400 && cp <= 0x4DBF) ||  // CJK ext A
                                    (cp >= 0x3000 && cp <= 0x303F) ||  // CJK symbols & punct
                                    (cp >= 0xFF01 && cp <= 0xFF5E)) {  // fullwidth forms
                                    text_chars++;
                                }
                            }
                        } else {
                            char_len = 4;  // emoji, supplementary — treated as non-text
                        }
                        total_chars++;
                        ci += char_len;
                    }
                    // If more than half of characters are non-text symbols
                    bool is_symbol_token = (total_chars > 0 && text_chars * 2 < total_chars);
                    if (is_symbol_token) {
                        consecutive_symbol_tokens++;
                    } else {
                        consecutive_symbol_tokens = 0;
                    }
                    constexpr size_t kMaxConsecutiveSymbolTokens = 10;
                    if (consecutive_symbol_tokens >= kMaxConsecutiveSymbolTokens && generated.size() > 50) {
                        stop_requested_.store(true);
                    }
                }

                if (!emit_text(delta, next_id)) {
                    stop_requested_.store(true);
                }
            } else if (callback) {
                StreamChunk chunk;
                chunk.event = StreamEvent::TOKEN;
                chunk.token_id = next_id;
                if (!callback(chunk)) {
                    stop_requested_.store(true);
                }
            }
        }
        const auto decode_end = std::chrono::steady_clock::now();

        // Flush remaining buffered text
        if (token_processor_) {
            std::string remaining = token_processor_->flush();
            if (!remaining.empty()) {
                emit_text(remaining, -1);
            }
        }

        // ─── Build result ───
        GenerateResult result;
        result.token_ids = std::move(generated);
        result.prompt_tokens = static_cast<int>(prompt_len);
        result.generated_tokens = static_cast<int>(result.token_ids.size());
        result.thinking_tokens = thinking_tokens;
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

        // Build text results from accumulated streams
        result.thinking_text = accumulated_thinking;
        result.text = accumulated_content;

        // If no streaming was active, decode all at once
        if (result.text.empty() && result.thinking_text.empty() && model_.tokenizer()) {
            std::string full_text = model_.tokenizer()->decode(
                result.token_ids, ov::genai::skip_special_tokens(true));
            // If thinking tracker was used, split the text
            if (params.enable_thinking) {
                ThinkingTracker final_tracker;
                // Model output starts inside thinking block (prompt has <think>)
                final_tracker.start_in_thinking();
                auto tr = final_tracker.process(full_text);
                result.thinking_text = tr.thinking_text;
                result.text = tr.content_text;
            } else {
                result.text = full_text;
            }
        }

        // Collect tool calls from parser
        auto tool_flush = tool_call_parser_.flush();
        // (tool calls already in result via accumulated_content)

        // Notify FINISH
        if (callback) {
            StreamChunk chunk;
            chunk.event = StreamEvent::FINISH;
            chunk.stop_reason = result.stop_reason;
            chunk.prompt_tokens = result.prompt_tokens;
            chunk.generated_tokens = result.generated_tokens;
            chunk.thinking_tokens = result.thinking_tokens;
            chunk.prefill_ms = result.prefill_ms;
            chunk.decode_ms = result.decode_ms;
            chunk.ttft_ms = result.ttft_ms;
            chunk.throughput = result.throughput;
            callback(chunk);
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
        auto [input_ids, attention_mask] = impl_->tokenize_text(
            prompt, params.enable_thinking, params.raw_prompt);

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
        auto [input_ids, attention_mask] = impl_->tokenize_vl(prompt, grid_thw,
                                                               params.raw_prompt);

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

void Session::warmup(int max_seq_len) {
    if (max_seq_len <= 0) return;

    const auto& cfg = impl_->model_.config();
    const size_t seq = static_cast<size_t>(max_seq_len);
    constexpr size_t B = 1;
    constexpr int kDecodeSteps = 16;

    auto run_text_warmup = [&](const ov::Tensor* visual_embeds,
                               const ov::Tensor* visual_pos_mask,
                               const ov::Tensor* grid_thw) {
        // ── Build dummy prefill inputs ──
        ov::Tensor input_ids(ov::element::i64, {B, seq});
        std::fill_n(input_ids.data<int64_t>(), seq, int64_t{1});

        ov::Tensor attention_mask(ov::element::i64, {B, seq});
        std::fill_n(attention_mask.data<int64_t>(), seq, int64_t{1});

        models::Qwen3_5InputPlanner planner(cfg);
        auto plan = planner.build_plan(input_ids, &attention_mask, grid_thw);

        auto usm_ids = clone_as_usm_host(impl_->gpu_ctx_, input_ids);
        auto usm_mask = clone_as_usm_host(impl_->gpu_ctx_, attention_mask);
        auto usm_pos = clone_as_usm_host(impl_->gpu_ctx_, plan.position_ids);

        // ── Prefill ──
        impl_->text_request_.reset_state();
        impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kInputIds, usm_ids);
        impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kAttentionMask, usm_mask);
        impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kPositionIds, usm_pos);
        impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kBeamIdx, impl_->beam_idx_);

        if (visual_embeds && visual_pos_mask) {
            // VL path: use real visual embeddings
            auto usm_vis = clone_as_usm_host(impl_->gpu_ctx_, *visual_embeds);
            auto usm_vis_mask = clone_as_usm_host(impl_->gpu_ctx_, *visual_pos_mask);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, usm_vis);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, usm_vis_mask);
        } else if (impl_->model_.compiled_vision()) {
            // Text-only request on VL model: zero visual tensors
            const auto hidden = static_cast<size_t>(cfg.text.hidden_size);
            auto zero_vis = make_usm_host_tensor(impl_->gpu_ctx_, ov::element::f32, {B, seq, hidden});
            std::memset(zero_vis.data(), 0, zero_vis.get_byte_size());
            auto zero_mask = make_usm_host_tensor(impl_->gpu_ctx_, ov::element::boolean, {B, seq});
            std::memset(zero_mask.data(), 0, zero_mask.get_byte_size());
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, zero_vis);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, zero_mask);
        }

        impl_->text_request_.infer();
        int64_t past_len = static_cast<int64_t>(seq);

        // ── Decode steps (warm up KV cache growth path) ──
        for (int step = 0; step < kDecodeSteps; ++step) {
            impl_->step_ids_.data<int64_t>()[0] = 1;

            auto* pos_data = impl_->decode_pos_.data<int64_t>();
            pos_data[0] = past_len;
            pos_data[B] = past_len;
            pos_data[2 * B] = past_len;

            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kInputIds, impl_->step_ids_);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kAttentionMask, impl_->step_mask_);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kPositionIds, impl_->decode_pos_);
            impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kBeamIdx, impl_->beam_idx_);

            if (impl_->model_.compiled_vision()) {
                impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualEmbeds, impl_->decode_visual_);
                impl_->text_request_.set_tensor(models::Qwen3_5TextIO::kVisualPosMask, impl_->decode_visual_mask_);
            }

            impl_->text_request_.infer();
            past_len += 1;
        }

        impl_->text_request_.reset_state();
    };

    // ── Phase 1: Text-only warmup (always) ──
    run_text_warmup(nullptr, nullptr, nullptr);

    // ── Phase 2: VL warmup (if vision model loaded) ──
    if (impl_->model_.compiled_vision()) {
        // Create a small dummy image (224x224 RGB) and run through full VL pipeline
        constexpr size_t kWarmupImageH = 224;
        constexpr size_t kWarmupImageW = 224;
        constexpr size_t kChannels = 3;
        ov::Tensor dummy_image(ov::element::u8, {kWarmupImageH, kWarmupImageW, kChannels});
        std::memset(dummy_image.data(), 128, dummy_image.get_byte_size());

        // Vision encode
        auto [visual_embeds, grid_thw] = impl_->encode_vision(dummy_image);

        // Build VL text inputs with image tokens
        const int64_t num_vis_tokens = models::Qwen3_5VisionPreprocessor::count_visual_tokens(
            grid_thw, cfg.vision.spatial_merge_size);
        // Ensure total seq is at least max_seq_len by padding with text tokens
        const size_t vl_seq = std::max(seq, static_cast<size_t>(num_vis_tokens + 32));

        ov::Tensor vl_input_ids(ov::element::i64, {B, vl_seq});
        auto* id_data = vl_input_ids.data<int64_t>();
        // Fill with regular token, mark image positions with image_token_id
        std::fill_n(id_data, vl_seq, int64_t{1});
        for (int64_t i = 0; i < num_vis_tokens && i < static_cast<int64_t>(vl_seq); ++i) {
            id_data[i] = cfg.image_token_id;
        }

        ov::Tensor vl_mask(ov::element::i64, {B, vl_seq});
        std::fill_n(vl_mask.data<int64_t>(), vl_seq, int64_t{1});

        models::Qwen3_5InputPlanner planner(cfg);
        auto plan = planner.build_plan(vl_input_ids, &vl_mask, &grid_thw);

        auto visual_padded = models::Qwen3_5InputPlanner::scatter_visual_embeds(
            visual_embeds, plan.visual_pos_mask);

        run_text_warmup(&visual_padded, &plan.visual_pos_mask, &grid_thw);
    }

    impl_->past_len_ = 0;
    impl_->generated_ids_.clear();
}

void Session::recreate() {
    impl_->text_request_ = {};  // release old request first
    impl_->init_request_and_tensors();
}

bool Session::is_generating() const {
    return impl_->is_generating_.load();
}

}  // namespace ov::genai::modeling
