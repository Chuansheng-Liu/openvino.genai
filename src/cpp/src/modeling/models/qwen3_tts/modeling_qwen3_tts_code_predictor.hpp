// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS Code Predictor Module
//
// The Code Predictor generates Layer 1-15 codec tokens in parallel using
// a smaller 5-layer transformer. It takes the hidden states from the Talker
// and autoregressively predicts each of the 15 remaining codec layers.
//
// Architecture: 5-layer transformer with GQA (16 heads, 8 KV heads)
// Input: Talker hidden states projected to 1024 dim + previous codec embeddings
// Output: Codec token logits for layers 1-15
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include "modeling/models/qwen3_tts/modeling_qwen3_tts.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/vocab_embedding.hpp"

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {

class BuilderContext;

namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights

namespace models {

//===----------------------------------------------------------------------===//
// Code Predictor Attention - Standard RoPE (not mRoPE)
//===----------------------------------------------------------------------===//
class Qwen3TTSCodePredictorAttention : public Module {
public:
    Qwen3TTSCodePredictorAttention(BuilderContext& ctx,
                                   const std::string& name,
                                   const Qwen3TTSCodePredictorConfig& cfg,
                                   Module* parent = nullptr);

    // Forward without KV cache
    Tensor forward_no_cache(const Tensor& hidden_states,
                            const Tensor& rope_cos,
                            const Tensor& rope_sin,
                            const Tensor& causal_mask) const;

    // Forward with KV cache (for decode)
    AttentionKVOutput forward_with_cache(const Tensor& hidden_states,
                                         const Tensor& rope_cos,
                                         const Tensor& rope_sin,
                                         const Tensor& attention_mask,
                                         const std::optional<Tensor>& past_key,
                                         const std::optional<Tensor>& past_value) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    // Q/K normalization (standard for Qwen3)
    RMSNorm q_norm_;
    RMSNorm k_norm_;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 0.0f;
};

//===----------------------------------------------------------------------===//
// Code Predictor MLP - SwiGLU
//===----------------------------------------------------------------------===//
class Qwen3TTSCodePredictorMLP : public Module {
public:
    Qwen3TTSCodePredictorMLP(BuilderContext& ctx,
                             const std::string& name,
                             const Qwen3TTSCodePredictorConfig& cfg,
                             Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Code Predictor Decoder Layer
//===----------------------------------------------------------------------===//
class Qwen3TTSCodePredictorDecoderLayer : public Module {
public:
    Qwen3TTSCodePredictorDecoderLayer(BuilderContext& ctx,
                                      const std::string& name,
                                      const Qwen3TTSCodePredictorConfig& cfg,
                                      Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward_no_cache(const Tensor& hidden_states,
                                               const Tensor& rope_cos,
                                               const Tensor& rope_sin,
                                               const Tensor& causal_mask,
                                               const std::optional<Tensor>& residual) const;

    // Forward with KV cache (for decode)
    DecoderLayerKVOutput forward_with_cache(const Tensor& hidden_states,
                                            const Tensor& rope_cos,
                                            const Tensor& rope_sin,
                                            const Tensor& attention_mask,
                                            const std::optional<Tensor>& residual,
                                            const std::optional<Tensor>& past_key,
                                            const std::optional<Tensor>& past_value) const;

private:
    Qwen3TTSCodePredictorAttention self_attn_;
    Qwen3TTSCodePredictorMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

//===----------------------------------------------------------------------===//
// Code Predictor Model (5-layer transformer)
// Contains: layers + norm + 15 codec_embeddings (in model namespace)
//===----------------------------------------------------------------------===//
class Qwen3TTSCodePredictorModel : public Module {
public:
    Qwen3TTSCodePredictorModel(BuilderContext& ctx,
                               const Qwen3TTSCodePredictorConfig& cfg,
                               Module* parent = nullptr);

    // Forward without KV cache
    // Input: inputs_embeds [B, T, hidden_size=1024]
    // Output: hidden_states [B, T, hidden_size=1024]
    Tensor forward_no_cache(const Tensor& inputs_embeds,
                            const Tensor& position_ids) const;

    // Forward with KV cache
    // Input: inputs_embeds [B, T, hidden_size=1024], position_ids [B, T],
    //        attention_mask [B, 1, T, kv_len], past KV caches
    // Output: hidden_states, key/value caches for all layers
    TalkerModelKVOutput forward_with_cache(const Tensor& inputs_embeds,
                                           const Tensor& position_ids,
                                           const Tensor& attention_mask,
                                           const std::vector<Tensor>& past_keys,
                                           const std::vector<Tensor>& past_values) const;

    // Get codec embedding for a specific layer (0..14 -> layers 1..15)
    Tensor get_codec_embed(const Tensor& codec_ids, int layer_idx) const;

    // Access codec embedding
    VocabEmbedding& codec_embedding(int layer_idx);

private:
    Qwen3TTSCodePredictorConfig cfg_;
    std::vector<Qwen3TTSCodePredictorDecoderLayer> layers_;
    RMSNorm norm_;
    // 15 codec embeddings for layers 1-15
    // HF path: talker.code_predictor.model.codec_embedding.N.weight
    std::vector<VocabEmbedding> codec_embeddings_;

    int32_t head_dim_ = 0;
    float rope_theta_ = 0.0f;
};

//===----------------------------------------------------------------------===//
// Code Predictor For Conditional Generation
// Contains: model (with codec_embeddings) + 15 lm_heads + small_to_mtp_projection
// Module path: talker.code_predictor
//===----------------------------------------------------------------------===//
class Qwen3TTSCodePredictorForConditionalGeneration : public Module {
public:
    Qwen3TTSCodePredictorForConditionalGeneration(BuilderContext& ctx,
                                                  const Qwen3TTSCodePredictorConfig& cfg,
                                                  Module* parent = nullptr);

    // Forward for a specific generation step (predicting layer `step+1`)
    // Input:
    //   - inputs_embeds: [B, T, talker_hidden_size] (sum of talker hidden + codec embeds)
    //   - position_ids: [B, T]
    //   - step: 0..14 (generation step for layers 1..15)
    // Output: logits [B, T, vocab_size]
    // Note: inputs_embeds is projected from talker_hidden_size to hidden_size internally
    Tensor forward_no_cache(const Tensor& inputs_embeds,
                            const Tensor& position_ids,
                            int step) const;

    // Forward with KV cache — outputs logits for ALL 15 steps plus updated KV caches
    // At runtime, caller picks logits for the relevant step
    struct CPForwardKVOutput {
        std::vector<Tensor> all_logits;  // 15 logits tensors
        std::vector<Tensor> key_caches;
        std::vector<Tensor> value_caches;
    };
    CPForwardKVOutput forward_with_cache(const Tensor& inputs_embeds,
                                         const Tensor& position_ids,
                                         const Tensor& attention_mask,
                                         const std::vector<Tensor>& past_keys,
                                         const std::vector<Tensor>& past_values) const;

    // Forward with KV cache returning only hidden state (no lm_heads)
    // Used by the hidden model to reduce GPU kernel count
    TalkerModelKVOutput forward_hidden_with_cache(const Tensor& inputs_embeds,
                                                   const Tensor& position_ids,
                                                   const Tensor& attention_mask,
                                                   const std::vector<Tensor>& past_keys,
                                                   const std::vector<Tensor>& past_values) const;

    // Get codec embedding for a specific layer (0..14 -> layers 1..15)
    Tensor get_codec_embed(const Tensor& codec_ids, int layer_idx) const;

    // Get sum of all codec embeddings for layers 0..layer_idx
    Tensor get_codec_embeds_sum(const std::vector<Tensor>& codec_ids_list) const;

    // Access sub-modules
    Qwen3TTSCodePredictorModel& model();
    VocabEmbedding& codec_embedding(int layer_idx);
    LMHead& lm_head(int step);

    // Apply input projection: talker_hidden_size -> hidden_size
    Tensor apply_input_projection(const Tensor& inputs_embeds) const;

    // Forward with KV cache for a single step (applies only the specified lm_head)
    struct SingleStepKVOutput {
        Tensor logits;
        std::vector<Tensor> key_caches;
        std::vector<Tensor> value_caches;
    };
    SingleStepKVOutput forward_step_with_cache(
        const Tensor& inputs_embeds,
        const Tensor& position_ids,
        const Tensor& attention_mask,
        const std::vector<Tensor>& past_keys,
        const std::vector<Tensor>& past_values,
        int32_t step) const;

private:
    Qwen3TTSCodePredictorConfig cfg_;
    Qwen3TTSCodePredictorModel model_;

    // 15 lm_heads for predicting layers 1-15
    // HF path: talker.code_predictor.lm_head.N.weight
    std::vector<LMHead> lm_heads_;

    // Projection from talker hidden_size to code predictor hidden_size
    // HF path: talker.code_predictor.small_to_mtp_projection.weight/bias
    // Only used when talker_hidden_size != hidden_size
    WeightParameter* projection_weight_ = nullptr;
    WeightParameter* projection_bias_ = nullptr;
    bool needs_projection_ = false;
};

//===----------------------------------------------------------------------===//
// Factory Functions - Code Predictor Models
//===----------------------------------------------------------------------===//

// Create Code Predictor model (for Layer 1-15 codec generation)
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor AR model for specific generation step
// Input:
//   - inputs_embeds: [batch, seq_len, hidden_size] from codec embeddings
//   - position_ids: [batch, seq_len] position indices
// Output:
//   - logits: [batch, seq_len, vocab_size] next token logits
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int generation_step,  // 0..14 for groups 1..15
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor codec embedding model (for decode phase)
// Gets the sum of all 15 codec embeddings from CodePredictor (layers 1-15)
// Input:
//   - codec_input_0 to codec_input_14: [batch, 1] tokens for each layer
// Output:
//   - codec_embeds_sum: [batch, 1, hidden_size] sum of all embeddings
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor single codec embedding model
// Input: codec_input [batch, 1] token for specific layer
// Output: codec_embed [batch, 1, hidden_size]
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_single_codec_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    int codec_layer,  // 0..14 for layers 1..15
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create unified Code Predictor AR model with all 15 lm_heads
// Single transformer forward, outputs logits for all 15 generation steps
// Input:
//   - inputs_embeds: [batch, seq_len, talker_hidden_size]
//   - position_ids: [batch, seq_len]
// Output:
//   - logits_0 through logits_14: [batch, 1, vocab_size] for each step
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_unified_ar_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create Code Predictor AR decode model with KV cache and all 15 lm_heads
// Supports both prefill (T>1) and decode (T=1) via KV cache
// Input:
//   - inputs_embeds: [batch, T, talker_hidden_size]
//   - position_ids: [batch, T]
//   - attention_mask: [batch, 1, T, kv_len]
//   - past_key_i / past_value_i: [batch, kv_heads, past_len, head_dim] for each layer
// Output:
//   - logits_0 through logits_14: [batch, 1, vocab_size]
//   - present_key_i / present_value_i: [batch, kv_heads, total_len, head_dim]
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_decode_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create unified codec embedding model for all 15 layers
// Input: codec_input [batch, 1] token, layer_index [1] int32
// Output: codec_embed [batch, 1, talker_hidden_size]
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_unified_embed_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create CP AR decode model that outputs hidden_state (no lm_heads on GPU)
// lm_head matmuls are done on CPU to reduce GPU kernel count by ~30
// Input: inputs_embeds, position_ids, attention_mask, past_key/value per layer
// Output: hidden_state [B, 1, hidden_size], present_key/value per layer
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_ar_hidden_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

// Create unrolled CP model — all 15 AR steps in a single OV model.
// Uses re-prefill approach (no KV cache): each step processes the full growing sequence.
// ArgMax and embedding lookups happen ON GPU, eliminating 14 CPU-GPU sync points.
// Input: past_hidden [1, 1, talker_hidden], layer0_embed [1, 1, talker_hidden]
// Output: codec_sum [1, 1, talker_hidden], token_0..token_14 [1, 1] each
std::shared_ptr<ov::Model> create_qwen3_tts_code_predictor_unrolled_model(
    const Qwen3TTSCodePredictorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
