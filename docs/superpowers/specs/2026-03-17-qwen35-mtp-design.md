# Qwen3.5 Multi-Token Prediction (MTP) — Design Spec

**Date:** 2026-03-17
**Author:** Chuansheng Liu
**Status:** Approved
**Repositories:** openvino.genai, openvino (GPU plugin — read-only, no changes required)

---

## 1. Goal

Implement the Qwen3.5 MTP (Multi-Token Prediction) speculative decoding feature using the openvino.genai modeling API. MTP is a built-in draft mechanism: the target model's checkpoint contains dedicated MTP layers that predict one additional token per decode step using the main model's final hidden state, enabling ~1.5–2× throughput improvement with zero accuracy loss via rejection sampling.

**Target model:** Qwen3.5-35B-A3B (MoE, 4-bit), already running at 18+ tok/s up to 32K context.
**Reference:** vLLM `qwen3_5_mtp.py` + `eagle.py` + `rejection_sampler.py`.

---

## 2. Scope

- Single-step MTP: draft exactly 1 token per decode step.
- Strict (argmax) rejection sampling: accept draft iff main model's argmax at that position matches.
- Qwen3.5-35B-A3B (MoE) as primary target; design is generic for dense Qwen3.5 variants.
- No GPU plugin changes required (all needed ops already exist).
- No changes to the existing non-MTP pipeline paths.

---

## 3. Architecture

### 3.1 High-Level Structure

Two separate compiled OpenVINO models; one new draft runner class; one new pipeline strategy.

```
StatefulSpeculativeLLMPipeline
├── main_runner (LLMInferWrapper)
│     OV model: Qwen3_5ForCausalLM (modified)
│     outputs: logits [B,T,V]  +  hidden_states [B,T,H]  (new, optional)
│
└── draft_runner (MtpDraftRunner)          ← new
      OV model: Qwen3_5MtpForDraft         ← new
      inputs:  input_ids, position_ids, hidden_states, beam_idx
      outputs: logits [B,1,V]
```

### 3.2 Decode Loop (single-step MTP)

```
prefill:
  main.infer_first(prompt)  →  token_0, hidden_0

decode step N:
  1. mtp.infer(token_N, hidden_N)  →  draft_token_{N+1}      [MTP KV advances to N+1]
  2. main.infer(draft_token_{N+1}) →  logits_{N+1}, hidden_{N+1}  [main KV advances]
  3. predicted = argmax(logits_N_slot)   (main's prediction for position N)
  4. if draft_token_{N+1} == predicted:
         ACCEPT — emit draft_token_{N+1}, next_hidden = hidden_{N+1}
     else:
         REJECT — emit predicted, mtp.trim_kv_cache(1), main.trim_kv_cache(1)
         re-run: main.infer(predicted, position=N+1) → logits, hidden
         # position_ids stays at N+1 for the re-run; both KVs were trimmed back to N
```

Rejection triggers one extra main-model step to correct state; accepted steps cost only one MTP inference instead of one full main-model decode step.

---

## 4. Changes in openvino.genai

### 4.1 `processing_qwen3_5.hpp`

Add to `Qwen3_5TextConfig`:
```cpp
int32_t mtp_num_hidden_layers = 0;   // 0 = MTP disabled
```
Populated from `config.json` key `mtp_num_hidden_layers`.

### 4.2 `modeling_qwen3_5_text.cpp` — two changes only

**A. Silently skip `mtp.*` weights** (identical to Qwen3Next pattern, line ~1365):
```cpp
if (starts_with(name, "mtp.") || ends_with(name, "_scale_inv")) {
    continue;
}
```

**B. Optional `hidden_states` output.** Add overload to `Qwen3_5ForCausalLM`:
```cpp
// New: returns {logits, hidden_states_after_norm}
std::pair<Tensor, Tensor> forward_with_hidden(
    const Tensor& input_ids,
    const Tensor& position_ids,
    const Tensor& beam_idx,
    const Tensor* attention_mask = nullptr,
    const Tensor* sdpa_mask = nullptr);
```

`hidden_states` is the post-norm tensor immediately before `lm_head_` (same tensor that feeds the LM head; no extra computation, just an additional graph output).

`create_qwen3_5_text_model()` gets a new flag:
```cpp
std::shared_ptr<ov::Model> create_qwen3_5_text_model(
    const Qwen3_5TextConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer,
    bool output_hidden_states = false);
```

When `output_hidden_states = true`, a second `Result` node named `"hidden_states"` is added to the model outputs. The flag defaults to `false` so no existing code paths are affected.

### 4.3 New file: `modeling_qwen3_5_mtp.hpp`

```cpp
namespace ov::genai::modeling::models {

class Qwen3_5MtpForDraft : public Module {
public:
    Qwen3_5MtpForDraft(BuilderContext& ctx,
                       const std::string& name,
                       const Qwen3_5TextConfig& cfg,
                       Module* parent = nullptr);

    // Returns logits [B, 1, vocab_size]
    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor* attention_mask = nullptr) const;

private:
    // Components (all existing Module types)
    VocabEmbedding           embed_tokens_;
    Qwen3_5RMSNorm           pre_fc_norm_embed_;
    Qwen3_5RMSNorm           pre_fc_norm_hidden_;
    WeightParameter*         fc_param_ = nullptr;   // [hidden, 2*hidden]
    Qwen3_5DecoderLayer      layer_;                // full_attention, index = cfg.num_hidden_layers
    Qwen3_5RMSNorm           norm_;
    LMHead                   lm_head_;
};

// Factory — builds and compiles the MTP OV model
std::shared_ptr<ov::Model> create_qwen3_5_mtp_model(
    const Qwen3_5TextConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

} // namespace
```

### 4.4 New file: `modeling_qwen3_5_mtp.cpp`

**Forward pass:**
```
1. embed        = embed_tokens_(input_ids)                  [B,1,H]
2. e_norm       = pre_fc_norm_embed_.forward(embed)         [B,1,H]
3. h_norm       = pre_fc_norm_hidden_.forward(hidden_states)[B,1,H]
4. fused        = ops::concat({e_norm, h_norm}, /*axis=*/-1)[B,1,2H]
5. projected    = ops::linear(fused, fc_weight)             [B,1,H]
6. out, _       = layer_.forward(position_ids, projected,
                                 beam_idx, attention_mask)  [B,1,H]
7. normed       = norm_.forward(out)                        [B,1,H]
8. logits       = lm_head_.forward(normed)                  [B,1,V]
```

`Qwen3_5DecoderLayer` is instantiated with `layer_type = "full_attention"` and `layer_index = cfg.num_hidden_layers` (so the MTP layer's KV cache variable names don't collide with the main model's layers).

`create_qwen3_5_mtp_model()`:
- Builds the graph using the same `BuilderContext` pattern as the main model factory.
- Parameters: `input_ids [B,1]`, `position_ids [B,1]`, `hidden_states [B,1,H]`, `beam_idx [B]`.
- Sets `rt_info` for f16 KV cache precision (same as main model).
- Weight loading uses `build_qwen3_5_mtp_weight_specs()`.
- Silently skips any unmatched weights under `model.` prefix (main-model weights not relevant here).

### 4.5 `qwen3_5_weight_specs.hpp/cpp` — add `build_qwen3_5_mtp_weight_specs()`

Checkpoint → module path mapping:

| Checkpoint key | Module path in Qwen3_5MtpForDraft |
|---|---|
| `mtp.embed_tokens.weight` | `embed_tokens.weight` |
| `mtp.pre_fc_norm_embedding.weight` | `pre_fc_norm_embed.weight` |
| `mtp.pre_fc_norm_hidden.weight` | `pre_fc_norm_hidden.weight` |
| `mtp.fc.weight` | `fc.weight` |
| `mtp.layers.0.input_layernorm.weight` | `layer.input_layernorm.weight` |
| `mtp.layers.0.post_attention_layernorm.weight` | `layer.post_attention_layernorm.weight` |
| `mtp.layers.0.self_attn.q_proj.weight` | `layer.self_attn.q_proj.weight` |
| `mtp.layers.0.self_attn.k_proj.weight` | `layer.self_attn.k_proj.weight` |
| `mtp.layers.0.self_attn.v_proj.weight` | `layer.self_attn.v_proj.weight` |
| `mtp.layers.0.self_attn.o_proj.weight` | `layer.self_attn.o_proj.weight` |
| `mtp.layers.0.self_attn.q_norm.weight` | `layer.self_attn.q_norm.weight` |
| `mtp.layers.0.self_attn.k_norm.weight` | `layer.self_attn.k_norm.weight` |
| `mtp.layers.0.mlp.gate.weight` | `layer.mlp.gate.weight` |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | `layer.mlp.shared_expert_gate.weight` |
| `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight` | `layer.mlp.shared_expert.{...}.weight` |
| `mtp.layers.0.mlp.experts.N.{gate,up,down}_proj.weight` | `layer.mlp.experts.N.{...}.weight` |
| `mtp.norm.weight` | `norm.weight` |
| `model.lm_head.weight` | `lm_head.weight` |

The last entry (`model.lm_head.weight`) reuses the same checkpoint tensor as the main model's LM head. Both models load it independently (same values, separate OV Parameter nodes).

For quantized MoE experts, the same scale/zp auxiliary capture mechanism used in `Qwen3_5SparseMoeBlock` applies via the custom `weight_loader` lambda.

### 4.6 New files: `mtp_draft_strategy.hpp/cpp`

**`MtpDraftRunner`** — wraps the MTP OV model, mirrors `LLMInferWrapper` API:
```cpp
class MtpDraftRunner {
public:
    MtpDraftRunner(ov::InferRequest mtp_request,
                   const ov::InferRequest& main_request);  // read hidden_states from main

    void infer_first(const TokenIds& prompt_ids);  // no-op: MTP KV pre-filled lazily
    int64_t infer_next(int64_t prev_token_id, int32_t position);
    ov::Tensor get_logits() const;
    void trim_kv_cache(int trim_by);
    void reset_state();

private:
    ov::InferRequest mtp_runner_;
    const ov::InferRequest& main_runner_;  // read-only
    ov::Tensor input_ids_buf_, position_ids_buf_, beam_idx_buf_;
};
```

`infer_next()` implementation:
1. Copy `hidden_states` output tensor from `main_runner_` into `mtp_runner_`'s `hidden_states` input.
2. Set `input_ids = prev_token_id`, `position_ids = position`, `beam_idx = 0`.
3. Call `mtp_runner_.infer()`.
4. Return `argmax(get_logits())` as draft token.

**Pipeline wiring** in `StatefulSpeculativeLLMPipeline` (or a thin `MtpSpeculativeStrategy` subclass): when `cfg.mtp_num_hidden_layers > 0`, instantiate `MtpDraftRunner` instead of `LLMInferWrapper` for the draft side. The verification (steps 3–4 of the decode loop) stays in the existing rejection-sampling path — it only needs to compare argmax values, so no new sampler code is needed.

---

## 5. What Does NOT Change

- Existing `fast_draft_strategy` (dflash_draft path) — untouched.
- Existing `StatefulLLMPipeline` (non-speculative) — untouched.
- GPU plugin: no new ops, no new kernels, no new passes.
- Continuous batching pipeline — out of scope for this feature.
- Multi-step MTP (>1 draft token) — out of scope; architecture supports it trivially later by looping `MtpDraftRunner::infer_next()`.

---

## 6. Memory & Performance Impact

| Item | Delta |
|---|---|
| Additional `hidden_states` output tensor | +`B × T × H × sizeof(f16)` per infer — negligible for decode (T=1) |
| MTP model KV cache | One extra full-attention layer's KV: `2 × num_kv_heads × max_seq × head_dim × sizeof(f16)` ≈ 32 MB at 32K context (num_kv_heads=2, head_dim=128) |
| MTP model weights | ~1 MoE layer: comparable to one main-model layer, ~300–500 MB 4-bit |
| Expected throughput gain | ~1.5–1.8× tok/s (acceptance rate depends on task; MTP typically 60–80%) |

---

## 7. Testing Plan

1. **Unit test** `create_qwen3_5_mtp_model()`: verify graph inputs/outputs, weight loading from checkpoint, no OV compilation errors.
2. **Correctness test** with `auto_tests.py --tests 39` (1K): verify that MTP output matches greedy-decode reference token-for-token (MTP with argmax rejection sampling is lossless).
3. **Performance test** `--tests 39 40 41 42 43` (1K–16K): confirm tok/s improvement vs baseline.
4. **Rejection path test**: inject a forced mismatch (temperature sampling so draft ≠ main prediction), verify KV state remains consistent across multiple steps.

---

## 8. File Paths Summary

All paths relative to `openvino.genai/`:

```
src/cpp/src/modeling/models/qwen3_5/
  processing_qwen3_5.hpp                   ← add mtp_num_hidden_layers
  modeling_qwen3_5_text.cpp                ← add hidden_states output + skip mtp.* weights
  modeling_qwen3_5_mtp.hpp                 ← NEW
  modeling_qwen3_5_mtp.cpp                 ← NEW
  qwen3_5_weight_specs.hpp                 ← add build_qwen3_5_mtp_weight_specs decl
  qwen3_5_weight_specs.cpp                 ← add build_qwen3_5_mtp_weight_specs impl

src/cpp/src/speculative_decoding/stateful/
  mtp_draft_strategy.hpp                   ← NEW: MtpDraftRunner
  mtp_draft_strategy.cpp                   ← NEW: MtpDraftRunner impl + pipeline wiring
```
