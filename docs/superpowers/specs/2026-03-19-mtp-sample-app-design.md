# MTP in modeling_qwen3_5 Sample App — Design Spec

**Date:** 2026-03-19
**Author:** Chuansheng Liu
**Status:** Approved
**Repository:** openvino.genai

---

## 1. Goal

Integrate MTP (Multi-Token Prediction) speculative decoding directly into the
`modeling_qwen3_5.cpp` sample app (`--mode text`) without going through the
`LLMPipeline` / `MtpSpeculativeLLMPipeline` layer. This exercises the explicit
modeling API end-to-end and enables per-step decode control not available via the
high-level pipeline.

**Scope:**
- Single-step MTP only (1 draft token per decode step). Multi-step deferred.
- Auto-detect MTP model XML/BIN pair alongside the main model path.
- `--no-mtp` flag to force single-token mode.
- Reuse `MtpDraftRunner` after refactoring it to remove the pipeline-layer dependency.

---

## 2. Architecture

### 2.1 Current vs Target

**Current:**
```
speculative_decoding/stateful/mtp_draft_strategy.cpp
  └── MtpDraftRunner  ← owns MTP InferRequest
        └── reads hidden_states via LLMInferWrapper& (pipeline-layer dependency)
```

**Target:**
```
modeling/models/qwen3_5/
  ├── mtp_draft_runner.hpp   ← new home, decoupled interface (namespace ov::genai)
  └── mtp_draft_runner.cpp

speculative_decoding/stateful/mtp_draft_strategy.cpp
  └── #include mtp_draft_runner.hpp
  └── MtpDraftRunner(mtp_request, m_main_runner->get_infer_request())
        ↑ passes raw InferRequest instead of LLMInferWrapper

modeling/samples/modeling_qwen3_5.cpp  (--mode text)
  └── auto-detect mtp_model.xml + mtp_model.bin alongside main model
  └── compile MTP → InferRequest  (same device as main model; no CPU fallback needed)
  └── MtpDraftRunner(mtp_request, main_infer_request)
  └── draft-verify decode loop
  └── --no-mtp flag bypasses MTP entirely
  └── MTP suppressed automatically when --mode vl
```

### 2.2 Data Flow (one decode step, MTP active)

```
main InferRequest.infer()         →  hidden_states output
MtpDraftRunner.infer_next()       →  reads hidden_states, produces draft_token
main InferRequest.infer_next()    →  verifies out_token, produces ref0
  if ref0 == draft_token: ACCEPT  →  run main once more, emit 2 tokens
  else:            REJECT         →  emit ref0, 1 token
```

---

## 3. Changes

### 3.1 New files: `mtp_draft_runner.hpp` / `mtp_draft_runner.cpp`

Location: `src/cpp/src/modeling/models/qwen3_5/`
Namespace: `ov::genai` (unchanged from current `MtpDraftRunner` location — keeps
`mtp_draft_strategy.cpp` include/namespace changes minimal).

**Interface:**
```cpp
namespace ov::genai {

class MtpDraftRunner {
public:
    // mtp_request: owned. main_request: non-owning reference (read hidden_states).
    MtpDraftRunner(ov::InferRequest mtp_request,
                   ov::InferRequest& main_request);

    // Draft one token.
    // position: 0-based KV length of the main model BEFORE this call
    //   (i.e. the position_id the draft token will occupy).
    // Returns the draft token (argmax of MTP logits).
    int64_t infer_next(int64_t prev_token_id, int64_t position);

    void trim_kv_cache(std::size_t trim_count);
    void reset_state();

private:
    ov::InferRequest  mtp_runner_;
    ov::InferRequest& main_runner_ref_;   // replaces LLMInferWrapper&
    // kv_pos_, scalar buffers, num_processed_tokens_ — unchanged
};

} // namespace ov::genai
```

**Key change from current `MtpDraftRunner`:**
- `main_runner_ref_` type: `LLMInferWrapper&` → `ov::InferRequest&`
- `infer_next()`: `main_runner_ref_.get_infer_request().get_tensor("hidden_states")`
  becomes `main_runner_ref_.get_tensor("hidden_states")`
- All other logic unchanged (KV trim, reset, argmax, f16 handling)

### 3.2 Update `mtp_draft_strategy.hpp`

Replace internal `MtpDraftRunner` definition with `#include` of the new external header.

### 3.3 Update `mtp_draft_strategy.cpp`

Constructor call site:
```cpp
// Before:
m_mtp_runner = std::make_unique<MtpDraftRunner>(std::move(mtp_request), *m_main_runner);

// After:
m_mtp_runner = std::make_unique<MtpDraftRunner>(std::move(mtp_request),
                                                  m_main_runner->get_infer_request());
```

`LLMInferWrapper::get_infer_request()` already exists — no changes to that class.

### 3.4 `modeling_qwen3_5.cpp` — `--mode text` decode loop

**New CLI flag:**
```
--no-mtp    Disable MTP speculative decoding even if MTP model is auto-detected
```

**MTP is suppressed (not activated) when `--mode vl`** — the VL path uses
`use_inputs_embeds=true` which is incompatible with `output_hidden_states=true`
(enforced by a compile-time assert in `create_qwen3_5_text_model()`).

**Auto-detect logic** (when `--mode text` and `--no-mtp` not set):
Check both files exist:
```
<model_dir>/mtp_model.xml
<model_dir>/mtp_model.bin
```
Use `has_ir_model_pair()` (same helper already used in the sample for other IR
checks) to verify both files are present. If found → compile MTP model on the
same device as the main model; construct `MtpDraftRunner`.
If not found → silent fallback to single-token decode.

**`output_hidden_states` and caching:** the main model is always built with
`output_hidden_states=true` when `cfg.mtp_num_hidden_layers > 0` (determined
from `config.json`), regardless of whether `mtp_model.xml` is present at
runtime. This means:
- One cache file per model config (`qwen3_5_text.xml`) — no split filename.
- MTP can be toggled on/off at runtime via `--no-mtp` or by adding/removing
  `mtp_model.xml` without invalidating or rebuilding the cached IR.
- Models with `mtp_num_hidden_layers == 0` build without `hidden_states` output
  and cache under the same filename — no change for those models.
- OV compiled model cache (`ov::cache_dir`): the same `cache_dir` property is
  passed to both main model and MTP model `compile_model()` calls, so GPU
  kernel blobs for both are stored in the same cache directory.

If `hidden_states` is absent from a loaded cached IR at runtime (stale cache
from before this feature), assert with a clear message: "hidden_states output
missing — delete qwen3_5_text.xml and retry".

**Decode loop (MTP active):**

```
prefill:
  main.infer(full_prompt)  →  out_token

mtp_prefix_token = -1   // sentinel: -1 means "no pending KV sync token"
                        // (-1 is safe; token IDs are non-negative)

loop while can_continue:           // can_continue: generated < max_new_tokens
                                   //   and out_token not in stop_token_ids
  past_len = <sample's existing past_len counter>
  // past_len = number of tokens in main KV cache at start of this iteration
  // (same variable already maintained by the sample's single-token loop)

  // KV sync after ACCEPT: advance MTP KV to catch up with main KV
  // After ACCEPT, main KV length = past_len_at_draft_time + 2.
  // MTP KV length = past_len_at_draft_time + 1 (from the draft call).
  // We call infer_next at position = past_len - 1 to bring MTP to past_len.
  if mtp_prefix_token >= 0:
    mtp_runner.infer_next(mtp_prefix_token, past_len - 1)
    mtp_prefix_token = -1

  draft_token = mtp_runner.infer_next(out_token, past_len)
  ref0        = main.infer_next(out_token)

  if ref0 == draft_token:           // ACCEPT
    // Check draft_token stop condition BEFORE running second main infer
    emit draft_token
    increment generated count
    if draft_token in stop_token_ids or generated >= max_new_tokens:
      break
    ref1 = main.infer_next(draft_token)
    emit ref1
    increment generated count
    out_token = ref1
    mtp_prefix_token = draft_token
    if ref1 in stop_token_ids or generated >= max_new_tokens:
      break
  else:                             // REJECT
    emit ref0
    increment generated count
    out_token = ref0
    // mtp_prefix_token stays -1

// Print accept/reject rate to stderr (mirrors existing MtpSpeculativeLLMPipeline)
```

**Decode loop (MTP inactive / --no-mtp):**
Unchanged from current single-token loop.

**Timing / metrics:** the sample already measures TTFT and TPOT manually;
no changes required. Accept/reject rate printed to stderr.

### 3.5 `CMakeLists.txt`

Add `mtp_draft_runner.cpp` to the `modeling/models/qwen3_5/` source list.

---

## 4. Error Handling

| Condition | Behaviour |
|---|---|
| MTP XML+BIN found but fails to compile | Log warning to stderr; fall back to single-token |
| MTP XML+BIN found but `hidden_states` absent from main model output | `OPENVINO_ASSERT` with message: "hidden_states output missing — delete qwen3_5_text.xml and retry" |
| `--mode vl` with MTP XML+BIN present | MTP silently suppressed; single-token decode used |
| `--no-mtp` set (regardless of MTP XML presence) | Single-token decode; no error |

---

## 5. What Does NOT Change

- `MtpSpeculativeLLMPipeline` decode logic — unaffected (only constructor call site updated)
- `LLMInferWrapper` — unaffected
- `fast_draft_strategy` (dflash_draft path) — unaffected
- `StatefulLLMPipeline` (non-speculative) — unaffected
- GPU plugin — no changes

---

## 6. Out of Scope

- Multi-step MTP (N > 1 draft tokens): deferred. Would require the MTP OV model
  to expose its own updated hidden states as an output (currently only `logits`
  is exported), plus partial-accept logic for the verify phase.

---

## 7. Testing Plan

| Test | Command | What it verifies |
|---|---|---|
| [27] 2B — MTP on | `modeling_qwen3_5.exe --mode text` (mtp_model.xml present) | MTP auto-detect on small model; accept rate + coherent output |
| [27] 2B — MTP off | `modeling_qwen3_5.exe --mode text --no-mtp` | Single-token fallback; output matches baseline |
| [37] 35B-A3B — MTP on | `modeling_qwen3_5.exe --mode text` (mtp_model.xml present) | Full-scale MTP; accept rate + coherent output |
| [37] 35B-A3B — MTP off | `modeling_qwen3_5.exe --mode text --no-mtp` | Single-token fallback at full scale |
| [39] benchmark_genai | existing `auto_tests.py --tests 39` | Regression: `MtpSpeculativeLLMPipeline` unaffected by call-site change |

---

## 8. File Paths Summary

All paths relative to `openvino.genai/src/cpp/src/`:

```
modeling/models/qwen3_5/
  mtp_draft_runner.hpp              ← NEW (extracted + decoupled from pipeline)
  mtp_draft_runner.cpp              ← NEW
  CMakeLists.txt                    ← add mtp_draft_runner.cpp

speculative_decoding/stateful/
  mtp_draft_strategy.hpp            ← update #include + remove inline MtpDraftRunner
  mtp_draft_strategy.cpp            ← update constructor call site

modeling/samples/
  modeling_qwen3_5.cpp              ← add --no-mtp, auto-detect, MTP decode loop,
                                       VL suppression, output_hidden_states by config
```
