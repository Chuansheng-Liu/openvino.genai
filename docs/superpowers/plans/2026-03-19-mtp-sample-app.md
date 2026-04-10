# MTP Sample App Integration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate MTP speculative decoding into `modeling_qwen3_5.cpp` (`--mode text`) by refactoring `MtpDraftRunner` into a standalone, pipeline-decoupled utility and wiring it into the sample's decode loop.

**Architecture:** Extract `MtpDraftRunner` from `mtp_draft_strategy.hpp/cpp` into `modeling/models/qwen3_5/mtp_draft_runner.{hpp,cpp}`, replacing the `LLMInferWrapper&` dependency with a raw `ov::InferRequest&`. Update the existing pipeline call site. Then wire MTP into the sample's greedy decode loop with auto-detect, `--no-mtp` flag, and single-file caching driven by `cfg.text.mtp_num_hidden_layers`.

**Tech Stack:** C++17, OpenVINO Runtime (`ov::InferRequest`, `ov::Tensor`), CMake (`GLOB_RECURSE` picks up new `.cpp` files automatically — no CMakeLists edits needed).

**Spec:** `docs/superpowers/specs/2026-03-19-mtp-sample-app-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.hpp` | **CREATE** | `MtpDraftRunner` class — decoupled interface taking `ov::InferRequest&` |
| `src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.cpp` | **CREATE** | `MtpDraftRunner` implementation |
| `src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.hpp` | **MODIFY** | Remove inline `MtpDraftRunner` class, add `#include` of new header |
| `src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.cpp` | **MODIFY** | Remove `MtpDraftRunner` ctor/methods, update `MtpSpeculativeLLMPipeline` call sites |
| `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` | **MODIFY** | `--no-mtp` flag, auto-detect, `output_hidden_states` by config, MTP compile, MTP decode loop |

---

## Chunk 1: Extract MtpDraftRunner

### Task 1: Create `mtp_draft_runner.hpp`

**File:** `src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.hpp` (CREATE)

- [ ] **Step 1.1: Create the header file**

```cpp
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "openvino/runtime/infer_request.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Wraps the compiled MTP OV draft model (from create_qwen3_5_mtp_model()).
// Takes a non-owning reference to the main model InferRequest to read
// hidden_states from its output after each main-model inference.
class MtpDraftRunner {
public:
    // mtp_request: owned.
    // main_request: non-owning reference — must outlive this object.
    //   Used only to read "hidden_states" output after main model infer.
    MtpDraftRunner(ov::InferRequest mtp_request,
                   ov::InferRequest& main_request);

    // Draft one token.
    // prev_token_id: the last confirmed output token (fed as input_ids to MTP).
    // position: 0-based KV length of the main model BEFORE this call
    //           (i.e. the position_id the draft token will occupy).
    // Reads hidden_states from main_request, runs MTP infer, returns argmax token.
    int64_t infer_next(int64_t prev_token_id, int64_t position);

    // Roll back MTP KV cache by trim_count positions.
    void trim_kv_cache(std::size_t trim_count);

    // Reset all KV state (new sequence).
    void reset_state();

    std::size_t get_num_processed_tokens() const { return num_processed_tokens_; }

private:
    ov::InferRequest               mtp_runner_;
    ov::InferRequest&              main_runner_ref_;   // non-owning
    ov::genai::utils::KVAxesPosition kv_pos_;

    ov::Tensor input_ids_buf_;
    ov::Tensor position_ids_buf_;
    ov::Tensor beam_idx_buf_;

    std::size_t num_processed_tokens_ = 0;
};

}  // namespace genai
}  // namespace ov
```

---

### Task 2: Create `mtp_draft_runner.cpp`

**File:** `src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.cpp` (CREATE)

This is a copy of the existing `MtpDraftRunner` methods from `mtp_draft_strategy.cpp`
with one change: `main_runner_ref_.get_infer_request().get_tensor(...)` →
`main_runner_ref_.get_tensor(...)`.

- [ ] **Step 2.1: Create the implementation file**

```cpp
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"

#include <algorithm>

#include "openvino/core/type/float16.hpp"

namespace ov {
namespace genai {

MtpDraftRunner::MtpDraftRunner(ov::InferRequest mtp_request,
                               ov::InferRequest& main_request)
    : mtp_runner_(std::move(mtp_request)),
      main_runner_ref_(main_request) {
    // Get KV cache axis positions from the MTP model's runtime model
    kv_pos_ = ov::genai::utils::get_kv_axes_pos(
        mtp_runner_.get_compiled_model().get_runtime_model());

    // Pre-allocate scalar inputs
    input_ids_buf_    = ov::Tensor(ov::element::i64, {1, 1});
    position_ids_buf_ = ov::Tensor(ov::element::i64, {1, 1});
    beam_idx_buf_     = ov::Tensor(ov::element::i32, {1});
    beam_idx_buf_.data<int32_t>()[0] = 0;

    mtp_runner_.set_tensor("input_ids",    input_ids_buf_);
    mtp_runner_.set_tensor("position_ids", position_ids_buf_);
    mtp_runner_.set_tensor("beam_idx",     beam_idx_buf_);
}

int64_t MtpDraftRunner::infer_next(int64_t prev_token_id, int64_t position) {
    input_ids_buf_.data<int64_t>()[0]    = prev_token_id;
    position_ids_buf_.data<int64_t>()[0] = position;

    // Copy hidden_states from main model's output to avoid aliasing
    // KEY CHANGE vs old MtpDraftRunner: direct get_tensor() instead of
    // main_runner_ref_.get_infer_request().get_tensor()
    const ov::Tensor src = main_runner_ref_.get_tensor("hidden_states");
    ov::Tensor hidden_copy(src.get_element_type(), src.get_shape());
    src.copy_to(hidden_copy);
    mtp_runner_.set_tensor("hidden_states", hidden_copy);

    mtp_runner_.infer();
    ++num_processed_tokens_;

    // Argmax over vocabulary: logits shape [1, 1, V]
    const ov::Tensor logits = mtp_runner_.get_tensor("logits");
    const auto vocab_size   = logits.get_shape()[2];

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<ov::float16>();
        std::size_t best = 0;
        float best_val = static_cast<float>(data[0]);
        for (std::size_t i = 1; i < vocab_size; ++i) {
            float v = static_cast<float>(data[i]);
            if (v > best_val) { best_val = v; best = i; }
        }
        return static_cast<int64_t>(best);
    } else {
        const auto* data = logits.data<float>();
        return static_cast<int64_t>(
            std::max_element(data, data + vocab_size) - data);
    }
}

void MtpDraftRunner::trim_kv_cache(std::size_t trim_count) {
    if (trim_count == 0 || num_processed_tokens_ < trim_count) return;

    ov::genai::utils::KVCacheState to_trim_state;
    to_trim_state.num_tokens_to_trim = trim_count;
    to_trim_state.seq_length_axis    = kv_pos_.seq_len;
    to_trim_state.reset_mem_state    = false;
    ov::genai::utils::trim_kv_cache(mtp_runner_, to_trim_state, {});

    num_processed_tokens_ -= trim_count;
}

void MtpDraftRunner::reset_state() {
    mtp_runner_.reset_state();
    num_processed_tokens_ = 0;
}

}  // namespace genai
}  // namespace ov
```

---

### Task 3: Update `mtp_draft_strategy.hpp`

**File:** `src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.hpp` (MODIFY)

Remove the entire `MtpDraftRunner` class block (lines 21–49) and replace with
an `#include` of the new header.

- [ ] **Step 3.1: Remove inline `MtpDraftRunner` class, add `#include`**

Use the block comment `// ── MtpDraftRunner` as the start anchor and the closing
`};` immediately before the `// ── MtpSpeculativeLLMPipeline` comment as the end
anchor (approximately lines 21–49 — do not rely on line numbers, use text anchors).

Replace the block:
```cpp
// ── MtpDraftRunner ──────────────────────────────────────────────────────────
// Wraps the compiled MTP OV model (from create_qwen3_5_mtp_model()).
class MtpDraftRunner {
public:
    MtpDraftRunner(ov::InferRequest mtp_request,
                   LLMInferWrapper& main_runner);

    // Draft one token. Returns argmax of MTP logits.
    int64_t infer_next(int64_t prev_token_id, int64_t position);

    // Roll back MTP KV cache by trim_count positions.
    void trim_kv_cache(std::size_t trim_count);

    // Reset all KV state (new sequence).
    void reset_state();

    std::size_t get_num_processed_tokens() const { return num_processed_tokens_; }

private:
    ov::InferRequest          mtp_runner_;
    LLMInferWrapper&          main_runner_ref_;
    ov::genai::utils::KVAxesPosition kv_pos_;

    ov::Tensor input_ids_buf_;
    ov::Tensor position_ids_buf_;
    ov::Tensor beam_idx_buf_;

    std::size_t num_processed_tokens_ = 0;
};
```

With:
```cpp
#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"
```

The existing `#include` block at the top of the file (lines 1–17) keeps all other
includes. The `fast_draft_strategy.hpp` include (which pulls in `LLMInferWrapper`)
stays — `MtpSpeculativeLLMPipeline` still uses `LLMInferWrapper`.

---

### Task 4: Update `mtp_draft_strategy.cpp`

**File:** `src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.cpp` (MODIFY)

Two changes:
1. Remove the `MtpDraftRunner` constructor and method implementations (lines 32–99 of the current file) — they now live in `mtp_draft_runner.cpp`.
2. Update the two `MtpSpeculativeLLMPipeline` constructor call sites to pass `m_main_runner->get_infer_request()` instead of `*m_main_runner`.

- [ ] **Step 4.1: Remove `MtpDraftRunner` constructor and methods from this file**

Delete the entire `// ── MtpDraftRunner ───────────────────────────────────────────────────────────`
section through the closing `}` of `MtpDraftRunner::reset_state()` (approximately
lines 30–99 — use the section comment as start anchor and the `reset_state` closing
`}` as end anchor; do not rely on line numbers):

```
// ── MtpDraftRunner ───────────────────────────────────────────────────────────
MtpDraftRunner::MtpDraftRunner(...) { ... }
int64_t MtpDraftRunner::infer_next(...) { ... }
void MtpDraftRunner::trim_kv_cache(...) { ... }
void MtpDraftRunner::reset_state() { ... }
```

These are now in `mtp_draft_runner.cpp`.

- [ ] **Step 4.2: Update first constructor call site**

In `MtpSpeculativeLLMPipeline::MtpSpeculativeLLMPipeline(ModelDesc, ...)` (around line 130):

```cpp
// Before:
m_mtp_runner = std::make_unique<MtpDraftRunner>(
    std::move(mtp_request), *m_main_runner);

// After:
m_mtp_runner = std::make_unique<MtpDraftRunner>(
    std::move(mtp_request), m_main_runner->get_infer_request());
```

- [ ] **Step 4.3: Update second constructor call site**

In `MtpSpeculativeLLMPipeline::MtpSpeculativeLLMPipeline(unique_ptr<LLMInferWrapper>, ...)` (around line 177):

```cpp
// Before:
m_mtp_runner = std::make_unique<MtpDraftRunner>(
    std::move(mtp_request), *m_main_runner);

// After:
m_mtp_runner = std::make_unique<MtpDraftRunner>(
    std::move(mtp_request), m_main_runner->get_infer_request());
```

`LLMInferWrapper::get_infer_request()` already exists — no changes needed there.

---

### Task 5: Build and verify Chunk 1

- [ ] **Step 5.1: Build the openvino_genai target**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino/build
cmake --build . --target openvino_genai --config Release -j12
```

Expected: zero errors. Common errors and fixes:
- `'get_infer_request' is not a member of LLMInferWrapper` → check that `LLMInferWrapper::get_infer_request()` is declared in `fast_draft_strategy.hpp`
- `'KVCacheState' undeclared` in `mtp_draft_runner.cpp` → add `#include "utils.hpp"` (already in the file above)
- `'ov::float16' not found` → add `#include "openvino/core/type/float16.hpp"` (already in the file above)

- [ ] **Step 5.2: Run regression test [39] to confirm pipeline is unchanged**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
python scripts/auto_tests.py --tests 39 --models-root "C:\data\models"
```

Expected: exit code 0, coherent output, same accept rate as before (~60–80%).

- [ ] **Step 5.3: Commit Chunk 1**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino.genai
git add src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.hpp
git add src/cpp/src/modeling/models/qwen3_5/mtp_draft_runner.cpp
git add src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.hpp
git add src/cpp/src/speculative_decoding/stateful/mtp_draft_strategy.cpp
git commit -m "refactor: extract MtpDraftRunner to modeling layer, decouple from LLMInferWrapper"
```

---

## Chunk 2: Sample App Integration

### Task 6: Add `--no-mtp`, auto-detect, and `output_hidden_states` by config

**File:** `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` (MODIFY)

- [ ] **Step 6.1: Add includes near the top of the file** (after line 42, after existing `#include` block)

```cpp
#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_mtp.hpp"
```

- [ ] **Step 6.2: Add `no_mtp` field to `SampleOptions` struct** (after `bool cache_model = false;` at line 54)

```cpp
bool no_mtp = false;
```

- [ ] **Step 6.3: Add `--no-mtp` to the CLI arg parser** (after the `--think` handler, before the `else { throw ... }` at line ~277)

```cpp
} else if (arg == "--no-mtp") {
    opts.no_mtp = true;
```

- [ ] **Step 6.4: Change `output_hidden_states` from hardcoded `false` to config-driven**

In the `create_qwen3_5_text_model()` call (lines ~957–963):
```cpp
// Before:
text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
    cfg,
    weight_source,
    text_finalizer,
    false,
    use_vl,
    false);   // <-- output_hidden_states

// After:
const bool output_hidden_states = cfg.text.mtp_num_hidden_layers > 0 && !use_vl;
text_model = ov::genai::modeling::models::create_qwen3_5_text_model(
    cfg,
    weight_source,
    text_finalizer,
    false,
    use_vl,
    output_hidden_states);
```

- [ ] **Step 6.5: Add stale-cache check after text model is loaded**

Insert **after** the closing `}` of the `if (!text_model) { ... }` block (around
line ~968), so it runs regardless of whether the model was loaded from a cached IR
or freshly rebuilt from weights. This is the stale-cache detector: if an old cached
IR lacks `hidden_states`, the assert fires here.

```cpp
// Verify hidden_states output is present when config requires it.
// Runs after both the cached-IR path and the weight-rebuild path so stale
// caches (built without output_hidden_states=true) are caught immediately.
if (cfg.text.mtp_num_hidden_layers > 0 && !use_vl) {
    bool has_hs = false;
    for (const auto& out : text_model->outputs()) {
        if (out.get_any_name() == "hidden_states") { has_hs = true; break; }
    }
    OPENVINO_ASSERT(has_hs,
        "hidden_states output missing from text model — "
        "delete the cached qwen3_5_text*.xml/.bin and retry");
}
```

- [ ] **Step 6.6: Add MTP model compilation**

Insert **after the closing `}` of the VL block** (the `if (use_vl) { ... }` block
that ends around line ~1028, just before `std::unique_ptr<ov::genai::Tokenizer>`).
Do NOT insert between lines 987–989: the VL block begins immediately after `compiled_text`.
After `compiled_mtp.reset()` in the error handler, the `if (compiled_mtp)` guard
in Task 7.1 evaluates to false and the runner is never constructed — MTP is silently
disabled for the rest of the run.

```cpp
// MTP auto-detect: look for mtp_model.xml + mtp_model.bin alongside the main model.
// Only active for --mode text (not vl), when config declares MTP layers, and --no-mtp not set.
const bool try_mtp = !use_vl
                  && !opts.no_mtp
                  && !use_dummy_mode_flag
                  && cfg.text.mtp_num_hidden_layers > 0;
std::optional<ov::CompiledModel> compiled_mtp;
if (try_mtp) {
    const auto mtp_xml = model_dir / "mtp_model.xml";
    const auto mtp_bin = model_dir / "mtp_model.bin";
    if (has_ir_model_pair(mtp_xml, mtp_bin)) {
        try {
            auto mtp_ov_model = core.read_model(mtp_xml.string(), mtp_bin.string());
            compiled_mtp = core.compile_model(mtp_ov_model, opts.device);
            std::cout << "[MTP] Draft model compiled on " << opts.device << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MTP] Warning: failed to compile MTP model (" << e.what()
                      << "), falling back to single-token." << std::endl;
            compiled_mtp.reset();
        }
    } else {
        std::cout << "[MTP] mtp_model.xml not found alongside model, using single-token." << std::endl;
    }
}
```

---

### Task 7: Add MTP decode loop

**File:** `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` (MODIFY)

The MTP runner must be constructed after `text_request` (line 1119) since it holds
a reference to it.

- [ ] **Step 7.1: Construct `MtpDraftRunner` after `text_request` is created** (after line 1119)

```cpp
// Construct MTP runner (holds non-owning ref to text_request)
std::unique_ptr<ov::genai::MtpDraftRunner> mtp_runner;
if (compiled_mtp) {
    auto mtp_infer_req = compiled_mtp->create_infer_request();
    mtp_runner = std::make_unique<ov::genai::MtpDraftRunner>(
        std::move(mtp_infer_req), text_request);
}
```

- [ ] **Step 7.2: Replace the existing decode loop with an MTP-aware version**

The existing `for (int step = 1; step < opts.max_new_tokens; ++step)` loop (lines 1245–1288)
is replaced with a conditional: if `mtp_runner` is set AND `!use_sampling`, use the
MTP speculative loop; otherwise fall back to the original loop unchanged.

**`penalty_processor` decision:** The MTP path intentionally bypasses
`penalty_processor` (no `apply()`, `register_new_generated_token()`, or
`update_generated_len()` calls). This is consistent with `MtpSpeculativeLLMPipeline`,
which also uses pure greedy argmax without penalties. Users who need repetition/
frequency/presence penalties must pass `--no-mtp` to use the existing loop.
Document this in a comment at the branch point.

**`logits` and `logit_buf` capture:** The `run_main_step` lambda captures `logits`
and `logit_buf` **by reference from the outer scope**. `logits` is declared at line
~1166 (`ov::Tensor logits = text_request.get_tensor(...)`), and `logit_buf` is
declared just before. Do NOT declare a new local `logits` inside the MTP block —
that would shadow the outer variable and break the non-MTP `else` branch. The lambda
reassigns the outer `logits` variable.

Insert before the existing `for (int step...)` loop:

```cpp
// MTP speculative decode is greedy-only. Bypass if sampling is active or
// no MTP runner was constructed. Penalties are intentionally skipped in the
// MTP path (consistent with MtpSpeculativeLLMPipeline); use --no-mtp if
// repetition/frequency/presence penalties are required.
if (mtp_runner && !use_sampling) {
    // run_main_step: set step_ids/position_ids, call text_request.infer(),
    // advance past_len, return argmax of logits.
    // Captures logits and logit_buf from outer scope by reference — do NOT
    // declare a new local 'logits' inside this block.
    auto run_main_step = [&](int64_t token) -> int64_t {
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) step_data[b] = token;

        auto* pos_data = usm_decode_pos.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            const int64_t value = past_len + rope_deltas_data[b];
            pos_data[b] = value;
            pos_data[batch + b] = value;
            pos_data[2 * batch + b] = value;
        }

        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kInputIds,      step_ids);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kPositionIds,   usm_decode_pos);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kBeamIdx,       usm_beam_idx);
        text_request.infer();

        // Reassigns the outer 'logits' variable (captured by ref from line ~1166)
        logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3_5TextIO::kLogits);
        extract_last_logits_f32(logits, logit_buf);
        past_len += 1;
        return argmax_f32(logit_buf);
    };

    int64_t mtp_prefix_token = -1;  // -1 = no pending KV sync; token IDs are non-negative
    size_t accept_count = 0, reject_count = 0;

    while (generated.size() < static_cast<size_t>(opts.max_new_tokens)) {
        if (!stop_token_ids.empty() && stop_token_ids.count(next_id) > 0) break;

        // KV sync after ACCEPT (deferred to start of next iteration).
        // After ACCEPT: main KV = past_len, MTP KV = past_len - 1.
        // infer_next at position = past_len - 1 advances MTP KV to past_len.
        if (mtp_prefix_token >= 0) {
            mtp_runner->infer_next(mtp_prefix_token, past_len - 1);
            mtp_prefix_token = -1;
        }

        int64_t draft = mtp_runner->infer_next(next_id, past_len);
        int64_t ref0  = run_main_step(next_id);   // past_len advances by 1 inside

        if (ref0 == draft) {
            // ACCEPT: draft was correct. Check draft stop before 2nd main call.
            generated.push_back(draft);
            ++decode_steps;
            ++accept_count;
            if ((!stop_token_ids.empty() && stop_token_ids.count(draft) > 0) ||
                generated.size() >= static_cast<size_t>(opts.max_new_tokens)) {
                next_id = draft;
                break;
            }
            int64_t ref1 = run_main_step(draft);   // past_len advances by 1 inside
            generated.push_back(ref1);
            ++decode_steps;
            next_id          = ref1;
            mtp_prefix_token = draft;
        } else {
            // REJECT: use ref0 as next token.
            // mtp_prefix_token stays -1: both KVs are now at past_len.
            generated.push_back(ref0);
            ++decode_steps;
            ++reject_count;
            next_id = ref0;
        }
    }

    if (accept_count + reject_count > 0) {
        std::cerr << "[MTP] Accept rate: " << accept_count
                  << "/" << (accept_count + reject_count)
                  << " = " << (100.0 * accept_count / (accept_count + reject_count))
                  << "%" << std::endl;
    }
} else {
```

Then wrap the existing `for (int step = 1; ...)` loop by adding **one** closing `}`
immediately after the loop's own closing `}` (after `past_len += 1;`):

```cpp
    // (entire existing for loop body unchanged)
    for (int step = 1; step < opts.max_new_tokens; ++step) {
        ...
        past_len += 1;
    }   // ← existing for-loop closing brace (already present)
}       // ← NEW: closes the `else {` block opened above
```

**Note on `decode_steps` and throughput:** `decode_steps` starts at 0 (line ~1243).
The MTP loop increments it once per emitted token (same as the existing loop).
Throughput = `decode_steps * 1000.0 / decode_ms` gives tokens/s, unchanged.

---

### Task 8: Build and run tests

- [ ] **Step 8.1: Build `modeling_qwen3_5` sample**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino/build
cmake --build . --target modeling_qwen3_5 --config Release -j12
```

Expected: zero errors. Common errors and fixes:
- `'compiled_mtp' may be used uninitialized` → ensure `std::optional<ov::CompiledModel> compiled_mtp` is default-initialized (it is with `std::optional`)
- `'MtpDraftRunner' not in scope` → check `#include "modeling/models/qwen3_5/mtp_draft_runner.hpp"` is present and path is correct relative to include dirs
- `'Qwen3_5TextIO' undeclared` in MTP loop → add `#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"` (already included at line 36)

- [ ] **Step 8.2: Test [39] — regression (pipeline unchanged)**

```bash
python scripts/auto_tests.py --tests 39 --models-root "C:\data\models"
```

Expected: exit code 0, coherent output.

- [ ] **Step 8.3: Test [37] — 35B-A3B via `modeling_qwen3_5.exe`, MTP on**

Run test [37] which invokes `modeling_qwen3_5.exe --mode text` with the 35B-A3B model.
`mtp_model.xml` must be present at `C:\data\models\Huggingface\Qwen3.5-35B-A3B\mtp_model.xml`.

```bash
python scripts/auto_tests.py --tests 37 --models-root "C:\data\models"
```

Expected: `[MTP] Draft model compiled`, accept rate printed to stderr (~60–80%), coherent output, exit code 0.

- [ ] **Step 8.4: Test [37] — 35B-A3B, MTP off (`--no-mtp`)**

If test [37] in `auto_tests.py` does not have a `--no-mtp` variant, run manually:

```bash
D:/chuansheng/src_code/explicit_modeling/openvino/build/bin/Release/modeling_qwen3_5.exe \
  --model "C:\data\models\Huggingface\Qwen3.5-35B-A3B" \
  --mode text --no-mtp --output-tokens 64 \
  --prompt "Hello, what is 2+2?"
```

Expected: no `[MTP]` lines, coherent output, similar throughput to pre-MTP baseline.

- [ ] **Step 8.5: Test [27] — Qwen3.5-2B via `modeling_qwen3_5.exe`, MTP on**

```bash
python scripts/auto_tests.py --tests 27 --models-root "C:\data\models"
```

`mtp_model.xml` must be present at `C:\data\models\Huggingface\Qwen3.5-2B\mtp_model.xml`.
Expected: accept rate printed, coherent output, exit code 0.

- [ ] **Step 8.6: Test [27] — Qwen3.5-2B, MTP off**

```bash
D:/chuansheng/src_code/explicit_modeling/openvino/build/bin/Release/modeling_qwen3_5.exe \
  --model "C:\data\models\Huggingface\Qwen3.5-2B" \
  --mode text --no-mtp --output-tokens 64 \
  --prompt "Hello, what is 2+2?"
```

Expected: no `[MTP]` lines, coherent output.

---

### Task 9: Commit Chunk 2

- [ ] **Step 9.1: Commit sample app changes**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino.genai
git add src/cpp/src/modeling/samples/modeling_qwen3_5.cpp
git commit -m "feat: add MTP speculative decoding to modeling_qwen3_5 sample (--mode text)"
```

---

## Quick Reference

| Variable | What it is |
|---|---|
| `past_len` | Number of tokens currently in main model KV cache (updated after each `text_request.infer()`) |
| `next_id` | Current output token being fed as input to the next step |
| `mtp_prefix_token` | `-1` = no pending sync; ≥0 = last accepted draft token needing MTP KV sync |
| `run_main_step(token)` | Sets step tensors, calls `text_request.infer()`, advances `past_len`, returns argmax |
| `decode_steps` | Count of tokens emitted in decode phase (for throughput = `decode_steps * 1000 / decode_ms`) |

### MTP KV sync explanation

After ACCEPT, at the start of the NEXT iteration:
- Main KV length = `past_len` (already advanced twice by the two `run_main_step` calls)
- MTP KV length = `past_len - 1` (only advanced once by `infer_next(draft_token, ...)`)
- The sync call `mtp_runner->infer_next(mtp_prefix_token, past_len - 1)` advances MTP KV by 1, bringing it to `past_len`
- After sync, both KV caches are aligned at `past_len` — ready for the next draft

### Cache file names

| Condition | IR cache name |
|---|---|
| `mtp_num_hidden_layers > 0`, `!use_vl` | `qwen3_5_text<quant_suffix>.xml` with `hidden_states` output |
| `mtp_num_hidden_layers == 0` OR `use_vl` | `qwen3_5_text<quant_suffix>.xml` without `hidden_states` output |

If you switch from a build without `output_hidden_states` to one with it (or vice versa),
delete the cached `qwen3_5_text*.xml/.bin` files and let the sample rebuild from weights.
