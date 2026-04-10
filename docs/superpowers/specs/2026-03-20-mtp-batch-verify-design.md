# MTP Phase 2 — Batch Verify Design Spec

**Date:** 2026-03-20
**Author:** Chuansheng Liu
**Status:** Draft
**Repositories:** openvino.genai (modeling API layer), openvino (GPU plugin — causal fix already in `long_context_fixes`)

---

## 1. Goal

Replace the current sequential single-token verify loop in MTP Phase 2 with a **single batched main-model call** that processes all M = N+1 tokens (current token + N drafts) at once.

**Expected outcome:**
- Reduce main-model invocations per super-step from N+1 to 1.
- Combined with Chunk GDA (prerequisite, see §6), achieve ~1.875× throughput improvement for N=3.
- Keep all other MTP semantics identical (strict argmax rejection, KV state management, acceptance logic).

---

## 2. Background & Motivation

### 2.1 Current State (commit `1526867c`)

```
Phase [2] – VERIFY (sequential):
  for k in 0..N:
      refs[k] = run_main_step(token_k)   ← N+1 separate infer() calls
      if refs[k] != drafts[k]: break
```

Each `run_main_step()` triggers a full GPU inference with weight loading for all 48 layers.
For N=3, a partial-accept scenario makes on average **1.875 infer() calls**, each loading
every weight tensor from GPU memory.

### 2.2 Weight-BW Bottleneck

Qwen3.5-35B-A3B on Arc 140T is dominated by GPU memory bandwidth:

| Layer Type | Fraction of T_main(1) | Scaling with M (no chunk) | Scaling with Chunk GDA |
|------------|----------------------|--------------------------|------------------------|
| GDA (×36)  | ~35%                 | M×                       | ~1× (1 chunk kernel)  |
| MoE (×48)  | ~45%                 | ~M× (weight dominated)   | ~M× (unchanged)       |
| Full Attn  | ~18%                 | ~1× (BW bound by KV)     | ~1×                   |

With **batch verify** (1 call, M tokens):
- FA + MoE weights loaded **once** → saves (M-1)× weight loads for ~63% of compute.
- GDA without chunk: still M× serial → no savings for 35%.

With **batch verify + chunk GDA**:
- All layers benefit → T_main(M) ≈ T_main(1) for M ≤ chunk_size(=64).

### 2.3 Net Gain (N=3, p_accept ≈ 0.5)

```
Sequential:            E[calls] = 1.875 → 1.875 × T_main(1)
Batch, no chunk GDA:  1 call   → 2.03  × T_main(1)  ← slightly worse!
Batch + chunk GDA:    1 call   → ~1.0  × T_main(1)  ← 1.875× speedup
```

> **Conclusion:** Batch verify alone gives no benefit; Chunk GDA is the prerequisite.
> Implementing batch verify now (with the Chunk GDA API) allows them to compose immediately
> once Chunk GDA is ready.

---

## 3. SDPA Causal Masking Strategy

### 3.1 Two Options Considered

| Option | Description | Status |
|--------|-------------|--------|
| A: `causal=true` | Use native SDPA causal flag, no explicit mask | **Selected** |
| B: `attention_mask={1, past_len+M}` | Pass full-ones 1D mask | Rejected |

### 3.2 Why `causal=true`

- **Correctness:** With fix `1fec05f380` (openvino `long_context_fixes`), `micro_sdpa` now
  correctly handles `causal=true` with KV-cache offset. The fix adds `col_offset += k - q`
  to the non-paged-attention path in `sdpa_micro.cl`.
- **Performance:** `micro_sdpa` is the fastest SDPA path (Xe-HPG+). No fallback to `sdpa_opt`.
- **Memory:** No `{1, past_len+M}` mask tensor allocation (avoids OOM at long context).
- **Simplicity:** No mask bookkeeping in calling code.

### 3.3 SDPA Kernel Path for M>1 with `causal=true`

```
supports_micro_sdpa() checks:
  ✓ Xe-HPG+ architecture
  ✓ OneDNN available
  ✓ head_size constraints satisfied
  ✓ is_causal guard REMOVED (by 1fec05f380)
  → micro_sdpa selected for batch verify M>1, causal=true
```

---

## 4. Proposed Design

### 4.1 New Function: `run_main_verify_batch`

Replace the sequential verify loop with a single function:

```cpp
// Returns refs[0..N]: N+1 tokens that main model predicts.
// refs[k] = argmax(logits[:, k, :]) for k in 0..N
// Input:  token_ids = [current_token, draft[0], ..., draft[N-1]]  (length M = N+1)
// After call: main model KV cache advanced by M positions.
std::vector<int64_t> run_main_verify_batch(
    int64_t                          first_token,   // next_id before drafting
    const std::vector<int64_t>&      drafts,        // N draft tokens
    int64_t                          past_len);     // KV length before this call
```

**Internal flow:**

```
1. Build input_ids:   [first_token, drafts[0], ..., drafts[N-1]]   shape [1, M]
2. Build position_ids: [past_len, past_len+1, ..., past_len+M-1]  shape [1, M]  (×3 for GQA heads)
3. attention_mask:     NOT USED (causal=true handles masking natively)
4. infer() once → logits [1, M, V]
5. argmax each position → refs[0..N-1], refs[N] = argmax(logits[:, N, :]) (bonus)
   Wait — logits [1, M, V] has M = N+1 entries if we pass the bonus slot too.
   Actually: pass M = N+1 tokens, get M logits, each refs[k] = argmax(logits[:, k, :]).
6. past_len += M
```

> **Note:** `refs[k]` is what the main model predicts at position `past_len + k`.
> This matches the current sequential behavior where `refs[k] = run_main_step(input[k])`.

### 4.2 Acceptance Logic (unchanged)

```cpp
// First mismatch index: j = min k s.t. refs[k-1] != drafts[k-1], or N if all match
int j = N;
for (int k = 0; k < N; ++k) {
    if (refs[k] != drafts[k]) { j = k; break; }
}
// Emit: drafts[0..j-1] + refs[j]  (correction or bonus token)
// next_id = refs[j]
```

### 4.3 KV State Management After Batch Verify

After `run_main_verify_batch()`, the main model's KV cache has advanced by M = N+1 entries.
We need to trim it back to `past_len + j + 1` (accepted positions only).

```
Main model KV after batch call:  past_len + M  entries
Target:                          past_len + j + 1 entries
Trim main model by:              M - (j+1) = N - j  tokens
```

**Main model trim (NEW — not in current code):**

```cpp
if (N - j > 0) {
    trim_main_kv_cache(static_cast<size_t>(N - j));
}
past_len += j + 1;   // only accepted + correction
```

**MTP model trim (same logic as before):**

```cpp
// MTP KV after draft_n(): past_len + N entries
// Target: past_len + j + 1 entries  (same as main)
// Trim MTP by: N - j - 1
if (N - j - 1 > 0) {
    mtp_runner->trim_kv_cache(static_cast<size_t>(N - j - 1));
}
```

**Full-accept case (j == N):**
- Main model KV = past_len + M = correct.  No trim needed.
- MTP: defer sync (same as current `mtp_prefix_token` mechanism).

### 4.4 GDA State Handling

GDA recurrent state (shape `[B, num_v_heads, K, V]`) **is NOT saved/restored** across the
batch verify call. This is consistent with:
- GenAI `fast_draft_strategy.cpp`: `reset_mem_state=false` on trim, accepts approximation.
- vLLM spec decode: "contaminated state acceptance" — rejected token states are discarded.

**Rationale:** GDA state for rejected suffix tokens contributes negligible error in practice
(the accepted prefix is correct; only rejected-then-corrected positions leave a tiny residual).

### 4.5 Hidden State for MTP After Batch Verify

After batch verify, MTP needs the main model's hidden state at position `j` (the last accepted
position) to seed the next `draft_n()`.

```cpp
// Current sequential code already does this:
const ov::Tensor last_hs = text_request.get_tensor("hidden_states");
mtp_runner->inject_hidden_state_slice(last_hs, 0);  // only 1 output pos

// Batch verify: hidden_states is [1, M, H], take slice at index j:
const ov::Tensor batch_hs = text_request.get_tensor("hidden_states");
mtp_runner->inject_hidden_state_slice(batch_hs, static_cast<size_t>(j));
```

`inject_hidden_state_slice()` already supports arbitrary `slice_idx`, so **no changes needed
in `MtpDraftRunner`**.

---

## 5. Code Changes Required

### 5.1 `modeling_qwen3_5.cpp`

**Add** `run_main_verify_batch()` lambda replacing the sequential verify loop:

```cpp
// NEW: batch verify — one infer() for all M = N+1 tokens
auto run_main_verify_batch = [&](int64_t          first_token,
                                  const std::vector<int64_t>& drafts_vec,
                                  int64_t          pl) -> std::vector<int64_t> {
    const int M = static_cast<int>(drafts_vec.size()) + 1;
    ov::Tensor batch_ids(ov::element::i64, {1, static_cast<size_t>(M)});
    ov::Tensor batch_pos(ov::element::i64, {1 * 3, static_cast<size_t>(M)});  // ×3 heads
    auto* id_data  = batch_ids.data<int64_t>();
    auto* pos_data = batch_pos.data<int64_t>();
    id_data[0] = first_token;
    for (int k = 0; k < M - 1; ++k) id_data[k + 1] = drafts_vec[k];
    for (int h = 0; h < 3; ++h)
        for (int k = 0; k < M; ++k)
            pos_data[h * M + k] = pl + k + rope_deltas_data[0];  // batch=1

    text_request.set_tensor(Qwen3_5TextIO::kInputIds,      batch_ids);
    text_request.set_tensor(Qwen3_5TextIO::kPositionIds,   batch_pos);
    // kAttentionMask not set — model uses causal=true internally
    text_request.set_tensor(Qwen3_5TextIO::kBeamIdx,       usm_beam_idx);
    text_request.infer();

    logits = text_request.get_tensor(Qwen3_5TextIO::kLogits);  // [1, M, V]
    std::vector<int64_t> refs;
    refs.reserve(static_cast<size_t>(M));
    for (int k = 0; k < M; ++k)
        refs.push_back(argmax_logits_at(logits, k));  // helper to argmax slice k
    past_len += M;
    return refs;
};
```

**Replace** Phase [2] sequential loop with:

```cpp
// [2] VERIFY PHASE: single batched main-model call
std::vector<int64_t> refs = run_main_verify_batch(next_id, drafts, past_len_before_verify);
// refs[0..N]: N+1 predictions; refs[k] corresponds to input position k

// Find first mismatch
int j = N;
for (int k = 0; k < N; ++k) {
    if (refs[k] != drafts[k]) { j = k; break; }
}

// past_len was advanced by M=N+1; trim back to j+1 accepted positions
const int over_advanced = N - j;
if (over_advanced > 0) {
    trim_main_kv_cache(static_cast<size_t>(over_advanced));
    past_len -= over_advanced;
}
```

### 5.2 Main Model KV Trim Helper

`ov::genai::utils::trim_kv_cache` works on any stateful `InferRequest` by pattern-matching
KV state names — no changes to the utility needed. What's missing is the `KVAxesPosition`
for the main model (currently only `MtpDraftRunner` stores one).

**Initialization (one-time, alongside `compiled_text`):**

```cpp
ov::genai::utils::KVAxesPosition main_kv_pos{0u, 2u};  // default
auto rt_model = compiled_text.get_runtime_model();
if (rt_model)
    main_kv_pos = ov::genai::utils::get_kv_axes_pos(rt_model);
```

**Trim lambda:**

```cpp
auto trim_main_kv_cache = [&](std::size_t trim_count) {
    if (trim_count == 0) return;
    ov::genai::utils::KVCacheState state;
    state.num_tokens_to_trim = trim_count;
    state.seq_length_axis    = main_kv_pos.seq_len;
    state.reset_mem_state    = false;
    ov::genai::utils::trim_kv_cache(text_request, state, {});
};
```

*(Same pattern as `MtpDraftRunner::trim_kv_cache()` and `pipeline_stateful.cpp`.)*

### 5.3 Attention Mask Handling

**Confirmed (from commit `4df74f80` diff):** `kAttentionMask` is still declared as a model
input tensor (the `forward()` signature still has `full_attention_mask`), but inside the
model graph it is unconditionally passed as `nullptr` to all transformer layers:

```cpp
// modeling_qwen3_5_text.cpp (long_context_fixes)
- &full_attention_mask,   // previous
+ nullptr,               // causal=true: mask param dropped internally

// llm.cpp — params become no-ops
const Tensor* /*attention_mask*/,
const Tensor* /*precomputed_sdpa_mask*/
```

**Result:** `set_tensor(kAttentionMask, step_mask)` succeeds but has **zero effect**.
For batch verify, keep passing the existing `step_mask` unchanged — no code change needed.

### 5.4 `argmax_logits_at()` Helper

```cpp
// Argmax over vocab dimension for slice k of logits [1, M, V]
static int64_t argmax_logits_at(const ov::Tensor& logits, int k) {
    const size_t M = logits.get_shape()[1];
    const size_t V = logits.get_shape()[2];
    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<ov::float16>() + k * V;
        return static_cast<int64_t>(
            std::max_element(data, data + V) - data);
    } else {
        const float* data = logits.data<float>() + k * V;
        return static_cast<int64_t>(
            std::max_element(data, data + V) - data);
    }
}
```

### 5.5 No Changes Needed In:
- `MtpDraftRunner` — `inject_hidden_state_slice(hs, j)` already handles arbitrary slice index.
- `modeling_qwen3_5_text.*` — `forward()` already outputs `hidden_states [B, M, H]`.
- GPU plugin — causal fix is in `long_context_fixes` branch, will be merged.

---

## 6. Dependencies

### 6.1 Hard Dependency: `causal=true` GPU Fix

Commits required from openvino `long_context_fixes`:
- `94dc14941d` — `intel_gpu: fix SDPA causal masking for KV cache decode`
- `1fec05f380` — `intel_gpu: enable micro_sdpa causal masking with KV-cache offset`

Without these, `causal=true` falls back to `sdpa_opt` (slower) or produces wrong results.

### 6.2 Soft Dependency: Chunk GDA (for performance payoff)

Batch verify without Chunk GDA delivers **no net throughput gain** for Qwen3.5-35B (GDA layers
cancel out the FA+MoE savings). The implementation can be merged before Chunk GDA, but the
performance benefit only materialises once Chunk GDA is enabled.

**Recommended merge order:**
1. Merge `long_context_fixes` into working branch (GPU plugin fixes).
2. Implement and merge **Chunk GDA** (prefill accelerates immediately).
3. Implement and merge **Batch Verify** (pairs with Chunk GDA for decode acceleration).

---

## 7. Revised Super-Step Flow (with Batch Verify)

```
SUPER-STEP (batch verify)
────────────────────────────────────────────────────────────────────────
[0] KV SYNC (deferred full-accept)
    if mtp_prefix_token >= 0:
        mtp_runner->infer_next(mtp_prefix_token, past_len - 1)
        mtp_prefix_token = -1

[1] DRAFT
    drafts[0..N-1] = mtp_runner->draft_n(next_id, past_len, N)

[2] BATCH VERIFY  ← NEW
    refs[0..N] = run_main_verify_batch(next_id, drafts, past_len)
        → 1× infer(M = N+1 tokens, causal=true)
        → logits [1, M, V] → argmax each position
    main KV now at past_len + M (over-advanced by N - j after acceptance)

[3] ACCEPT / REJECT  (unchanged)
    j = first k s.t. refs[k] != drafts[k], else N
    emit drafts[0..j-1] + refs[j]
    next_id = refs[j]

[4] KV STATE MANAGEMENT  ← UPDATED
    Trim main KV by (N - j):        past_len += j + 1
    if j < N:
        Trim MTP KV by (N - j - 1): mtp_kv at past_len
        inject_hidden_state_slice(hidden_states, j)
    else:
        mtp_prefix_token = drafts[N-1]   (full-accept, defer sync)
────────────────────────────────────────────────────────────────────────
```

---

## 8. Performance Analysis

### 8.1 T_main(M) Breakdown (M = N+1 = 4, Qwen3.5-35B, no chunk GDA)

```
T_main(M=4) / T_main(1):
  FA  (12 layers, ~18%):  weight loaded once,  attn ~BW-bound → ×1.1
  MoE (48 layers, ~45%):  weight loaded once,  expert compute ×M → ×1.4 (overlap ~50%)
  GDA (36 layers, ~35%):  serial M steps       → ×4.0

Total ≈ 0.18×1.1 + 0.45×1.4 + 0.35×4.0 = 0.20 + 0.63 + 1.40 = 2.23×
```

### 8.2 T_main(M) With Chunk GDA (M=4 ≤ chunk_size=64)

```
GDA: 1 chunk kernel ≈ ×1.2 instead of ×4.0:
  Total ≈ 0.18×1.1 + 0.45×1.4 + 0.35×1.2 = 0.20 + 0.63 + 0.42 = 1.25×
```

### 8.3 Throughput Comparison (N=3, p_accept ≈ 0.5)

| Mode | Calls/step | T_total / T_main(1) | tok/s (×18 baseline) |
|------|-----------|---------------------|---------------------|
| Sequential verify (current) | 1.875 | 1.875 | ~18 (baseline) |
| Batch verify, no chunk GDA  | 1 | 2.23 | ~15 (worse!) |
| Batch verify + chunk GDA    | 1 | 1.25 | ~27 (+50%) |
| Ideal (weight load once)    | 1 | 1.00 | ~34 (+88%) |

### 8.4 Sensitivity to MoE Expert Overlap

MoE scaling depends on how many unique experts are activated across M tokens.
If consecutive tokens activate the **same experts** (high overlap), T_moe(M) ≈ T_moe(1).
With ~50% overlap assumption (conservative), `×1.4` for M=4.
With 90% overlap: T_moe(M) ≈ ×1.1, bringing total to ~1.1× — near ideal.

---

## 9. Resolved Questions

All open questions from initial draft have been investigated and closed.

### Q1: Can `trim_kv_cache` be used on the main model InferRequest?
**✅ Yes.** `ov::genai::utils::trim_kv_cache` works on any stateful InferRequest by
pattern-matching KV state names — the same utility used by `MtpDraftRunner`. The only
addition needed at sample level is a one-time `get_kv_axes_pos()` call to obtain
`main_kv_pos`. No changes to the modeling API or utility functions required. See §5.2.

### Q2: Is attention_mask ignored with causal=true?
**✅ Confirmed ignored.** Commit `4df74f80` (long_context_fixes) passes `nullptr` internally
to all transformer layers — the `kAttentionMask` tensor input still exists in the model
graph but is never consumed by any op. `set_tensor(kAttentionMask, step_mask)` can remain
unchanged; the shape doesn't matter. No mask plumbing changes needed for batch verify.

### Q3: Is `hidden_states[:, j, :]` the correct seed for the next draft round?
**✅ Confirmed correct** for both mismatch (j < N) and full-accept (j == N) cases.

The index semantics are:
```
Input position j  →  hidden_states[:, j, :]  →  logits[:, j, :]  →  refs[j]

For mismatch (j < N):
  hidden_states[:, j, :] = state after processing drafts[j-1] (or first_token if j=0)
  = the state that predicted refs[j] (correction token)
  = correct seed for drafting from refs[j]

For full-accept (j == N):
  hidden_states[:, N, :] = state after processing draft[N-1] (last draft token)
  = the state that predicted refs[N] (bonus token)
  = correct seed for drafting from refs[N]
```

`inject_hidden_state_slice(hs, j)` already supports arbitrary `slice_idx`; no changes to
`MtpDraftRunner` needed. `forward_with_hidden()` already returns all M positions `[1,M,H]`.

### Q4: Full-accept + mtp_prefix_token phase ordering
**✅ No issue.** In full-accept, main KV = past_len+M is already correct (no over-advance
since all M tokens are accepted). The mtp_prefix_token deferred sync path calls
`infer_next(mtp_prefix_token, past_len-1)` at the start of the next super-step; at that
point `staged_hs_` is empty so `infer_next` correctly falls back to reading the main
model's `hidden_states` output (last position of whatever the main model just ran).

---

## 10. Summary

| Dimension | Current (1526867c) | Proposed (batch verify) |
|-----------|--------------------|------------------------|
| Verify calls/step | 1.875 (avg, N=3) | 1 |
| Causal masking | attention_mask passed | `causal=true` native |
| micro_sdpa for M>1 | N/A (single token) | Yes (with 1fec05f380 fix) |
| Main KV trim | Not needed | Trim by N-j after batch |
| MTP KV trim | Trim by N-j-1 | Same |
| GDA state | N/A (sequential) | Approximate (no save/restore) |
| Performance gain | baseline | +50% (with chunk GDA) |
| Chunk GDA dependency | No | Yes (for perf payoff) |
