# Draft Model & Batch Verify Optimization via GPUView Profiling

**Date:** 2026-03-27
**Model:** Qwen3.5-4B, MTP N=1 batch verify (M=2)
**Hardware:** Intel Arc 140T iGPU, 32 GB shared RAM, Windows 11

---

## Background

Per-phase timing baseline (4B model, N=1, greedy, 300 tokens):

**Baseline (before optimization):**

| Phase | Avg ms/step | Notes |
|-------|-------------|-------|
| kv_sync | 8.41 | Deferred MTP infer_next (~50.5% accept rate) |
| draft | 16.85 | lm_head GEMV ~7ms, transformer ~9ms |
| verify | 111.90 | M=2 main model batch infer |
| penalty | 0.14 | CPU-side penalty processor |
| fixup | 30.40 | gda ~20ms + kv_trim ~10ms |
| mtp_kv | 0.01 | MTP KV state management |
| **total** | **167.71** | **8.88 tok/s baseline; 9.57 tok/s MTP N=1** |

**After `try_trim_seq_axis` optimization (2026-03-27):**

| Phase | Avg ms/step | Delta | Notes |
|-------|-------------|-------|-------|
| kv_sync | 9.23 | — | |
| draft | 17.60 | — | |
| verify | 117.76 | — | M=2 only 4.5% overhead vs M=1 |
| penalty | 0.15 | — | |
| fixup | 20.37 | **-10ms** | gda=20.31ms (new bottleneck), kv_trim=0.06ms ✅ |
| mtp_kv | 0.00 | — | |
| **total** | **165.10** | **-2.6ms** | **9.57 tok/s (+7.8% vs baseline)** |

Three targets for this work:
1. **Draft inference** (~17ms) — sub-kernel analysis needs GPUView
2. **Main model batch verify** (~118ms) — ✅ SDPA handles M=2 well, no action needed
3. **Dispatch overhead / fixup** (~20ms, down from 30ms) — KV trim fixed; GDA state fixup is next candidate

---

## Architecture

```
C++ ETW markers  →  GPUView timeline  →  optimization decisions
(CPU phase tags)     (GPU kernel rows)     (code changes)
```

**Files to create/modify:**

| File | Change |
|------|--------|
| `src/cpp/src/modeling/samples/etw_markers.hpp` | New thin ETW TraceLogging wrapper |
| `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` | Add `ETW_MARK(name, ...)` calls at 8 phase boundaries per batch-verify step |
| `src/cpp/src/utils.cpp` | Wire `try_trim_seq_axis` fast path in `trim_kv_cache` |
| `scripts/capture_gpuview.bat` | GPUView start/stop capture script |

---

## Section 1: ETW Marker Implementation

### `etw_markers.hpp`

Uses `TraceLoggingProvider.h` from the Windows SDK (already present at
`C:\Program Files (x86)\Windows Kits\10\...`). No additional dependency.

```cpp
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <TraceLoggingProvider.h>

TRACELOGGING_DECLARE_PROVIDER(g_mtp_etw_provider);
// Define in one .cpp:
// TRACELOGGING_DEFINE_PROVIDER(g_mtp_etw_provider, "MTPBenchmark",
//     (0xaabbccdd, 0x1122, 0x3344, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc));

#define ETW_REGISTER()   TraceLoggingRegister(g_mtp_etw_provider)
#define ETW_UNREGISTER() TraceLoggingUnregister(g_mtp_etw_provider)
#define ETW_MARK(name, ...)  TraceLoggingWrite(g_mtp_etw_provider, name, ##__VA_ARGS__)
#else
#define ETW_REGISTER()
#define ETW_UNREGISTER()
#define ETW_MARK(name, ...)
#endif
```

### Marker placement in batch-verify loop

8 markers per iteration (wrapping the 6 existing timer phases).

> **Placement note for `kv_sync_start`:** In the real loop, the deferred `infer_next()` runs inside an `if (mtp_prefix_token >= 0)` conditional at the very top of the super-step. Place `ETW_MARK("kv_sync_start")` **before** that `if` block so the bracket is never empty on non-deferred steps; `ETW_MARK("kv_sync_end")` goes immediately after the closing `}` of the conditional.

```cpp
ETW_MARK("bv_step_start", TraceLoggingUInt32(step, "step"),
                           TraceLoggingUInt32(past_len, "past_len"));
// --- kv_sync ---
ETW_MARK("kv_sync_start");
  // mtp_runner->infer_next(prefix, past_len-1);
ETW_MARK("kv_sync_end");
// --- draft ---
ETW_MARK("draft_start");
  // drafts = mtp_runner->draft_n(next_id, past_len, N);
ETW_MARK("draft_end");
// --- verify ---
ETW_MARK("verify_start");
  // run_main_verify_batch(next_id, drafts);
ETW_MARK("verify_end");
// --- penalty (CPU only, no GPU) ---
// ... penalty processing ...
// --- fixup ---
ETW_MARK("fixup_start", TraceLoggingUInt32(N - j, "trim_count"));
  // gda_state_fixup(j);
  // trim_main_kv_cache(N - j);
ETW_MARK("fixup_end");
// --- mtp_kv ---
ETW_MARK("mtp_kv_start");
  // trim MTP KV / inject hidden state
ETW_MARK("mtp_kv_end");
ETW_MARK("bv_step_end", TraceLoggingUInt32(j, "accepted_j"));
```

---

## Section 2: GPUView Capture Workflow

### `scripts/capture_gpuview.bat`

```bat
@echo off
setlocal
set GPUVIEW_DIR=C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview
set CAPTURE_DIR=%~dp0..\gpuview_captures

if not exist "%CAPTURE_DIR%" mkdir "%CAPTURE_DIR%"
cd /d "%CAPTURE_DIR%"

echo Starting GPUView capture...
echo Run the workload now. Press any key when done (or Ctrl+C to abort).
"%GPUVIEW_DIR%\log.cmd" start

pause

"%GPUVIEW_DIR%\log.cmd" stop
echo.
echo Capture saved. Opening GPUView...
"%GPUVIEW_DIR%\GPUView.exe" merged.etl
```

**Run as Administrator.**

> **If `log.cmd` is not found:** Some WPT installs ship only `xperf.exe`. Use the manual equivalent:
> ```bat
> xperf -on DiagEasy+Microsoft-Windows-DxgKrnl:0x1FF:5 -f kernel.etl
> xperf -start user -on YourProviderGUID -f user.etl
> :: ... run workload ...
> xperf -stop user -stop
> xperf -merge kernel.etl user.etl merged.etl
> gpuview merged.etl
> ```

**Test command to capture:**
```
.\modeling_qwen3_5.exe --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --prompt "repeat 10 times: what is ffmpeg?" \
  --output-tokens 50 --think 0 --temperature 0 --mtp --mtp-draft-n 1
```

---

## Section 3: Analysis Methodology

### Reading the GPUView trace

Open `merged.etl`. In the main timeline:
- **GPU Compute Engine row**: actual OpenCL kernel DMA packets
- **GPU Copy Engine row**: memory copy operations
- **Generic Events channel**: ETW markers appear as labeled vertical lines

Zoom to the region between `bv_step_start` and `bv_step_end` for one typical step.

### Target 1: Draft inference (~16ms)

**✅ MEASURED 2026-03-27**: `draft=17.60ms/step` (averaged over 31 steps, 4B model, MTP N=1).

Sub-kernel breakdown requires GPUView. Zoom `draft_start → draft_end`:

| Observation | Diagnosis | Next action |
|-------------|-----------|-------------|
| 1 large packet (~7ms) + many small | lm_head GEMV dominates | Profile lm_head INT8 vs FP16 kernel |
| Many uniform ~0.3ms packets | GDA layer dispatch per-token (21 layers × small kernels) | N=2 draft batching to amortize |
| Large gaps between packets | OCL dispatch stall | Investigate queue depth / async submission |

Note: 4B model has 21 GDA layers and 7 SDPA layers. The draft runs a single-token forward pass through all 28 layers, so most compute is GDA linear attention (which may be faster than SDPA for M=1).

### Target 2: Batch verify (~112ms)

**✅ MEASURED 2026-03-27**:

| Run | TPOT | Throughput |
|-----|------|------------|
| Baseline M=1 (no MTP) | 112.62 ms/token | 8.88 tok/s |
| MTP N=1 M=2 verify | 117.76ms verify phase | — |
| MTP N=1 end-to-end | 104.46 ms/token | **9.57 tok/s** |

M=2 verify = 117.76ms vs M=1 step = 112.62ms → **only 4.5% overhead**. SDPA handles M=2 well — the CAUSAL_KV_OFFSET bottleneck does NOT appear for the 4B model. The 7.8% throughput gain from 58.1% accept rate outweighs the verify overhead.

**GPUView still useful for sub-kernel verification**: zoom `verify_start → verify_end` to confirm SDPA packet duration is ~4ms longer at M=2 (4.5% of ~112ms ≈ 5ms), not 2×.

| Observation | Diagnosis | Next action |
|-------------|-----------|-------------|
| SDPA packet ~5ms longer at M=2 (consistent with 4.5% overhead) | SDPA handles M=2 well | No action needed |
| SDPA packet 2× longer at M=2 | Micro-SDPA CAUSAL_KV_OFFSET regression | Investigate GPU kernel change |
| Feedforward packets 2× longer | MoE activation at M=2 (would affect 35B, not 4B) | Sequential verify still optimal for 35B |

### Target 3: Dispatch overhead / fixup

**✅ MEASURED 2026-03-27** — `BV_PROF_AVG` from 31-step run:

| Phase | Before | After | Delta |
|-------|--------|-------|-------|
| kv_trim | ~10ms avg | **0.06ms** | **-10ms** ✅ |
| gda fixup | ~20ms avg | 20.31ms | unchanged |
| fixup total | ~30.40ms | **20.37ms** | **-10ms** |

`try_trim_seq_axis` is **confirmed working**: `kv_trim=0.06ms` (metadata-only, no Copy Engine work).

**New finding: gda_fixup is the remaining bottleneck (~48ms per reject step)**

Root cause: Qwen3.5-4B has 21 GDA (linear attention) layers out of 28 (`full_attention_interval=4`). On each partial-reject step, `gda_state_fixup` calls `.data()` on 42 GPU intermediate tensors (21 layers × recurrent + conv), each forcing a GPU→CPU sync. This costs ~48ms per reject step (42% of steps) = 20ms average.

GPUView will show this as 42 sequential micro-packets in the Copy Engine during `fixup_start → fixup_end` on reject steps.

**Next optimization candidate**: avoid the `.data()` GPU→CPU copies in `gda_state_fixup` — similar to `try_trim_seq_axis` but for reading GDA intermediates: inject them via a GPU-side select/copy without materializing on CPU.

---

## Section 4: `try_trim_seq_axis` Implementation

The GPU plugin API and fast path in `utils.cpp` are **already implemented**:
- `ivariable_state.hpp`: default `return false`
- `variable_state.hpp` (GPU): declaration + implementation in `variable_state.cpp`
- `multi_tensor_variable_state.hpp/cpp`: delegation override (trims KV + beam table sub-states)
- `utils.cpp` `trim_kv_cache`: fast path already wired — calls `state.try_trim_seq_axis(seq_axis, trim)` with `continue` before the slow GPU→CPU→GPU fallback

**Reference (already in production — no code change needed):**

### `variable_state.cpp` (GPU plugin)

```cpp
bool VariableState::try_trim_seq_axis(size_t seq_axis, size_t trim_count) {
    if (!m_memory)
        return false;
    auto layout = m_layout;
    if (seq_axis >= layout.get_rank())
        return false;
    auto shape = layout.get_partial_shape();
    auto dim = shape[seq_axis];
    if (!dim.is_static() || static_cast<size_t>(dim.get_length()) < trim_count)
        return false;
    const size_t new_len = static_cast<size_t>(dim.get_length()) - trim_count;
    shape[seq_axis] = new_len;
    layout.set_partial_shape(shape);
    // reinterpret_buffer: adjusts layout metadata, no data copy
    m_memory = m_memory->reinterpret_buffer(layout);
    m_layout = layout;
    return true;
}
```

### `multi_tensor_variable_state.cpp`

```cpp
bool VariableStateIndirectKVCache::try_trim_seq_axis(size_t seq_axis, size_t trim_count) {
    // Delegate to KV cache sub-state (index 0)
    auto kv_state = std::dynamic_pointer_cast<VariableState>(m_hidden_states[0]);
    if (!kv_state)
        return false;
    bool ok = kv_state->try_trim_seq_axis(seq_axis, trim_count);
    // Beam table stays unchanged (beam axis != seq axis)
    return ok;
}
```

### `utils.cpp` — fast path in `trim_kv_cache` (already wired)

```cpp
// Fast path: metadata-only O(1) trim
if (state.try_trim_seq_axis(seq_axis, trim)) {
    continue;
}
// Slow fallback: GPU→CPU→GPU copy (existing code)
```

**Expected result:** fixup phase collapses from ~30ms to ~0ms. GPUView confirms: empty Copy Engine during `fixup_start → fixup_end`.

**Validation task:** Build the GPU plugin, run the test command under GPUView capture, and confirm the Copy Engine is empty between `fixup_start` and `fixup_end` markers. If micro-packets still appear, the `try_trim_seq_axis` override is returning `false` — check `variable_state.cpp` implementation.

---

## Optimization Priority Order

1. ✅ **`try_trim_seq_axis` confirmed working** — `kv_trim=0.06ms`, KV trim eliminated
2. ✅ **Verify SDPA overhead measured** — M=2 is only 4.5% slower than M=1; no SDPA fix needed
3. ✅ **Throughput confirmed** — MTP N=1 batch verify: 9.57 tok/s (+7.8% vs 8.88 baseline)
4. **[NEXT] Eliminate GDA state fixup GPU→CPU copies (~48ms per reject step)** — use GPU-side select to avoid `.data()` on 42 intermediate tensors per reject step
5. **[OPTIONAL] Analyze draft sub-kernels in GPUView** — zoom `draft_start → draft_end` to identify lm_head vs GDA layer breakdown (still requires manual GPUView inspection)

---

## Test Command

```
D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release>
.\modeling_qwen3_5.exe --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --prompt "repeat 10 times: what is ffmpeg?" \
  --output-tokens 50 --think 0 --temperature 0 --mtp --mtp-draft-n 1
```

GPUView tool: `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview`
