# Draft Model & Batch Verify GPUView Optimization Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ETW phase markers to the MTP batch-verify loop so GPUView shows labeled CPU/GPU timeline annotations for each phase (kv_sync, draft, verify, fixup, mtp_kv), then capture a GPUView trace to confirm the `try_trim_seq_axis` fast path eliminates the 56-copy KV trim bottleneck (~30ms → ~0ms).

**Architecture:** A thin `etw_markers.hpp` header wraps `TraceLoggingWrite` behind a `#ifdef _WIN32` guard; `modeling_qwen3_5.cpp` adds 8 markers per batch-verify step around the existing `_t0`–`_t7` timer calls. A `capture_gpuview.bat` script automates the GPUView start/stop/open workflow. All three GPU plugin `.cpp` implementations (`variable_state.cpp`, `multi_tensor_variable_state.cpp`, `utils.cpp` fast path) are already in the codebase — no plugin code needs to be written.

**Tech Stack:** Windows TraceLogging ETW (`TraceLoggingProvider.h`, Windows SDK 10), C++17, Intel Arc 140T iGPU, GPUView (WPT at `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview`)

---

## Chunk 1: ETW Marker Header

**Files:**
- Create: `src/cpp/src/modeling/samples/etw_markers.hpp`

---

### Task 1: Create `etw_markers.hpp`

**Files:**
- Create: `src/cpp/src/modeling/samples/etw_markers.hpp`

The header must:
- Be guarded with `#ifdef _WIN32` so Linux/CI builds are unaffected
- Use `TraceLoggingProvider.h` (Windows SDK, no extra install needed)
- Expose: `ETW_REGISTER()`, `ETW_UNREGISTER()`, `ETW_MARK(name, ...)` macros
- Declare (not define) `g_mtp_etw_provider` — the define goes in the `.cpp` that calls `ETW_REGISTER()`

- [ ] **Step 1: Create the header file**

```cpp
// src/cpp/src/modeling/samples/etw_markers.hpp
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef _WIN32
#include <windows.h>
#include <TraceLoggingProvider.h>

// Provider declared here; defined once in the .cpp that calls ETW_REGISTER().
TRACELOGGING_DECLARE_PROVIDER(g_mtp_etw_provider);

#define ETW_REGISTER()   TraceLoggingRegister(g_mtp_etw_provider)
#define ETW_UNREGISTER() TraceLoggingUnregister(g_mtp_etw_provider)
#define ETW_MARK(name, ...)  TraceLoggingWrite(g_mtp_etw_provider, name, ##__VA_ARGS__)

#else
// No-ops on non-Windows platforms
#define ETW_REGISTER()
#define ETW_UNREGISTER()
#define ETW_MARK(name, ...)
#endif  // _WIN32
```

- [ ] **Step 2: Verify the file was written correctly**

```bash
head -30 src/cpp/src/modeling/samples/etw_markers.hpp
```

Expected: see `TRACELOGGING_DECLARE_PROVIDER` and the three macros.

---

### Task 1b: Add Advapi32 linkage to CMakeLists.txt (required on Windows)

**Files:**
- Modify: `src/cpp/src/modeling/samples/CMakeLists.txt` (after line ~60, the `target_link_libraries` block)

`TraceLoggingRegister` resolves to `EventRegister` in `Advapi32.lib`. Without this the build will fail with an unresolved external symbol. This step is **mandatory**, not a fallback.

- [ ] **Step 1: Add Advapi32 to target_link_libraries**

Find:
```cmake
target_link_libraries(modeling_qwen3_5 PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)
```
Replace with:
```cmake
target_link_libraries(modeling_qwen3_5 PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

if(WIN32)
    target_link_libraries(modeling_qwen3_5 PRIVATE Advapi32)
endif()
```

- [ ] **Step 2: Verify the change**

```bash
grep -A3 "Advapi32" src/cpp/src/modeling/samples/CMakeLists.txt
```

Expected: see the `if(WIN32)` block.

---

## Chunk 2: Wire ETW Markers into modeling_qwen3_5.cpp

**Files:**
- Modify: `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp`

Changes needed (all within `modeling_qwen3_5.cpp`):
1. Add `#include "etw_markers.hpp"` in the includes block (after existing includes, ~line 39)
2. Add `TRACELOGGING_DEFINE_PROVIDER(...)` immediately after the include (one translation unit only)
3. Add `ETW_REGISTER()` before the batch-verify `while` loop (~line 1529)
4. Add `ETW_UNREGISTER()` after the `while` loop exits
5. Add 8 `ETW_MARK` calls inside the loop at the existing `_t0`–`_t7` timestamps

---

### Task 2: Add include, provider definition, and register/unregister

**Files:**
- Modify: `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` (lines ~39, ~1529, ~1700)

- [ ] **Step 1: Add the `#include` and provider definition after the existing includes**

Find the line:
```cpp
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
```
(currently ~line 39)

Add immediately after it:
```cpp
#include "etw_markers.hpp"

#ifdef _WIN32
// Define the ETW provider once in this translation unit.
// GUID generated for MTPBenchmark provider — do not reuse for other providers.
TRACELOGGING_DEFINE_PROVIDER(
    g_mtp_etw_provider,
    "MTPBenchmark",
    (0x3d6e7c8a, 0x1f2b, 0x4e5d, 0x9a, 0x8b, 0x7c, 0x6d, 0x5e, 0x4f, 0x3a, 0x2b));
#endif
```

- [ ] **Step 2: Add `ETW_REGISTER()` before the batch-verify `while` loop**

Find the line (currently ~line 1529):
```cpp
        while (generated.size() < static_cast<size_t>(opts.max_new_tokens)) {
```

Add immediately before it:
```cpp
        ETW_REGISTER();
```

- [ ] **Step 3: Add `ETW_UNREGISTER()` after the `while` loop**

Find the line after the closing `}` of the batch-verify while loop. The loop ends around line 1700. The code after is the final `[BV_PROF_AVG]` print block. Find the pattern:
```cpp
        if (t_prof_steps > 0) {
```
Add immediately before it:
```cpp
        ETW_UNREGISTER();
```

---

### Task 3: Add ETW_MARK calls at each phase boundary

**Files:**
- Modify: `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` (lines ~1532–1657)

The 8 markers map exactly onto the existing `_t0`–`_t7` timer timestamps. Add each marker on the line immediately following its corresponding timestamp.

- [ ] **Step 1: Add step-start and kv_sync markers**

The batch-verify loop body begins with `double _t0 = now_ms();` (~line 1532).

Change (replace this block — include the blank line after `_t0` so the match is unambiguous):
```cpp
            double _t0 = now_ms();

            // [0] KV sync (deferred full-accept from previous super-step).
            // After full accept: main KV = past_len, MTP KV = past_len - 1.
```
with:
```cpp
            double _t0 = now_ms();
            ETW_MARK("bv_step_start",
                TraceLoggingUInt32(static_cast<uint32_t>(t_prof_steps), "step"),
                TraceLoggingUInt32(static_cast<uint32_t>(past_len), "past_len"));
            ETW_MARK("kv_sync_start");

            // [0] KV sync (deferred full-accept from previous super-step).
            // After full accept: main KV = past_len, MTP KV = past_len - 1.
```

- [ ] **Step 2: Add kv_sync_end / draft_start after `_t1`**

Find:
```cpp
            double _t1 = now_ms();

            // [1] DRAFT PHASE: draft N tokens sequentially via MTP
```
Replace with:
```cpp
            double _t1 = now_ms();
            ETW_MARK("kv_sync_end");
            ETW_MARK("draft_start");

            // [1] DRAFT PHASE: draft N tokens sequentially via MTP
```

- [ ] **Step 3: Add draft_end / verify_start after `_t2`**

Find:
```cpp
            double _t2 = now_ms();

            // [2] BATCH VERIFY
```
Replace with:
```cpp
            double _t2 = now_ms();
            ETW_MARK("draft_end");
            ETW_MARK("verify_start");

            // [2] BATCH VERIFY
```

- [ ] **Step 4: Add verify_end after `_t3`**

Find:
```cpp
            double _t3 = now_ms();

            // [3] PENALTY-ADJUSTED REFS
```
Replace with:
```cpp
            double _t3 = now_ms();
            ETW_MARK("verify_end");

            // [3] PENALTY-ADJUSTED REFS
```

- [ ] **Step 5: Add fixup_start / fixup_end markers**

Find:
```cpp
            double _t4 = now_ms();

            // [4] GDA STATE FIXUP + KV TRIM
```
Replace with:
```cpp
            double _t4 = now_ms();
            ETW_MARK("fixup_start",
                TraceLoggingUInt32(static_cast<uint32_t>(j < N ? N - j : 0), "trim_count"));

            // [4] GDA STATE FIXUP + KV TRIM
```

Find the line after `trim_main_kv_cache` completes:
```cpp
            double _t5 = now_ms();
            t_gda_fixup += _t4b - _t4a;
```
Replace with:
```cpp
            double _t5 = now_ms();
            ETW_MARK("fixup_end");
            t_gda_fixup += _t4b - _t4a;
```

- [ ] **Step 6: Add mtp_kv_start / mtp_kv_end / bv_step_end markers**

Find the line:
```cpp
            double _t7 = now_ms();
            t_kv_sync += _t1 - _t0;
```
Replace with:
```cpp
            double _t7 = now_ms();
            ETW_MARK("mtp_kv_end");
            ETW_MARK("bv_step_end",
                TraceLoggingUInt32(static_cast<uint32_t>(j), "accepted_j"));
            t_kv_sync += _t1 - _t0;
```

Then add `ETW_MARK("mtp_kv_start");` immediately before the `[6] MTP KV STATE MANAGEMENT` comment (line ~1641). `[5] ACCEPT / REJECT` (line ~1619) is CPU-only token emit with no GPU work; the GPU-visible MTP KV work measured by `t_mtp_kv = _t7 - _t5` begins at section `[6]`.

Find:
```cpp
            // [6] MTP KV STATE MANAGEMENT
```
Replace with:
```cpp
            ETW_MARK("mtp_kv_start");
            // [6] MTP KV STATE MANAGEMENT
```

---

### Task 4: Build the genai sample and verify it compiles

**Files:**
- Build target: `modeling_qwen3_5` in `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build`

- [ ] **Step 1: Build the sample**

Run from the genai build directory:
```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino.genai/build
cmake --build . --target modeling_qwen3_5 --config Release -j12
```

Expected: build succeeds with no errors. The `TraceLoggingProvider.h` include may produce a linker requirement for `Advapi32.lib`; if the linker complains, check the CMakeLists.txt for the sample target and add:
```cmake
target_link_libraries(modeling_qwen3_5 PRIVATE Advapi32)
```

- [ ] **Step 2: Verify binary updated**

```bash
ls -la bin/Release/modeling_qwen3_5.exe
```

Expected: timestamp is current.

- [ ] **Step 3: Quick smoke test (no GPUView)**

Run the test command in a **separate terminal** and leave it running (use 200 tokens so there is time to check the ETW provider):
```
D:/chuansheng/src_code/explicit_modeling/openvino.genai/build/bin/Release/modeling_qwen3_5.exe \
  --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --prompt "repeat 10 times: what is ffmpeg?" \
  --output-tokens 200 --think 0 --temperature 0 --mtp --mtp-draft-n 1
```

Expected: runs to completion, prints `[BV_PROF_AVG]` line, no crash.

- [ ] **Step 4: Verify the ETW provider is registered (while the exe from Step 3 is still running)**

Open another cmd window **while the exe is still generating tokens** and run:
```bat
logman query providers | findstr /i "MTPBenchmark"
```

Expected: `MTPBenchmark  {3d6e7c8a-1f2b-4e5d-...}` appears in the output.

If the provider is missing: the Advapi32 link or `TraceLoggingRegister` call is not working — check that CMakeLists.txt was patched and rebuilt.

- [ ] **Step 5: Commit**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino.genai
git add src/cpp/src/modeling/samples/etw_markers.hpp \
        src/cpp/src/modeling/samples/CMakeLists.txt \
        src/cpp/src/modeling/samples/modeling_qwen3_5.cpp
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(mtp): add ETW phase markers to batch-verify loop for GPUView profiling"
```

---

## Chunk 3: GPUView Capture Script

**Files:**
- Create: `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling\scripts\capture_gpuview.bat`

---

### Task 5: Create the capture script

**Files:**
- Create: `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling\scripts\capture_gpuview.bat`

- [ ] **Step 1: Create `capture_gpuview.bat`**

```bat
@echo off
setlocal
:: GPUView capture script for MTP batch-verify profiling.
:: Must be run as Administrator.

set GPUVIEW_DIR=C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview

:: Verify log.cmd exists before proceeding
if not exist "%GPUVIEW_DIR%\log.cmd" (
    echo ERROR: log.cmd not found at %GPUVIEW_DIR%
    echo Verify that GPUView is installed as part of the Windows Performance Toolkit.
    pause
    exit /b 1
)
set EXE_DIR=D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release
set MODEL=C:\data\models\Huggingface\Qwen3.5-4B
set CAPTURE_DIR=%~dp0..\gpuview_captures

if not exist "%CAPTURE_DIR%" mkdir "%CAPTURE_DIR%"
cd /d "%CAPTURE_DIR%"

echo.
echo === GPUView MTP Batch-Verify Capture ===
echo.
echo This will:
echo   1. Start GPUView capture
echo   2. Run modeling_qwen3_5.exe (50 tokens, MTP N=1)
echo   3. Stop capture and open GPUView
echo.
echo Press any key to start capture (run as Administrator)...
pause >nul

"%GPUVIEW_DIR%\log.cmd" start

echo.
echo Capture started. Running workload...
echo.

"%EXE_DIR%\modeling_qwen3_5.exe" ^
    --model "%MODEL%" ^
    --prompt "repeat 10 times: what is ffmpeg?" ^
    --output-tokens 50 ^
    --think 0 ^
    --temperature 0 ^
    --mtp ^
    --mtp-draft-n 1

echo.
echo Workload complete. Stopping capture...
"%GPUVIEW_DIR%\log.cmd" stop

echo.
echo Opening GPUView...
"%GPUVIEW_DIR%\GPUView.exe" merged.etl
```

- [ ] **Step 2: Test the script runs (dry run — just check it launches without error)**

Right-click `capture_gpuview.bat` → "Run as administrator".
Expected: prompts appear, `log.cmd start` succeeds (no error about missing files).

If `log.cmd start` fails: check that GPUView is present at `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview\log.cmd`. If path differs, update `GPUVIEW_DIR` in the script.

- [ ] **Step 3: Commit**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
git add scripts/capture_gpuview.bat
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(profiling): add GPUView capture script for MTP batch-verify"
```

---

## Chunk 4: Baseline GPUView Capture & Analysis

This chunk produces the first real trace. It requires running as Administrator and opening GPUView manually.

---

### Task 6: Capture baseline trace

- [ ] **Step 1: Run the capture script as Administrator**

```
Right-click scripts\capture_gpuview.bat → Run as administrator
```

The script starts `log.cmd`, runs the 50-token workload, stops capture, and opens GPUView on `merged.etl` in `scripts\..\gpuview_captures\`.

- [ ] **Step 2: In GPUView, navigate to the batch-verify region**

1. Wait for GPUView to load `merged.etl`
2. Use Ctrl+F or the timeline scrollbar to find the region where the workload runs (GPU Compute Engine row shows dense activity)
3. Zoom to one complete `bv_step_start → bv_step_end` window (~167ms wide)
4. The Generic Events channel (top row) should show labeled ETW markers

- [ ] **Step 3: Identify the fixup Copy Engine packets**

Zoom to `fixup_start → fixup_end`.

**Expected (before try_trim_seq_axis working):**
- Copy Engine row: 56 sequential micro-packets (~0.5–1ms each) → total ~30–56ms
- Compute Engine row: idle during fixup (no kernel work)
- ETW markers confirm this is exactly the `fixup` phase

**Expected (if try_trim_seq_axis already working):**
- Copy Engine row: empty (zero packets) during `fixup_start → fixup_end`
- This means `try_trim_seq_axis` is returning `true` and the metadata-only path is active
- The fixup phase collapses to near-zero

Record which case you see. Proceed to Task 7 based on the observation.

---

### Task 7: Validate try_trim_seq_axis is active

- [ ] **Step 1: Check the BV_PROF_AVG output from the capture run**

The terminal output from the test run should include a `[BV_PROF_AVG]` line. Look at the `kv_trim` field:

```
[BV_PROF_AVG] ... fixup=X.XX (gda=X.XX kv_trim=Y.YY) ...
```

- If `kv_trim` ≈ 0.0 ms → `try_trim_seq_axis` is working, fast path active
- If `kv_trim` ≈ 25–30 ms → `try_trim_seq_axis` returning `false`, falling back to slow path

- [ ] **Step 2: If kv_trim is still high — check the GPU plugin build**

The fast path and implementation are already in the source. If it's slow, the most likely cause is that the installed GPU plugin DLL is stale (built before the `try_trim_seq_axis` implementation was added).

Rebuild the GPU plugin:
```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino/build
cmake --build . --target openvino_intel_gpu_plugin --config Release -j12
```

Then re-run the workload and check `kv_trim` again.

> **Note:** If cached IR exists at `C:\data\models\Huggingface\Qwen3.5-4B\*.xml.cache`, delete it after rebuilding the GPU plugin to avoid the 0xC0000374 crash.

- [ ] **Step 3: After confirming kv_trim ≈ 0 ms, capture a second trace**

Run `capture_gpuview.bat` again. The `fixup_start → fixup_end` window in GPUView should now show an empty Copy Engine row.

Compare the two traces side by side (save `merged.etl` from the first run as `baseline.etl` before the second run).

---

## Chunk 5: Draft & Verify Analysis

After the fixup bottleneck is confirmed resolved, use the trace to analyze the remaining two targets.

---

### Task 8: Analyze draft model inference in GPUView

- [ ] **Step 1: Zoom to `draft_start → draft_end` in the trace**

The draft phase is ~16ms. Count the Compute Engine packets inside this window.

Record:
- Total number of packets
- Duration of the largest packet
- Duration of gaps between packets (dispatch stall overhead)

- [ ] **Step 2: Identify the dominant kernel**

The largest packet corresponds to the dominant kernel. For the 4B model:
- If 1 packet ≈ 7ms out of 16ms total → lm_head GEMV dominates
- If no single packet > 2ms → uniform MoE expert dispatch

**If lm_head dominates (1 large packet ~7ms):**

The lm_head for 4B is `[1,1,2560] × [151936, 2560]` INT8. The next optimization step is to profile whether switching to FP16 (simpler kernel, no dequant overhead) is faster on Arc 140T. Note this finding in the spec doc and open a separate task.

**If uniform MoE dispatch (many ~0.3ms packets):**

The bottleneck is the number of OCL kernel launches, not arithmetic. Consider drafting N=2 tokens in a single forward pass to amortize dispatch overhead. Note this finding and open a separate task.

- [ ] **Step 3: Record findings in the spec doc**

Update `docs/superpowers/specs/2026-03-27-draft-model-gpuview-optimization-design.md`, Section 3 / Target 1, with the actual observed values and the chosen next action.

---

### Task 9: Analyze batch verify SDPA in GPUView

- [ ] **Step 1: Zoom to `verify_start → verify_end`**

The verify phase is ~112ms for M=2. Identify the SDPA-related packet(s) — these are typically 1–3 large packets per attention layer.

- [ ] **Step 2: Compare against a single-token baseline**

Run a single non-MTP decode (remove `--mtp` flag) and capture a second short trace:
```
modeling_qwen3_5.exe --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --prompt "repeat 10 times: what is ffmpeg?" \
  --output-tokens 20 --think 0 --temperature 0
```

In GPUView, zoom to one decode step and measure the SDPA packet duration for M=1.

Compare M=1 SDPA duration vs M=2 SDPA duration:
- If M=2 SDPA ≈ 2× M=1 SDPA → CAUSAL_KV_OFFSET kernel path issue confirmed
- If M=2 SDPA ≈ M=1 SDPA → SDPA is not the bottleneck; feedforward MoE is

- [ ] **Step 3: Record findings in the spec doc**

Update Section 3 / Target 2 with observed SDPA durations and chosen next action.

---

## Quick Reference

### Test command
```
D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe \
  --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --prompt "repeat 10 times: what is ffmpeg?" \
  --output-tokens 50 --think 0 --temperature 0 --mtp --mtp-draft-n 1
```

### GPU plugin build command
```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino/build
cmake --build . --target openvino_intel_gpu_plugin --config Release -j12
```

### GenAI sample build command
```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino.genai/build
cmake --build . --target modeling_qwen3_5 --config Release -j12
```

### GPUView location
```
C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview\GPUView.exe
```

### Key files
| File | Purpose |
|------|---------|
| `src/cpp/src/modeling/samples/etw_markers.hpp` | ETW provider macros |
| `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` | Batch-verify loop with markers |
| `openvino-explicit-modeling/scripts/capture_gpuview.bat` | Capture automation |
| `openvino/src/plugins/intel_gpu/src/plugin/variable_state.cpp:150` | try_trim_seq_axis impl |
| `openvino.genai/src/cpp/src/utils.cpp:651` | fast path in trim_kv_cache |
