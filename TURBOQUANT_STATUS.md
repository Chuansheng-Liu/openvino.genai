# TurboQuant KV Cache — Implementation Status

## Overview

TurboQuant (TQ) is a KV cache compression technique (arXiv:2504.19874) that reduces
memory usage during inference by quantizing KV cache entries with random rotation
to minimize outlier impact. This implementation spans two repos:

- **openvino** (`tp_dev` branch): GPU plugin i4 KV cache compression, SDPA dequant kernels, oneDNN denormal fix
- **openvino.genai** (`tp_dev` branch): TQ rotation, quantization, and multiple execution paths

## Implemented Features

### openvino (12 patches)

1. **GPU native i4 KV cache compression** — fused SDPA inline dequant for i4/u4
2. **i4 SDPA correctness** — BEAM_TABLE, V_HEAD_SIZE_LEFTOVER, concat type, layout remapping, head_size fixes
3. **Sub-group quantization** — group_size=128 for finer-grained i4 scales
4. **get_state() for compressed KV** — enable variable state readback
5. **sdpa_micro for i4/u4** — nGEN micro-kernel enablement with single-token decode skip
6. **L2-norm quantization** — MSE-optimal scales + f32 unpack precision
7. **oneDNN denormal fix** — IEEE denormals in microkernel for correct s4→f16 conversion
8. **SDPA causal masking** — KV-cache offset fix for multi-token decode
9. **f16 overflow fix** — f32 accumulator in MULTI_TOKEN kernel, MAX_PARTITIONS 128→1024
10. **micro_sdpa causal re-enablement** — with corrected KV-cache offset

### openvino.genai (6 patches)

1. **TurboQuant INT8 KV cache** — SRHT rotation matrix, per-head symmetric i8 quantization,
   post-SDPA unrotation, optimized decode path, default bits=8
2. **GPU-native KV compression** — rotation + standard f16 cache + GPU plugin i8 auto-compress
3. **TurboQuant INT4 packed KV** — true 74% memory savings with packed i4 codes
4. **GPU-native i4 wiring** — GPU plugin i4 auto-compress for TQ 4-bit mode
5. **Remove V rotation from GPU-native i4** — K-only rotation for the GPU-native path
6. **KV cache robustness** — memory usage reporting + try-catch on query_state()

## Execution Paths

All paths share these common env vars:
```
OV_GPU_MOE_DISABLE_ONEDNN=1
OV_GENAI_USE_MODELING_API=1
OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym
```

### 1. Baseline (f16 KV cache) — ✅ Working
No additional env vars. Standard f16 KV cache, no compression.

### 2. GPU compress i8 (no TQ) — ✅ Working
```
OV_GENAI_GPU_KV_COMPRESS=1
```
GPU plugin auto-compresses f16 KV to i8 with fused SDPA dequant. No TQ rotation.

### 3. TQ i8 GPU-native — ✅ Working
```
OV_GENAI_TURBOQUANT_KV_CACHE=1
OV_GENAI_TURBOQUANT_KV_BITS=8
OV_GENAI_TQ_GPU_NATIVE=1
```
TQ SRHT rotation → standard f16 KV cache → GPU plugin auto-compresses to i8.

### 4. TQ INT8 explicit (CPU path) — ✅ Working
```
OV_GENAI_TURBOQUANT_KV_CACHE=1
OV_GENAI_TURBOQUANT_KV_BITS=8
```
TQ rotation → CPU-side i8 quantize/dequant → decomposed attention in graph.

### 5. TQ INT4 f16-cache — ✅ Working
```
OV_GENAI_TURBOQUANT_KV_CACHE=1
OV_GENAI_TURBOQUANT_KV_BITS=4
OV_GENAI_TQ_F16_CACHE=1
```
TQ rotation → CPU-side i4 pack for storage + f16 dequant cache → standard SDPA on f16.

### 6. TQ INT4 explicit (CPU path) — ✅ Working
```
OV_GENAI_TURBOQUANT_KV_CACHE=1
OV_GENAI_TURBOQUANT_KV_BITS=4
OV_GENAI_TQ_DECOMPOSED=1
```
TQ rotation → CPU-side i4 quantize → decomposed attention with scale fusion.

### 7. TQ i4 GPU-native — ⚠️ Quality issue
```
OV_GENAI_TURBOQUANT_KV_CACHE=1
OV_GENAI_TURBOQUANT_KV_BITS=4
OV_GENAI_TQ_GPU_NATIVE=1
```
TQ rotation → standard f16 KV → GPU plugin auto-compresses to i4 with fused SDPA dequant.
Output degenerates after ~20 tokens — pre-existing GPU i4 dequant precision issue.

## Test Results (Qwen3.5-4B, 1K prompt, 1K output tokens)

| Test | Config | Throughput | KV Cache | Quality |
|------|--------|-----------|----------|---------|
| 68 | Baseline f16 | 29.0 tok/s | 205.8 MB | ✅ Coherent |
| 67 | GPU compress i8 | 30.7 tok/s | 70.6 MB | ✅ Coherent |
| 77 | TQ i8 GPU-native | 29.6 tok/s | 70.6 MB | ✅ Coherent |
| 74 | TQ INT8 explicit | 24.1 tok/s | — | ✅ Coherent |
| 75 | TQ INT4 f16-cache | 28.9 tok/s | 68.8 MB | ✅ Coherent |
| 78 | TQ i4 GPU-native | 28.5 tok/s | 48.0 MB | ⚠️ Degenerates |

## Known Issues

- **TQ i4 GPU-native quality**: The fused i4 dequant SDPA path degenerates after ~20
  tokens. This is a pre-existing issue in the GPU plugin's i4 dequant precision, not in
  the TQ rotation/quantization logic. The CPU-side TQ i4 path (f16-cache, test 75) works
  correctly with equivalent throughput.

## Date

2026-04-02
