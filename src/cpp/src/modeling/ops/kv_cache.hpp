// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "modeling/builder_context.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/ops/turboquant.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

// ---------------------------------------------------------------------------
// Standard FP16 KV cache
// ---------------------------------------------------------------------------
// KV cache append helper.
// Example shapes (before cache):
//   keys/values: [batch, num_kv_heads, seq_len, head_dim]
// After cache append (with existing cache_len):
//   outputs: [batch, num_kv_heads, cache_len + seq_len, head_dim]
std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                          const Tensor& values,
                                          const Tensor& beam_idx,
                                          int32_t num_kv_heads,
                                          int32_t head_dim,
                                          const std::string& cache_prefix,
                                          const BuilderContext& ctx);

// ---------------------------------------------------------------------------
// TurboQuant INT8 KV cache  (GPU-optimised)
// ---------------------------------------------------------------------------
// Drop-in replacement for append_kv_cache that stores quantised KV entries.
//
// Storage per layer (replacing two FP16 Variables with four smaller Variables):
//   k_codes  [B, H, S, D]  i8   — quantised key codes
//   k_scales [B, H, S, 1]  f16  — per-vector L2 norms (key)
//   v_codes  [B, H, S, D]  i8   — quantised value codes
//   v_scales [B, H, S, 1]  f16  — per-vector L2 norms (value)
//
// Memory savings vs FP16 cache (stored as i8):
//   FP16 baseline:  2 × D × 2B  =  4D bytes/token/head
//   TurboQuant i8:  2 × D × 1B  +  2 × 1 × 2B  ≈  2D + 4 bytes/token/head
//
// "Rotate Q, unrotate output" correctness fix:
//   K and V are stored in the SRHT-rotated space. To avoid decoding all
//   cached tokens with a [S×D²] rotation matmul:
//     1. Q_rot = Q @ R^T           — rotate query (O(S_q × D²))
//     2. K, V decoded without inv-rotation matmul
//     3. attn_out @ R              — unrotate output (O(S_q × D²))
//   Correctness (R orthogonal, R^T R = I):
//     SDPA(Q@R^T, K@R^T, V@R^T) @ R = softmax(Q K^T/√D) @ V  ✓
//
// Returns (q_rotated, k_norot, v_norot, R) for ops::llm::sdpa.
// Caller must apply: attn = matmul(sdpa(q_rot,k,v), R, false, false)
// When TurboQuant is disabled, R is identity and the matmul is a no-op.
std::tuple<Tensor, Tensor, Tensor, Tensor> append_kv_cache_turboquant(
    const Tensor& query,
    const Tensor& keys,
    const Tensor& values,
    const Tensor& beam_idx,
    int32_t num_kv_heads,
    int32_t head_dim,
    const std::string& cache_prefix,
    const BuilderContext& ctx,
    const turboquant::TurboQuantKVConfig& tq_config);

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
