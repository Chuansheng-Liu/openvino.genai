// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TurboQuant: data-oblivious, GPU-acceleratable KV cache vector quantization.
//
// Implements the MSE-optimal stage of TurboQuant (arXiv:2504.19874):
//
//   Encode(x):
//     1. Per-vector L2 normalisation → unit sphere
//     2. SRHT random rotation (matmul with precomputed [D×D] constant)
//     3. Symmetric scalar uniform quantisation to INT8 per coordinate
//
//   Decode(codes, scales):
//     1. INT8 → float dequantisation
//     2. Inverse rotation (matmul with the same [D×D] constant, no transpose)
//     3. Rescale by stored L2 norm
//
// Every step maps 1-to-1 onto standard OpenVINO ops (MatMul, ReduceSum, Clamp,
// Round, Convert …) and therefore runs natively on the GPU plugin with no
// custom kernels required.
//
// GPU-specific design notes
// --------------------------
// • Rotation matrix: a precomputed FP32 [D×D] constant built once via SRHT
//   (Structured Random Hadamard Transform). On GPU, the batched MatMul fuses
//   well with the preceding normalization and scales linearly with D².
//   For typical head_dim values (64, 128) the constant is only 16–64 KB.
// • INT8 storage: the GPU plugin's ReadValue/Assign pipeline supports i8
//   tensors, halving cache memory vs. FP16 for 8-bit quantisation and
//   achieving a 4× reduction for 4-bit (two codes packed per INT8, not yet
//   implemented here — left for a future BitPack op).
// • Decode-before-SDPA: after gather+concat the full quantised KV cache is
//   decoded back to FP16 before ops::llm::sdpa.  This keeps SDPA unchanged
//   and compatible with all existing GPU/NPU paths.
// • Stage 2 (QJL inner-product correction) is left as future work once the
//   Stage 1 decode-before-SDPA baseline is validated.

#include <cstdint>
#include <utility>

#include <openvino/openvino.hpp>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace turboquant {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// @brief Configuration for TurboQuant KV cache quantisation.
struct TurboQuantKVConfig {
    /// Enable TurboQuant.  When false, append_kv_cache_turboquant behaves
    /// identically to append_kv_cache (compile-time fall-through).
    bool enabled = false;

    /// Bits for the MSE quantiser (2–8).
    /// Paper result: 4-bit → quality-neutral, 3.5 effective bits after SRHT.
    int bits = 8;

    /// Clipping range for the scalar quantiser, expressed as the number of
    /// standard deviations of the rotated-coordinate distribution N(0,1).
    /// clip_val=4.0 covers 99.994% of the distribution; wider than 3.0 to
    /// reduce outlier clipping that degrades quality at long context lengths.
    float clip_val = 4.0f;

    /// Seed for the SRHT random diagonal.  Use the same seed for encode and
    /// decode (the rotation matrix is reconstructed from the seed, not stored).
    uint64_t rotation_seed = 42;
};

// ---------------------------------------------------------------------------
// Rotation matrix builder  (host-side, call once at model-build time)
// ---------------------------------------------------------------------------

/// @brief Parse TurboQuant KV cache configuration from environment variables.
///
/// Environment variables:
///   OV_GENAI_TURBOQUANT_KV_CACHE  : "1" to enable (default: disabled)
///   OV_GENAI_TURBOQUANT_KV_BITS   : bit-width 2-8 (default: 4)
///   OV_GENAI_TURBOQUANT_KV_CLIP   : clipping sigma (default: 3.0)
///   OV_GENAI_TURBOQUANT_KV_SEED   : rotation seed uint64 (default: 42)
TurboQuantKVConfig parse_turboquant_kv_config_from_env();

// ---------------------------------------------------------------------------
// Rotation matrix builder  (host-side, call once at model-build time)
// ---------------------------------------------------------------------------

/// @brief Generate an SRHT rotation matrix as a host ov::Tensor.
///
/// R = H × diag(signs) × (1/√D)  where
///   H[i,j] = (−1)^popcount(i & j)   (unnormalised Walsh–Hadamard matrix)
///   signs[j] ∈ {−1, +1}             (drawn from xorshift64(seed))
///
/// R is orthogonal for any power-of-2 D; for other D values it forms a valid
/// random projection (column norms ≈ 1) that still decorrelates the rotated
/// coordinates in practice.
///
/// @param head_dim  Dimension D of the key/value head vectors.
/// @param seed      RNG seed (use tq_config.rotation_seed ^ layer_idx).
/// @param dtype     Element type for the returned tensor (f32 or f16).
/// @return          Host ov::Tensor with shape [D, D].
ov::Tensor make_rotation_matrix(int32_t head_dim,
                                 uint64_t seed,
                                 ov::element::Type dtype = ov::element::f32);

// ---------------------------------------------------------------------------
// Graph-level encode / decode  (insert into the OV computation graph)
// ---------------------------------------------------------------------------

/// @brief Quantise a float KV tensor to INT8 codes + FP16 per-vector scales,
///        WITHOUT applying the SRHT rotation.
///
/// Use for value vectors when the attention output unrotation is to be skipped:
///   softmax(Q_rot @ K_rot^T) @ V_orig = softmax(Q @ K^T) @ V
/// so no rotation or inverse-rotation is needed on V.
///
/// The encode pipeline is identical to turboquant_encode() but omits the
/// matmul with the rotation matrix; the decode formula (turboquant_decode_norot)
/// is unchanged.
///
/// @param x         Input [B, H, S, D] in any float dtype.
/// @param bits      Bit-width matching TurboQuantKVConfig::bits.
/// @param clip_val  Clipping range matching TurboQuantKVConfig::clip_val.
/// @return Pair (codes [B,H,S,D] i8,  scales [B,H,S,1] f16).
std::pair<Tensor, Tensor> turboquant_encode_norot(const Tensor& x,
                                                   int bits,
                                                   float clip_val);

/// @brief Quantise a float KV tensor to INT8 codes + FP16 per-vector scales.
///
/// @param x         Input [B, H, S, D] in any float dtype.
/// @param rotation  OV constant [D, D] (FP32); produced by ops::constant().
/// @param bits      Bit-width matching TurboQuantKVConfig::bits.
/// @param clip_val  Clipping range matching TurboQuantKVConfig::clip_val.
/// @return Pair (codes [B,H,S,D] i8,  scales [B,H,S,1] f16).
std::pair<Tensor, Tensor> turboquant_encode(const Tensor& x,
                                             const Tensor& rotation,
                                             int bits,
                                             float clip_val);

/// @brief Reconstruct a float KV tensor from INT8 codes + FP16 scales.
///
/// @param codes    Quantised codes [B, H, S, D] i8.
/// @param scales   Per-vector L2 norms [B, H, S, 1] f16.
/// @param rotation OV constant [D, D] (FP32); same object used in encode.
/// @param bits     Bit-width (must match encode).
/// @param clip_val Clipping range (must match encode).
/// @return         Reconstructed tensor [B, H, S, D] f16.
Tensor turboquant_decode(const Tensor& codes,
                          const Tensor& scales,
                          const Tensor& rotation,
                          int bits,
                          float clip_val);

/// @brief Dequantise codes WITHOUT applying the inverse rotation.
///
/// Returns the vector in the rotated space with original L2 magnitude:
///   output ≈ x @ R^T   (rotated-space vector with ‖x‖ restored)
///
/// Used with turboquant_rotate_query so SDPA scores are preserved:
///   Q_rot @ K_norot^T = (Q@R^T) @ (K@R^T)^T = Q @ K^T  (R orthogonal).
/// Also used for V when the caller applies output unrotation after SDPA.
///
/// @param codes    [B, H, S, D] i8.
/// @param scales   [B, H, S, 1] f16.
/// @param bits     Bit-width (must match encode).
/// @param clip_val Clipping range (must match encode).
/// @return         [B, H, S, D] f16 in rotated space.
Tensor turboquant_decode_norot(const Tensor& codes,
                                const Tensor& scales,
                                int bits,
                                float clip_val);

/// @brief Rotate a query tensor into the SRHT-rotated key space.
///
/// Computes Q_rot = Q @ R^T so Q_rot @ K_norot^T = Q @ K^T.
///
/// @param query    [B, H, S_q, D] any float dtype.
/// @param rotation [D, D] FP32 constant.
/// @return         Rotated query in the same dtype as input.
Tensor turboquant_rotate_query(const Tensor& query, const Tensor& rotation);

/// @brief Pack signed 4-bit codes [B,H,S,D] i8 → [B,H,S,D/2] i8.
///
/// Two codes per byte: byte = ((code_even + 8) & 0xF) | ((code_odd + 8) << 4).
/// D must be even. Use tq_unpack_i4 to reverse.
Tensor tq_pack_i4(const Tensor& codes, int32_t head_dim);

/// @brief Unpack packed codes [B,H,S,D/2] i8 → [B,H,S,D] f32 signed codes.
///
/// Reverses tq_pack_i4: extracts two signed 4-bit values per byte.
Tensor tq_unpack_i4(const Tensor& packed, int32_t head_dim);

}  // namespace turboquant
}  // namespace modeling
}  // namespace genai
}  // namespace ov
