// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/turboquant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace turboquant {

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

namespace {

// Portable bit-population count (no compiler intrinsic assumed).
static inline int manual_popcount(unsigned int x) {
    int n = 0;
    while (x) {
        n += static_cast<int>(x & 1u);
        x >>= 1u;
    }
    return n;
}

// xorshift64 PRNG — fast, stateless, sufficient entropy for random rotations.
static inline uint64_t xorshift64(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * UINT64_C(0x2545F4914F6CDD1D);
}

// Clamp tensor values to [min_val, max_val].
// ov::op::v0::Clamp is GPU-native (maps to an elementwise kernel).
static Tensor tq_clamp(const Tensor& x, double min_val, double max_val) {
    auto node = std::make_shared<ov::op::v0::Clamp>(x.output(), min_val, max_val);
    return Tensor(node, x.context());
}

// Round to nearest integer (banker's rounding / half-to-even).
// ov::op::v5::Round is GPU-native.
static Tensor tq_round(const Tensor& x) {
    auto node = std::make_shared<ov::op::v5::Round>(
        x.output(), ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    return Tensor(node, x.context());
}

// Infer head_dim from the shape of the rotation constant [D, D].
// Falls back to -1 if the shape is dynamic (callers should treat -1 as unknown).
static int32_t infer_head_dim(const Tensor& rotation) {
    const auto& ps = rotation.output().get_partial_shape();
    if (ps.rank().is_static() && ps[0].is_static()) {
        return static_cast<int32_t>(ps[0].get_length());
    }
    return -1;
}

}  // namespace

// ---------------------------------------------------------------------------
// make_rotation_matrix
// ---------------------------------------------------------------------------

ov::Tensor make_rotation_matrix(int32_t head_dim, uint64_t seed, ov::element::Type dtype) {
    // SRHT:  R[i,j] = H[i,j] * signs[j] / sqrt(D)
    // H[i,j] = (-1)^popcount(i & j)   (Walsh–Hadamard matrix)
    // signs[j] in {-1, +1}

    const size_t D = static_cast<size_t>(head_dim);
    std::vector<float> R(D * D);

    // Generate random diagonal signs using xorshift64.
    uint64_t rng = seed ^ UINT64_C(0xBEEFCAFEDEADBEEF);
    std::vector<float> signs(D);
    for (size_t j = 0; j < D; ++j) {
        signs[j] = (xorshift64(rng) & 1ULL) ? 1.0f : -1.0f;
    }

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            const float h_ij =
                (manual_popcount(static_cast<unsigned int>(i & j)) % 2 == 0)
                    ? 1.0f
                    : -1.0f;
            R[i * D + j] = h_ij * signs[j] * inv_sqrt_d;
        }
    }

    ov::Shape shape{D, D};

    if (dtype == ov::element::f32) {
        ov::Tensor result(ov::element::f32, shape);
        std::copy(R.begin(), R.end(), result.data<float>());
        return result;
    }

    // Convert to f16 manually (ov::float16 bit-cast from IEEE 754 f16).
    ov::Tensor result(ov::element::f16, shape);
    auto* dst = result.data<ov::float16>();
    for (size_t k = 0; k < R.size(); ++k) {
        dst[k] = ov::float16(R[k]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// turboquant_encode
// ---------------------------------------------------------------------------
//
// Graph inserted per call (one pair of ops per K and per V):
//
//   x [B,H,S,D] → cast→f32 → sq-sum → sqrt → norms [B,H,S,1]
//                                                  ↓
//   x / norms → x_norm → @ R^T → x_rot → * sqrt_D → x_scaled
//                                                  ↓
//   clamp → round → clamp → cast→i8   codes [B,H,S,D]
//   norms → cast→f16                  scales [B,H,S,1]

std::pair<Tensor, Tensor> turboquant_encode(const Tensor& x,
                                             const Tensor& rotation,
                                             int bits,
                                             float clip_val) {
    // 1. Cast to FP32 for precision.
    auto x_f32 = x.to(ov::element::f32);

    // 2. Per-vector L2 norm: norms [B,H,S,1].
    //    x_norm = x / norm  ⟺  x * rsqrt(norm²)
    auto sum_sq = ops::reduce_sum(x_f32 * x_f32, -1L, /*keepdim=*/true);
    // rsqrt_norms = 1/sqrt(sum_sq + eps), shape [B,H,S,1]
    auto rsqrt_norms = (sum_sq + 1e-8f).rsqrt();
    auto x_norm = x_f32 * rsqrt_norms;  // unit-sphere vectors

    // Actual norms for scale storage: 1 / rsqrt_norms = sqrt(sum_sq + eps)
    auto norms = 1.0f / rsqrt_norms;  // [B,H,S,1]

    // 3. Random rotation: x_norm @ R^T  (tb=true transposes rotation).
    //    [B,H,S,D] @ [D,D]^T  ⟹  [B,H,S,D]
    auto rotation_f32 = rotation.to(ov::element::f32);
    auto x_rot = ops::matmul(x_norm, rotation_f32, /*ta=*/false, /*tb=*/true);

    // 4. Scale coordinates from N(0, 1/D) → N(0, 1) via multiply by sqrt(D).
    const int32_t head_dim = infer_head_dim(rotation);
    const float sqrt_d = (head_dim > 0) ? std::sqrt(static_cast<float>(head_dim)) : 1.0f;
    auto x_scaled = x_rot * sqrt_d;

    // 5. Symmetric scalar quantisation.
    //    half = 2^(bits-1) - 1   →  7 for 4-bit, 127 for 8-bit.
    const float half = static_cast<float>((1 << (bits - 1)) - 1);
    const float scale_to_int = half / clip_val;

    auto x_clamp    = tq_clamp(x_scaled, -static_cast<double>(clip_val),
                                          static_cast<double>(clip_val));
    auto x_q        = tq_round(x_clamp * scale_to_int);   // ∈ [-half, half]
    auto x_q_clamp  = tq_clamp(x_q,
                                -static_cast<double>(half),
                                 static_cast<double>(half));
    auto codes  = x_q_clamp.to(ov::element::i8);   // [B,H,S,D] i8 for all bit widths
    auto scales = norms.to(ov::element::f16);       // [B,H,S,1]

    return {codes, scales};
}

// ---------------------------------------------------------------------------
// turboquant_decode
// ---------------------------------------------------------------------------
//
// Inverse of encode:
//
//   codes [B,H,S,D] i8  → cast→f32 → * (clip/half) → / sqrt_D
//                       → @ R → x_norm_hat → * scales → output [B,H,S,D] f16

Tensor turboquant_decode(const Tensor& codes,
                          const Tensor& scales,
                          const Tensor& rotation,
                          int bits,
                          float clip_val) {
    // 1. Dequantise INT8 codes to float.
    //    codes ∈ [-half, half] → multiply by (clip_val / half) → N(0, ≈1)
    const float half = static_cast<float>((1 << (bits - 1)) - 1);
    auto codes_f32 = codes.to(ov::element::f32);
    auto x_dq      = codes_f32 * (clip_val / half);  // ~ N(0, 1) in rotated space

    // 2. Undo sqrt(D) scaling: back to ~ N(0, 1/D).
    const int32_t head_dim = infer_head_dim(rotation);
    const float sqrt_d = (head_dim > 0) ? std::sqrt(static_cast<float>(head_dim)) : 1.0f;
    auto x_dq_unscaled = x_dq / sqrt_d;

    // 3. Inverse rotation: x_rot_hat @ R
    //    Encode did x_norm @ R^T; since R is orthogonal (R^T R = I),
    //    decoding via x_rot_hat @ R recovers x_norm_hat.
    auto rotation_f32  = rotation.to(ov::element::f32);
    auto x_norm_hat    = ops::matmul(x_dq_unscaled, rotation_f32,
                                     /*ta=*/false, /*tb=*/false);  // [B,H,S,D]

    // 4. Rescale by stored L2 norms.
    auto x_hat = x_norm_hat * scales.to(ov::element::f32);  // [B,H,S,D]

    return x_hat.to(ov::element::f16);
}

// ---------------------------------------------------------------------------
// turboquant_decode_norot
// ---------------------------------------------------------------------------

Tensor turboquant_decode_norot(const Tensor& codes,
                                const Tensor& scales,
                                int bits,
                                float clip_val) {
    const float half = static_cast<float>((1 << (bits - 1)) - 1);
    const auto& ps = codes.output().get_partial_shape();
    const float sqrt_d = (ps.rank().is_static() && ps[3].is_static())
                         ? std::sqrt(static_cast<float>(ps[3].get_length()))
                         : 1.0f;
    const float fused_scale = clip_val / (half * sqrt_d);

    // i8 → f32 → dequantise → rescale. Use f32 throughout to avoid type
    // mismatches between tensor dtype and scalar constants (which OV infers as f32).
    auto codes_f32 = codes.to(ov::element::f32);
    auto x_dq      = codes_f32 * fused_scale;                          // unit-vector in rotated space
    return (x_dq * scales.to(ov::element::f32)).to(ov::element::f16);  // [B,H,S,D] f16
}

// ---------------------------------------------------------------------------
// turboquant_rotate_query
// ---------------------------------------------------------------------------

Tensor turboquant_rotate_query(const Tensor& query, const Tensor& rotation) {
    // Rotate in the input's native dtype to avoid unnecessary casts.
    auto rotation_native = rotation.to(query.dtype());
    return ops::matmul(query, rotation_native, /*ta=*/false, /*tb=*/true);
}

// ---------------------------------------------------------------------------
// parse_turboquant_kv_config_from_env
// ---------------------------------------------------------------------------

TurboQuantKVConfig parse_turboquant_kv_config_from_env() {
    TurboQuantKVConfig cfg;

    // OV_GENAI_TURBOQUANT_KV_CACHE: "1" enables TurboQuant KV cache.
    {
        const char* raw = std::getenv("OV_GENAI_TURBOQUANT_KV_CACHE");
        cfg.enabled = (raw && std::string(raw) == "1");
    }

    if (!cfg.enabled) {
        return cfg;
    }

    // OV_GENAI_TURBOQUANT_KV_BITS: integer 2–8 (default 4).
    {
        const char* raw = std::getenv("OV_GENAI_TURBOQUANT_KV_BITS");
        if (raw && raw[0] != '\0') {
            try {
                const int parsed = std::stoi(raw);
                if (parsed >= 2 && parsed <= 8) {
                    cfg.bits = parsed;
                }
            } catch (...) {}
        }
    }

    // OV_GENAI_TURBOQUANT_KV_CLIP: float (default 3.0).
    {
        const char* raw = std::getenv("OV_GENAI_TURBOQUANT_KV_CLIP");
        if (raw && raw[0] != '\0') {
            try {
                const float parsed = std::stof(raw);
                if (parsed > 0.0f) {
                    cfg.clip_val = parsed;
                }
            } catch (...) {}
        }
    }

    // OV_GENAI_TURBOQUANT_KV_SEED: uint64 (default 42).
    {
        const char* raw = std::getenv("OV_GENAI_TURBOQUANT_KV_SEED");
        if (raw && raw[0] != '\0') {
            try {
                cfg.rotation_seed = static_cast<uint64_t>(std::stoull(raw));
            } catch (...) {}
        }
    }

    return cfg;
}

}  // namespace turboquant
}  // namespace modeling
}  // namespace genai
}  // namespace ov