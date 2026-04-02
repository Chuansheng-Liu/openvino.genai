// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/kv_cache.hpp"

#include <iostream>
#include <regex>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {

namespace {

/**
 * @brief Extract layer index from cache_prefix for NPUW-compatible Variable naming.
 * 
 * The StatefulToStateless pass in OpenVINO expects Variable names in a specific format:
 *   past_key_values.<layer_idx>.key + present.<layer_idx>.key
 * 
 * This function extracts the layer index from prefixes like:
 *   "model.layers[15].self_attn" -> 15
 *   "layers[0].attention" -> 0
 * 
 * @param cache_prefix The module path prefix (e.g., "model.layers[15].self_attn")
 * @return The extracted layer index, or -1 if not found
 */
int extract_layer_index(const std::string& cache_prefix) {
    // Match patterns like "layers[15]" or "layers.15"
    static const std::regex layer_pattern(R"(layers[\[\.](\d+)[\]\.]?)");
    std::smatch match;
    if (std::regex_search(cache_prefix, match, layer_pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;  // Not found - will use original naming
}

/**
 * @brief Generate NPUW-compatible Variable names for KV cache.
 * 
 * NPUW's StatefulToStateless pass expects names like:
 *   past_key_values.N.keypresent.N.key  (for keys)
 *   past_key_values.N.valuepresent.N.value  (for values)
 * 
 * @param layer_idx The layer index (0-based)
 * @param is_key True for key cache, false for value cache
 * @return The Variable name compatible with StatefulToStateless
 */
std::string make_npuw_variable_name(int layer_idx, bool is_key) {
    const std::string type = is_key ? "key" : "value";
    return "past_key_values." + std::to_string(layer_idx) + "." + type +
           "present." + std::to_string(layer_idx) + "." + type;
}

}  // namespace

std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                          const Tensor& values,
                                          const Tensor& beam_idx,
                                          int32_t num_kv_heads,
                                          int32_t head_dim,
                                          const std::string& cache_prefix,
                                          const BuilderContext& ctx) {
    auto* op_ctx = keys.context();
    auto batch = shape::dim(keys, 0);
    auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim_vec = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim)});
    auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim_vec});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, cache_shape);
    auto v_init = shape::broadcast_to(zero, cache_shape);

    // Generate NPUW-compatible Variable names for KV cache
    // StatefulToStateless pass expects: past_key_values.N.keypresent.N.key format
    std::string k_name, v_name;
    int layer_idx = extract_layer_index(cache_prefix);
    if (layer_idx >= 0) {
        // Use NPUW-compatible naming for layers with index
        k_name = make_npuw_variable_name(layer_idx, true);
        v_name = make_npuw_variable_name(layer_idx, false);
    } else {
        // Fallback to original naming for non-indexed caches
        k_name = cache_prefix + ".key_cache";
        v_name = cache_prefix + ".value_cache";
    }

    // Create Variable with dynamic shapes
    ov::PartialShape var_shape{-1, num_kv_heads, -1, head_dim};
    ov::op::util::VariableInfo k_info{var_shape, keys.dtype(), k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{var_shape, values.dtype(), v_name};
    auto v_var = std::make_shared<ov::op::util::Variable>(v_info);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_init.output(), v_var);

    auto k_cached = ops::gather(Tensor(k_read->output(0), op_ctx), beam_idx, 0);
    auto v_cached = ops::gather(Tensor(v_read->output(0), op_ctx), beam_idx, 0);

    auto k_combined = ops::concat({k_cached, keys}, 2);
    auto v_combined = ops::concat({v_cached, values}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined.output(), k_var);
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined.output(), v_var);
    ctx.register_sink(k_assign);
    ctx.register_sink(v_assign);

    return {k_combined, v_combined};
}

// ---------------------------------------------------------------------------
// append_kv_cache_turboquant
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor, Tensor> append_kv_cache_turboquant(
    const Tensor& query,
    const Tensor& keys,
    const Tensor& values,
    const Tensor& beam_idx,
    int32_t num_kv_heads,
    int32_t head_dim,
    const std::string& cache_prefix,
    const BuilderContext& ctx,
    const turboquant::TurboQuantKVConfig& tq_config) {

    auto* op_ctx = keys.context();

    // Fall back to standard FP16 cache when TurboQuant is disabled.
    // Return an invalid (default-constructed) rotation tensor as sentinel —
    // the caller checks tq_config.enabled before using it.
    if (!tq_config.enabled) {
        auto [k, v] = append_kv_cache(keys, values, beam_idx, num_kv_heads, head_dim,
                                      cache_prefix, ctx);
        return {query, k, v, Tensor{}};
    }
    const int layer_idx = extract_layer_index(cache_prefix);

    // ------------------------------------------------------------------
    // Build the SRHT rotation matrix constant.
    // Use a per-layer seed so each layer gets an independent rotation;
    // Fibonacci hashing spreads the seeds well across the seed space.
    // ------------------------------------------------------------------
    const uint64_t layer_seed =
        tq_config.rotation_seed ^
        (layer_idx >= 0
             ? (static_cast<uint64_t>(layer_idx) * UINT64_C(0x9E3779B97F4A7C15))
             : UINT64_C(0));

    ov::Tensor rotation_host =
        turboquant::make_rotation_matrix(head_dim, layer_seed, ov::element::f32);
    // Embed as a shared constant in the OV graph ([D,D] FP32, ~64 KB for D=128).
    auto rotation = ops::constant(rotation_host, op_ctx);

    // ------------------------------------------------------------------
    // Encode incoming keys and values.
    // ------------------------------------------------------------------
    auto [k_codes, k_scales] = turboquant::turboquant_encode(
        keys, rotation, tq_config.bits, tq_config.clip_val);
    auto [v_codes, v_scales] = turboquant::turboquant_encode(
        values, rotation, tq_config.bits, tq_config.clip_val);

    // ------------------------------------------------------------------
    // Variable shapes.
    // codes : [batch, num_kv_heads, seq_len, head_dim]  i8
    // scales: [batch, num_kv_heads, seq_len,          1] f16
    // GPU plugin only supports i8/f16 for dynamic Variable tensors.
    // For 4-bit, codes use range [-7, 7] stored in i8 (quantisation noise
    // benefit of 4-bit, without packing — packing requires custom ops).
    // ------------------------------------------------------------------
    auto batch        = shape::dim(keys, 0);
    auto kv_heads_vec = ops::const_vec(op_ctx,
                            std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto zero_len     = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim_vec = ops::const_vec(op_ctx,
                            std::vector<int64_t>{static_cast<int64_t>(head_dim)});
    auto one_vec      = ops::const_vec(op_ctx, std::vector<int64_t>{1});

    auto codes_shape  = shape::make({batch, kv_heads_vec, zero_len, head_dim_vec});
    auto scales_shape = shape::make({batch, kv_heads_vec, zero_len, one_vec});

    // Zero-initialised empty tensors for ReadValue.
    auto zero_i8  = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(ov::element::i8);
    auto zero_f16 = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(ov::element::f16);
    auto k_codes_init  = shape::broadcast_to(zero_i8,  codes_shape);
    auto k_scales_init = shape::broadcast_to(zero_f16, scales_shape);
    auto v_codes_init  = shape::broadcast_to(zero_i8,  codes_shape);
    auto v_scales_init = shape::broadcast_to(zero_f16, scales_shape);

    // ------------------------------------------------------------------
    // Variable names.
    // We extend the NPUW "past_key_values.N.key" convention with _quant
    // and _scale suffixes so the StatefulToStateless pass ignores them
    // (it only patterns-matches the un-suffixed names).
    // ------------------------------------------------------------------
    auto make_tq_name = [&](bool is_key, const std::string& suffix) -> std::string {
        if (layer_idx >= 0) {
            const std::string t = is_key ? "key" : "value";
            return "past_key_values." + std::to_string(layer_idx) + "." + t + suffix +
                   "present." + std::to_string(layer_idx) + "." + t + suffix;
        }
        const std::string base = cache_prefix + (is_key ? ".key_cache" : ".value_cache");
        return base + suffix;
    };

    const std::string kc_name = make_tq_name(true,  "_quant");
    const std::string ks_name = make_tq_name(true,  "_scale");
    const std::string vc_name = make_tq_name(false, "_quant");
    const std::string vs_name = make_tq_name(false, "_scale");

    const ov::PartialShape codes_var_shape {-1, num_kv_heads, -1, head_dim};
    const ov::PartialShape scales_var_shape{-1, num_kv_heads, -1, 1};

    // Keys — codes
    ov::op::util::VariableInfo kc_info{codes_var_shape, ov::element::i8, kc_name};
    auto kc_var  = std::make_shared<ov::op::util::Variable>(kc_info);
    auto kc_read = std::make_shared<ov::op::v6::ReadValue>(k_codes_init.output(), kc_var);

    // Keys — scales
    ov::op::util::VariableInfo ks_info{scales_var_shape, ov::element::f16, ks_name};
    auto ks_var  = std::make_shared<ov::op::util::Variable>(ks_info);
    auto ks_read = std::make_shared<ov::op::v6::ReadValue>(k_scales_init.output(), ks_var);

    // Values — codes
    ov::op::util::VariableInfo vc_info{codes_var_shape, ov::element::i8, vc_name};
    auto vc_var  = std::make_shared<ov::op::util::Variable>(vc_info);
    auto vc_read = std::make_shared<ov::op::v6::ReadValue>(v_codes_init.output(), vc_var);

    // Values — scales
    ov::op::util::VariableInfo vs_info{scales_var_shape, ov::element::f16, vs_name};
    auto vs_var  = std::make_shared<ov::op::util::Variable>(vs_info);
    auto vs_read = std::make_shared<ov::op::v6::ReadValue>(v_scales_init.output(), vs_var);

    // ------------------------------------------------------------------
    // Gather cached entries along batch dim (beam search support).
    // ------------------------------------------------------------------
    auto kc_cached = ops::gather(Tensor(kc_read->output(0), op_ctx), beam_idx, 0);
    auto ks_cached = ops::gather(Tensor(ks_read->output(0), op_ctx), beam_idx, 0);
    auto vc_cached = ops::gather(Tensor(vc_read->output(0), op_ctx), beam_idx, 0);
    auto vs_cached = ops::gather(Tensor(vs_read->output(0), op_ctx), beam_idx, 0);

    // ------------------------------------------------------------------
    // Append new quantised tokens along the sequence dimension (axis 2).
    // ------------------------------------------------------------------
    auto kc_combined = ops::concat({kc_cached, k_codes},  2);
    auto ks_combined = ops::concat({ks_cached, k_scales}, 2);
    auto vc_combined = ops::concat({vc_cached, v_codes},  2);
    auto vs_combined = ops::concat({vs_cached, v_scales}, 2);

    // ------------------------------------------------------------------
    // Write updated quantised cache back to Variables.
    // ------------------------------------------------------------------
    auto kc_assign = std::make_shared<ov::opset13::Assign>(kc_combined.output(), kc_var);
    auto ks_assign = std::make_shared<ov::opset13::Assign>(ks_combined.output(), ks_var);
    auto vc_assign = std::make_shared<ov::opset13::Assign>(vc_combined.output(), vc_var);
    auto vs_assign = std::make_shared<ov::opset13::Assign>(vs_combined.output(), vs_var);
    ctx.register_sink(kc_assign);
    ctx.register_sink(ks_assign);
    ctx.register_sink(vc_assign);
    ctx.register_sink(vs_assign);

    // Rotate Q into the SRHT-rotated space (O(S_q × D²), cheap for S_q=1).
    auto q_rot = turboquant::turboquant_rotate_query(query, rotation);

    // K and V: dequantise + rescale only (no inverse-rotation matmul).
    // Results ≈ K @ R^T and V @ R^T with original magnitudes.
    auto k_out = turboquant::turboquant_decode_norot(
        kc_combined, ks_combined, tq_config.bits, tq_config.clip_val)
        .to(keys.dtype());
    auto v_out = turboquant::turboquant_decode_norot(
        vc_combined, vs_combined, tq_config.bits, tq_config.clip_val)
        .to(values.dtype());

    // Return rotation so caller can unrotate SDPA output: attn @ R.
    return {q_rot, k_out, v_out, rotation};
}

}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
