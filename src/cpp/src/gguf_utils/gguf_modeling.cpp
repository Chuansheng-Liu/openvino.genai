// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/serialize.hpp"

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf_weight_finalizer.hpp"
#include "gguf_utils/gguf_weight_source.hpp"
#include "gguf_utils/gguf_modeling.hpp"
#include "modeling/models/qwen3/modeling_qwen3.hpp"
#include "modeling/models/qwen3_moe/modeling_qwen3_moe.hpp"
#include "modeling/models/smollm3/modeling_smollm3.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

bool use_modeling_qwen3_dense_dummy_builder() {
    auto is_truthy = [](std::string v) {
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return v == "1" || v == "true" || v == "on" || v == "yes" ;
    };

    if (const char* v = std::getenv("OV_GENAI_USE_MODELING_API")) {
        return is_truthy(v);
    }
    return true;  // Default: enabled
}

void apply_runtime_options(const std::map<std::string, GGUFMetaData>& configs,
                           const std::shared_ptr<ov::Model>& model) {
    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
}

std::shared_ptr<ov::Model> create_language_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    // Create embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        qtypes.at("model.embed_tokens.qtype"));

    auto hidden_states = inputs_embeds;

    // Initialize RoPE
    auto rope_const = init_rope(
        std::get<int>(configs.at("head_size")),
        std::get<int>(configs.at("max_position_embeddings")),
        std::get<float>(configs.at("rope_freq_base")));

    // Get input shape components
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 3);

    // Process layers
    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    for (int i = 0; i < std::get<int>(configs.at("layer_num")); ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = layer(
            configs,
            consts,
            qtypes,
            i,
            hidden_states,
            attention_mask,
            causal_mask,
            position_ids,
            rope_const,
            beam_idx,
            batch_size,
            hidden_dim,
            cos_sin_cached,
            output_shape);

        hidden_states = new_hidden;
        causal_mask = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;

        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
    }

    // Final layer norm
    auto final_norm = make_rms_norm(
        "model.norm",
        hidden_states,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    // LM head
    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        embeddings,
        qtypes.at("lm_head.qtype"));

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name(logits, "logits");

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    // debuglog
    if (0) {
        ov::serialize(model, "full_model_original.xml", "full_model_original.bin");
    }

    // Set runtime options
    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return model;
}

} // namespace

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const bool enable_save_ov_model) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;
    ss << "Loading and unpacking model from: " << model_path;
    ov::genai::utils::print_gguf_debug_info(ss.str());
    auto [config, consts, qtypes] = load_gguf(model_path);
    auto load_finish_time = std::chrono::high_resolution_clock::now();

    ss.str("");
    ss << "Loading and unpacking model done. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish_time - start_time).count() << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    std::shared_ptr<ov::Model> model;
    const std::string model_arch = std::get<std::string>(config.at("architecture"));
    ss.str("");
    ss << "Start generating OpenVINO model...";
    ov::genai::utils::print_gguf_debug_info(ss.str());
    if (!model_arch.compare("smollm3")) {
        ov::genai::utils::print_gguf_debug_info("Using modeling API: SmolLM3 builder");
        ov::genai::modeling::models::SmolLM3Config cfg;
        cfg.architecture = std::get<std::string>(config.at("architecture"));
        cfg.hidden_size = std::get<int>(config.at("hidden_size"));
        cfg.num_hidden_layers = std::get<int>(config.at("layer_num"));
        cfg.num_attention_heads = std::get<int>(config.at("head_num"));
        cfg.num_key_value_heads = std::get<int>(config.at("head_num_kv"));
        cfg.head_dim = std::get<int>(config.at("head_size"));
        cfg.rope_theta = std::get<float>(config.at("rope_freq_base"));
        cfg.rms_norm_eps = std::get<float>(config.at("rms_norm_eps"));
        cfg.attention_bias = consts.count("model.layers[0].self_attn.q_proj.bias") > 0;
        cfg.mlp_bias = consts.count("model.layers[0].mlp.gate_proj.bias") > 0;
        cfg.tie_word_embeddings = consts.count("lm_head.weight") == 0;
        if (config.count("no_rope_layer_interval")) {
            cfg.no_rope_layer_interval = std::get<int>(config.at("no_rope_layer_interval"));
        }
        if (config.count("no_rope_layers")) {
            cfg.no_rope_layers = std::get<std::vector<int32_t>>(config.at("no_rope_layers"));
        }
        ov::genai::gguf::GGUFWeightSource source(consts);
        ov::genai::gguf::GGUFWeightFinalizer finalizer(consts, qtypes);
        model = ov::genai::modeling::models::create_smollm3_model(cfg, source, finalizer);
        apply_runtime_options(config, model);
    } else if (!model_arch.compare("qwen3") && use_modeling_qwen3_dense_dummy_builder()) {
        ov::genai::utils::print_gguf_debug_info("Using modeling API: qwen3 dense dummy builder");
        ov::genai::modeling::models::Qwen3DenseConfig cfg;
        cfg.architecture = std::get<std::string>(config.at("architecture"));
        cfg.hidden_size = std::get<int>(config.at("hidden_size"));
        cfg.num_hidden_layers = std::get<int>(config.at("layer_num"));
        cfg.num_attention_heads = std::get<int>(config.at("head_num"));
        cfg.num_key_value_heads = std::get<int>(config.at("head_num_kv"));
        cfg.head_dim = std::get<int>(config.at("head_size"));
        cfg.rope_theta = std::get<float>(config.at("rope_freq_base"));
        cfg.attention_bias = consts.count("model.layers[0].self_attn.q_proj.bias") > 0;
        cfg.rms_norm_eps = std::get<float>(config.at("rms_norm_eps"));
        cfg.tie_word_embeddings = consts.count("lm_head.weight") == 0;
        ov::genai::gguf::GGUFWeightSource source(consts);
        ov::genai::gguf::GGUFWeightFinalizer finalizer(consts, qtypes);
        model = ov::genai::modeling::models::create_qwen3_dense_model(cfg, source, finalizer);
        apply_runtime_options(config, model);
    } else if (!model_arch.compare("qwen3moe") && use_modeling_qwen3_dense_dummy_builder()) {
        ov::genai::utils::print_gguf_debug_info("Using modeling API: qwen3 moe builder");
        ov::genai::modeling::models::Qwen3MoeConfig cfg;
        cfg.architecture = std::get<std::string>(config.at("architecture"));
        cfg.hidden_size = std::get<int>(config.at("hidden_size"));
        cfg.num_hidden_layers = std::get<int>(config.at("layer_num"));
        cfg.num_attention_heads = std::get<int>(config.at("head_num"));
        cfg.num_key_value_heads = std::get<int>(config.at("head_num_kv"));
        cfg.head_dim = std::get<int>(config.at("head_size"));
        cfg.intermediate_size = std::get<int>(config.at("moe_inter_size"));
        cfg.rope_theta = std::get<float>(config.at("rope_freq_base"));
        cfg.attention_bias = consts.count("model.layers[0].self_attn.q_proj.bias") > 0;
        cfg.rms_norm_eps = std::get<float>(config.at("rms_norm_eps"));
        cfg.tie_word_embeddings = consts.count("lm_head.weight") == 0;
        cfg.expert_count = std::get<int>(config.at("expert_count"));
        cfg.expert_used_count = std::get<int>(config.at("expert_used_count"));
        cfg.moe_intermediate_size = std::get<int>(config.at("moe_inter_size"));
        ov::genai::gguf::GGUFWeightSource source(consts);
        ov::genai::gguf::GGUFWeightFinalizer finalizer(consts, qtypes);
        model = ov::genai::modeling::models::create_qwen3_moe_model(cfg, source, finalizer);
        apply_runtime_options(config, model);
    } else if (!model_arch.compare("llama") || !model_arch.compare("qwen2") || !model_arch.compare("qwen3") || !model_arch.compare("qwen3moe")) {
        model = create_language_model(config, consts, qtypes);
        if (enable_save_ov_model){
            std::filesystem::path gguf_model_path(model_path);
            std::filesystem::path save_path = gguf_model_path.parent_path() / "openvino_model.xml";
            ov::genai::utils::save_openvino_model(model, save_path.string(), true);
        }
    } else {
        OPENVINO_THROW("Unsupported model architecture '", model_arch, "'");
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - load_finish_time).count();
    ss.str("");
    ss << "Model generation done. Time: " << duration << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    return model;
}
