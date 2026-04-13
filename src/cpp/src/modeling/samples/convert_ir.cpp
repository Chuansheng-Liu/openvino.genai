// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// convert_ir — Convert HuggingFace Qwen3.5 model to OpenVINO IR files.
//
// Pre-converts safetensors weights to OpenVINO IR (.xml + .bin) so that
// subsequent ov_serve or modeling_qwen3_5 runs skip the expensive IR build step.
//
// DFlash mode (--dflash <draft_dir>):
//   Also converts the 3 DFlash sub-models (target, context_fc, combined_draft_v2)
//   needed for speculative decoding.  All IR files are saved into the target model
//   directory so modeling_qwen3_5_dflash can load them without safetensors.
//
// Usage:
//   convert_ir --model <hf_model_dir> [--vl] [--force] [--num-layers N]
//   convert_ir --model <target_dir> --dflash <draft_dir> [--vl] [--force]
//
// Quantization is controlled by environment variables:
//   OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
//   OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
//
// Examples:
//   convert_ir --model /models/Qwen3.5-4B
//   convert_ir --model /models/Qwen3.5-9B --dflash /models/Qwen3.5-9B-DFlash
//   OV_GENAI_INFLIGHT_QUANT_MODE=int4_sym OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=32 \
//     convert_ir --model /models/Qwen3.5-9B --dflash /models/Qwen3.5-9B-DFlash --vl

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/api/model_loader.hpp"
#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "loaders/model_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

using namespace ov::genai::modeling;

// Duplicated from model_loader.cpp (file-local there).
static std::string quant_mode_token(weights::QuantizationConfig::Mode m) {
    using Mode = weights::QuantizationConfig::Mode;
    switch (m) {
        case Mode::INT4_SYM:  return "4s";
        case Mode::INT4_ASYM: return "4a";
        case Mode::INT8_SYM:  return "8s";
        case Mode::INT8_ASYM: return "8a";
        default:              return "n";
    }
}

static std::string quant_cache_suffix(const weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) return "";
    return "_q" + quant_mode_token(cfg.mode) + "_b" + quant_mode_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

static bool has_ir_pair(const std::filesystem::path& xml, const std::filesystem::path& bin) {
    return std::filesystem::exists(xml) && std::filesystem::exists(bin);
}

static double file_size_mb(const std::filesystem::path& p) {
    if (!std::filesystem::exists(p)) return 0;
    return static_cast<double>(std::filesystem::file_size(p)) / (1024.0 * 1024.0);
}

// Convert DFlash sub-models to IR.
static void convert_dflash(const std::filesystem::path& target_dir,
                           const std::filesystem::path& draft_dir,
                           bool enable_vl,
                           bool force) {
    namespace fs = std::filesystem;

    // ─── Configs ───
    auto target_cfg = models::Qwen3_5Config::from_json_file(target_dir);
    auto draft_model_cfg = ov::genai::loaders::ModelConfig::from_hf_json(draft_dir / "config.json");

    models::DFlashDraftConfig dflash_cfg;
    dflash_cfg.hidden_size         = draft_model_cfg.hidden_size;
    dflash_cfg.intermediate_size   = draft_model_cfg.intermediate_size;
    dflash_cfg.num_hidden_layers   = draft_model_cfg.num_hidden_layers;
    dflash_cfg.num_target_layers   = (draft_model_cfg.num_target_layers > 0)
                                       ? draft_model_cfg.num_target_layers
                                       : target_cfg.text.num_hidden_layers;
    dflash_cfg.num_attention_heads = draft_model_cfg.num_attention_heads;
    dflash_cfg.num_key_value_heads = draft_model_cfg.num_key_value_heads > 0
                                       ? draft_model_cfg.num_key_value_heads
                                       : draft_model_cfg.num_attention_heads;
    dflash_cfg.head_dim            = draft_model_cfg.head_dim > 0
                                       ? draft_model_cfg.head_dim
                                       : (draft_model_cfg.hidden_size / draft_model_cfg.num_attention_heads);
    dflash_cfg.block_size          = draft_model_cfg.block_size > 0 ? draft_model_cfg.block_size : 16;
    dflash_cfg.rms_norm_eps        = draft_model_cfg.rms_norm_eps;
    dflash_cfg.rope_theta          = draft_model_cfg.rope_theta;
    dflash_cfg.hidden_act          = draft_model_cfg.hidden_act;
    dflash_cfg.attention_bias      = draft_model_cfg.attention_bias;
    dflash_cfg.target_layer_ids    = draft_model_cfg.target_layer_ids;

    auto target_layer_ids = dflash_cfg.target_layer_ids.empty()
        ? models::build_target_layer_ids(dflash_cfg.num_target_layers, dflash_cfg.num_hidden_layers)
        : dflash_cfg.target_layer_ids;
    dflash_cfg.target_layer_ids = target_layer_ids;

    std::cout << "[dflash convert] block_size=" << dflash_cfg.block_size
              << "  num_draft_layers=" << dflash_cfg.num_hidden_layers
              << "  target_layer_ids=[";
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << target_layer_ids[i];
    }
    std::cout << "]" << std::endl;

    // ─── Quantization ───
    auto target_quant = weights::parse_quantization_config_from_env();
    // Draft model is always FP16 (no quantization)
    weights::QuantizationConfig draft_quant{};

    std::string qsuffix = quant_cache_suffix(target_quant);
    std::cout << "[dflash convert] target quant: "
              << (target_quant.enabled() ? ("INT4, group_size=" + std::to_string(target_quant.group_size)) : "FP16")
              << "  suffix: " << (qsuffix.empty() ? "(none)" : qsuffix) << std::endl;

    // ─── IR file paths (all saved to target model dir) ───
    std::string vl_tag = enable_vl ? "_vl" : "";
    auto dflash_target_xml = target_dir / ("qwen3_5_dflash_target" + vl_tag + qsuffix + ".xml");
    auto dflash_target_bin = target_dir / ("qwen3_5_dflash_target" + vl_tag + qsuffix + ".bin");
    auto context_fc_xml    = target_dir / ("qwen3_5_dflash_context_fc.xml");
    auto context_fc_bin    = target_dir / ("qwen3_5_dflash_context_fc.bin");
    auto combined_v2_xml   = target_dir / ("qwen3_5_dflash_combined_draft_v2" + qsuffix + ".xml");
    auto combined_v2_bin   = target_dir / ("qwen3_5_dflash_combined_draft_v2" + qsuffix + ".bin");

    // ─── Check existing ───
    if (!force) {
        bool all_exist = has_ir_pair(dflash_target_xml, dflash_target_bin) &&
                         has_ir_pair(context_fc_xml, context_fc_bin) &&
                         has_ir_pair(combined_v2_xml, combined_v2_bin);
        if (all_exist) {
            std::cout << "[dflash convert] All DFlash IR files already exist (use --force to overwrite):" << std::endl;
            std::cout << "  target:   " << dflash_target_xml << std::endl;
            std::cout << "  ctx_fc:   " << context_fc_xml << std::endl;
            std::cout << "  draft_v2: " << combined_v2_xml << std::endl;
            return;
        }
    }

    // ─── Load safetensors ───
    std::cout << "[dflash convert] Loading target safetensors from " << target_dir << " ..." << std::endl;
    auto target_data = ov::genai::safetensors::load_safetensors(target_dir);
    ov::genai::safetensors::SafetensorsWeightSource target_source(std::move(target_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer target_finalizer(
        target_quant.enabled() ? target_quant : weights::QuantizationConfig{});

    std::cout << "[dflash convert] Loading draft safetensors from " << draft_dir << " ..." << std::endl;
    auto draft_data = ov::genai::safetensors::load_safetensors(draft_dir);
    ov::genai::safetensors::SafetensorsWeightSource draft_source(std::move(draft_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer draft_finalizer(
        draft_quant.enabled() ? draft_quant : weights::QuantizationConfig{});

    // ─── 1. DFlash target model ───
    if (force || !has_ir_pair(dflash_target_xml, dflash_target_bin)) {
        std::cout << "[dflash convert] Building DFlash target model" << (enable_vl ? " (VL)" : "") << " ..." << std::endl;
        auto target_model = models::create_qwen3_5_dflash_target_model(
            target_cfg, target_layer_ids, target_source, target_finalizer, dflash_cfg.block_size, enable_vl);
        ov::serialize(target_model, dflash_target_xml.string(), dflash_target_bin.string());
        std::cout << "[dflash convert] Saved: " << dflash_target_xml
                  << "  (" << file_size_mb(dflash_target_bin) << " MB)" << std::endl;
    } else {
        std::cout << "[dflash convert] Skipping target (already exists)" << std::endl;
    }

    // ─── 2. Context FC model ───
    if (force || !has_ir_pair(context_fc_xml, context_fc_bin)) {
        std::cout << "[dflash convert] Building context_fc model ..." << std::endl;
        auto ctx_fc_model = models::create_qwen3_5_dflash_context_fc_model(
            dflash_cfg, draft_source, draft_finalizer);
        ov::serialize(ctx_fc_model, context_fc_xml.string(), context_fc_bin.string());
        std::cout << "[dflash convert] Saved: " << context_fc_xml
                  << "  (" << file_size_mb(context_fc_bin) << " MB)" << std::endl;
    } else {
        std::cout << "[dflash convert] Skipping context_fc (already exists)" << std::endl;
    }

    // ─── 3. Combined draft model V2 ───
    if (force || !has_ir_pair(combined_v2_xml, combined_v2_bin)) {
        std::cout << "[dflash convert] Building combined draft model V2 ..." << std::endl;
        auto draft_model = models::create_qwen3_5_dflash_combined_draft_model_v2(
            target_cfg, dflash_cfg, target_source, target_finalizer,
            draft_source, draft_finalizer);
        ov::serialize(draft_model, combined_v2_xml.string(), combined_v2_bin.string());
        std::cout << "[dflash convert] Saved: " << combined_v2_xml
                  << "  (" << file_size_mb(combined_v2_bin) << " MB)" << std::endl;
    } else {
        std::cout << "[dflash convert] Skipping combined_draft_v2 (already exists)" << std::endl;
    }

    // ─── Summary ───
    std::cout << "\n[dflash convert] Done! DFlash IR files saved to " << target_dir << std::endl;
    std::cout << "  target:   " << dflash_target_xml.filename() << "  ("
              << file_size_mb(dflash_target_bin) << " MB)" << std::endl;
    std::cout << "  ctx_fc:   " << context_fc_xml.filename() << "  ("
              << file_size_mb(context_fc_bin) << " MB)" << std::endl;
    std::cout << "  draft_v2: " << combined_v2_xml.filename() << "  ("
              << file_size_mb(combined_v2_bin) << " MB)" << std::endl;
}

void print_usage() {
    std::cerr << "Usage: convert_ir --model <path> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --model <path>         HuggingFace model directory (required)\n"
              << "  --dflash <draft_dir>   Also convert DFlash sub-models (target, ctx_fc, draft_v2)\n"
              << "  --vl                   Also convert vision model for VL mode\n"
              << "  --force                Overwrite existing IR files\n"
              << "  --num-layers N         Only convert first N layers (debug)\n"
              << "  --help, -h             Show this help\n"
              << "\n"
              << "Quantization env vars:\n"
              << "  OV_GENAI_INFLIGHT_QUANT_MODE       int4_sym|int4_asym|int8_sym|int8_asym\n"
              << "  OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE  128 (default)\n"
              << "\n"
              << "DFlash example:\n"
              << "  OV_GENAI_INFLIGHT_QUANT_MODE=int4_sym OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=32 \\\n"
              << "    convert_ir --model /models/Qwen3.5-9B --dflash /models/Qwen3.5-9B-DFlash\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string dflash_draft_path;
    ConvertParams params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "--dflash") && i + 1 < argc) dflash_draft_path = argv[++i];
        else if (arg == "--vl") params.enable_vision = true;
        else if (arg == "--force") params.force = true;
        else if ((arg == "--num-layers") && i + 1 < argc) params.num_layers = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_usage();
        return 1;
    }

    try {
        auto t0 = std::chrono::steady_clock::now();

        // Standard text/vision IR conversion
        ModelLoader::convert(model_path, params);

        // DFlash IR conversion (if requested)
        if (!dflash_draft_path.empty()) {
            std::cout << "\n──── DFlash IR conversion ────\n" << std::endl;
            convert_dflash(model_path, dflash_draft_path, params.enable_vision, params.force);
        }

        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "\n[convert_ir] Total time: " << sec << "s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
