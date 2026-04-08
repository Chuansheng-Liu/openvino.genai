// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/api/model_loader.hpp"

#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include <openvino/op/constant.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

namespace ov::genai::modeling {

namespace {

// Name used for the extra pos_embed Result in cached vision IR
static constexpr const char* kPosEmbedCacheResultName = "__pos_embed_cache__";

bool has_safetensors_file(const std::filesystem::path& model_dir) {
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            return true;
        }
    }
    return false;
}

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool has_model_input_name(const std::shared_ptr<ov::Model>& model, const std::string& input_name) {
    if (!model) return false;
    for (const auto& input : model->inputs()) {
        if (input.get_names().count(input_name) > 0) return true;
    }
    return false;
}

bool is_vl_text_ir_compatible(const std::shared_ptr<ov::Model>& model) {
    return has_model_input_name(model, models::Qwen3_5TextIO::kVisualEmbeds) &&
           has_model_input_name(model, models::Qwen3_5TextIO::kVisualPosMask);
}

std::string quant_mode_token(weights::QuantizationConfig::Mode mode) {
    using Mode = weights::QuantizationConfig::Mode;
    switch (mode) {
        case Mode::INT4_SYM:  return "4s";
        case Mode::INT4_ASYM: return "4a";
        case Mode::INT8_SYM:  return "8s";
        case Mode::INT8_ASYM: return "8a";
        default:              return "n";
    }
}

std::string quant_cache_suffix(const weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) return "";
    return "_q" + quant_mode_token(cfg.mode) + "_b" + quant_mode_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

std::string resolve_pos_embed_name(weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight",
    };
    for (const auto& name : candidates) {
        if (source.has(name)) return name;
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) return name;
    }
    throw std::runtime_error("Failed to locate visual.pos_embed.weight");
}

void embed_pos_embed_in_vision_model(std::shared_ptr<ov::Model>& model, const ov::Tensor& pos_embed) {
    auto constant = std::make_shared<ov::op::v0::Constant>(pos_embed);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    result->set_friendly_name(kPosEmbedCacheResultName);
    model->add_results({result});
}

ov::Tensor extract_pos_embed_from_vision_model(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == kPosEmbedCacheResultName) {
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                result->input_value(0).get_node_shared_ptr());
            if (!const_node) {
                throw std::runtime_error("pos_embed cache result is not a Constant");
            }
            ov::Tensor tensor(const_node->get_element_type(), const_node->get_shape());
            std::memcpy(tensor.data(), const_node->get_data_ptr(), tensor.get_byte_size());
            model->remove_result(result);
            return tensor;
        }
    }
    throw std::runtime_error("Cached vision IR does not contain pos_embed data");
}

std::set<int64_t> resolve_stop_token_ids(const std::filesystem::path& model_dir,
                                          const ov::genai::Tokenizer* tokenizer) {
    std::set<int64_t> ids;
    if (!model_dir.empty()) {
        const auto gen_cfg_path = model_dir / "generation_config.json";
        if (std::filesystem::exists(gen_cfg_path) && std::filesystem::is_regular_file(gen_cfg_path)) {
            try {
                ov::genai::GenerationConfig gen_config(gen_cfg_path);
                ids = gen_config.stop_token_ids;
            } catch (const std::exception& e) {
                std::cerr << "[ModelLoader] Failed to load generation_config.json: " << e.what() << std::endl;
            }
        }
    }
    if (ids.empty() && tokenizer) {
        const int64_t eos = tokenizer->get_eos_token_id();
        if (eos >= 0) ids.insert(eos);
    }
    return ids;
}

/// Try to get the GPU RemoteContext from a CompiledModel.
std::optional<ov::RemoteContext> try_get_gpu_context(ov::CompiledModel& compiled) {
    try {
        return compiled.get_context();
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace

// ─── Impl ───

struct ModelLoader::Impl {
    std::filesystem::path model_dir_;
    std::string device_;
    models::Qwen3_5Config cfg_;
    models::Qwen3_5VisionPreprocessConfig pre_cfg_;
    weights::QuantizationConfig text_quant_;
    weights::QuantizationConfig vision_quant_;

    ov::CompiledModel compiled_text_;
    std::optional<ov::CompiledModel> compiled_vision_;
    std::optional<ov::RemoteContext> gpu_ctx_;
    ov::Tensor pos_embed_weight_;
    std::unique_ptr<ov::genai::Tokenizer> tokenizer_;
    std::set<int64_t> stop_token_ids_;

    void load(const std::filesystem::path& model_dir, const LoadParams& params) {
        model_dir_ = model_dir;
        device_ = params.device;

        // ─── Validate model directory ───
        if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
            throw std::runtime_error("Model directory does not exist: " + model_dir.string());
        }
        if (!std::filesystem::exists(model_dir / "config.json")) {
            throw std::runtime_error("Model directory missing config.json: " + model_dir.string());
        }
        if (!has_safetensors_file(model_dir)) {
            throw std::runtime_error("Model directory missing .safetensors files: " + model_dir.string());
        }

        // ─── Load config ───
        cfg_ = models::Qwen3_5Config::from_json_file(model_dir);
        if (params.num_layers.has_value()) {
            int nl = *params.num_layers;
            if (nl <= 0 || nl > cfg_.text.num_hidden_layers) {
                throw std::runtime_error("num_layers must be in [1, " +
                    std::to_string(cfg_.text.num_hidden_layers) + "], got: " + std::to_string(nl));
            }
            cfg_.text.num_hidden_layers = nl;
            if (!cfg_.text.layer_types.empty()) {
                if (cfg_.text.layer_types.size() >= static_cast<size_t>(nl)) {
                    cfg_.text.layer_types.resize(static_cast<size_t>(nl));
                } else {
                    cfg_.text.layer_types.clear();
                }
            }
            cfg_.finalize();
            cfg_.validate();
        }

        // ─── Quantization config ───
        auto shared_quant = params.quant_config;
        if (!shared_quant.enabled()) {
            shared_quant = weights::parse_quantization_config_from_env();
        }
        if (shared_quant.enabled() && shared_quant.group_size <= 0) {
            throw std::runtime_error("Quantization group_size must be > 0");
        }
        text_quant_ = shared_quant;
        // Vision quantization disabled: GPU compile hangs with many INT4 dequant subgraphs
        vision_quant_ = weights::QuantizationConfig{};

        // ─── Preprocess config ───
        const auto pre_cfg_path = model_dir / "preprocessor_config.json";
        if (std::filesystem::exists(pre_cfg_path)) {
            pre_cfg_ = models::Qwen3_5VisionPreprocessConfig::from_json_file(pre_cfg_path);
        }
        if (params.max_pixels > 0) {
            pre_cfg_.max_pixels = static_cast<int64_t>(params.max_pixels);
        }

        // ─── IR cache paths ───
        const bool use_vl = params.enable_vision;
        std::string text_ir_stem = (use_vl ? "qwen3_5_text_vl" : "qwen3_5_text") + quant_cache_suffix(text_quant_);
        if (params.num_layers.has_value()) {
            text_ir_stem += "_l" + std::to_string(*params.num_layers);
        }
        std::string vision_ir_stem = "qwen3_5_vision" + quant_cache_suffix(vision_quant_);
        const auto text_xml = model_dir / (text_ir_stem + ".xml");
        const auto text_bin = model_dir / (text_ir_stem + ".bin");
        const auto vision_xml = model_dir / (vision_ir_stem + ".xml");
        const auto vision_bin = model_dir / (vision_ir_stem + ".bin");

        const bool load_text_from_ir = params.cache_ir && has_ir_model_pair(text_xml, text_bin);
        const bool load_vision_from_ir = params.cache_ir && use_vl && has_ir_model_pair(vision_xml, vision_bin);

        // ─── Weight source (lazy) ───
        ov::Core core;
        std::unique_ptr<weights::WeightSource> source;
        auto ensure_source = [&]() -> weights::WeightSource& {
            if (!source) {
                auto data = ov::genai::safetensors::load_safetensors(model_dir);
                source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));
            }
            return *source;
        };

        // ─── Vision model ───
        std::shared_ptr<ov::Model> vision_model;
        if (load_vision_from_ir) {
            std::cout << "[ModelLoader] Reusing cached vision IR: " << vision_xml << std::endl;
            vision_model = core.read_model(vision_xml.string(), vision_bin.string());
            pos_embed_weight_ = extract_pos_embed_from_vision_model(vision_model);
        } else if (use_vl) {
            auto& ws = ensure_source();
            ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_);
            vision_model = models::create_qwen3_5_vision_model(cfg_, ws, finalizer);
            if (params.cache_ir) {
                const std::string pe_name = resolve_pos_embed_name(ws);
                pos_embed_weight_ = ws.get_tensor(pe_name);
                embed_pos_embed_in_vision_model(vision_model, pos_embed_weight_);
                ov::serialize(vision_model, vision_xml.string(), vision_bin.string());
                std::cout << "[ModelLoader] Saved vision IR: " << vision_xml << std::endl;
                for (const auto& result : vision_model->get_results()) {
                    if (result->get_friendly_name() == kPosEmbedCacheResultName) {
                        vision_model->remove_result(result);
                        break;
                    }
                }
            } else {
                const std::string pe_name = resolve_pos_embed_name(ws);
                pos_embed_weight_ = ws.get_tensor(pe_name);
            }
        }

        // ─── Text model ───
        std::shared_ptr<ov::Model> text_model;
        if (load_text_from_ir) {
            std::cout << "[ModelLoader] Reusing cached text IR: " << text_xml << std::endl;
            text_model = core.read_model(text_xml.string(), text_bin.string());
            if (use_vl && !is_vl_text_ir_compatible(text_model)) {
                std::cout << "[ModelLoader] Cached text IR not VL-compatible, rebuilding" << std::endl;
                text_model.reset();
            }
        }
        if (!text_model) {
            auto& ws = ensure_source();
            ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_);
            text_model = models::create_qwen3_5_text_model(cfg_, ws, finalizer, false, use_vl);
            if (params.cache_ir) {
                ov::serialize(text_model, text_xml.string(), text_bin.string());
                std::cout << "[ModelLoader] Saved text IR: " << text_xml << std::endl;
            }
        }

        // Release weight source early
        source.reset();

        // ─── Compile ───
        if (use_vl && vision_model) {
            std::string vision_device = device_;
            if (const char* env_dev = std::getenv("OV_GENAI_VISION_DEVICE")) {
                vision_device = env_dev;
            }
            std::cout << "[ModelLoader] Compiling vision model on " << vision_device << std::endl;
            compiled_vision_ = core.compile_model(vision_model, vision_device);
        }
        std::cout << "[ModelLoader] Compiling text model on " << device_ << std::endl;
        compiled_text_ = core.compile_model(text_model, device_);
        gpu_ctx_ = try_get_gpu_context(compiled_text_);

        // ─── Tokenizer ───
        try {
            tokenizer_ = std::make_unique<ov::genai::Tokenizer>(model_dir);
        } catch (const std::exception&) {
            tokenizer_.reset();
        }

        // ─── Stop tokens ───
        stop_token_ids_ = resolve_stop_token_ids(model_dir, tokenizer_.get());
    }
};

// ─── Public interface ───

ModelLoader::ModelLoader(const std::filesystem::path& model_dir, const LoadParams& params)
    : impl_(std::make_unique<Impl>()) {
    impl_->load(model_dir, params);
}

ModelLoader::~ModelLoader() = default;
ModelLoader::ModelLoader(ModelLoader&&) noexcept = default;
ModelLoader& ModelLoader::operator=(ModelLoader&&) noexcept = default;

const models::Qwen3_5Config& ModelLoader::config() const { return impl_->cfg_; }
const models::Qwen3_5VisionPreprocessConfig& ModelLoader::preprocess_config() const { return impl_->pre_cfg_; }
ov::CompiledModel& ModelLoader::compiled_text() { return impl_->compiled_text_; }

ov::CompiledModel* ModelLoader::compiled_vision() {
    return impl_->compiled_vision_.has_value() ? &*impl_->compiled_vision_ : nullptr;
}

const ov::Tensor& ModelLoader::pos_embed_weight() const { return impl_->pos_embed_weight_; }
ov::genai::Tokenizer* ModelLoader::tokenizer() { return impl_->tokenizer_.get(); }
const std::filesystem::path& ModelLoader::model_dir() const { return impl_->model_dir_; }
const std::string& ModelLoader::device() const { return impl_->device_; }
const std::set<int64_t>& ModelLoader::stop_token_ids() const { return impl_->stop_token_ids_; }

void ModelLoader::convert(const std::filesystem::path& model_dir, const ConvertParams& params) {
    // ─── Validate ───
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir)) {
        throw std::runtime_error("Model directory does not exist: " + model_dir.string());
    }
    if (!std::filesystem::exists(model_dir / "config.json")) {
        throw std::runtime_error("Model directory missing config.json: " + model_dir.string());
    }
    if (!has_safetensors_file(model_dir)) {
        throw std::runtime_error("Model directory missing .safetensors files: " + model_dir.string());
    }

    // ─── Config ───
    auto cfg = models::Qwen3_5Config::from_json_file(model_dir);
    if (params.num_layers.has_value()) {
        int nl = *params.num_layers;
        if (nl <= 0 || nl > cfg.text.num_hidden_layers) {
            throw std::runtime_error("num_layers must be in [1, " +
                std::to_string(cfg.text.num_hidden_layers) + "], got: " + std::to_string(nl));
        }
        cfg.text.num_hidden_layers = nl;
        if (!cfg.text.layer_types.empty()) {
            if (cfg.text.layer_types.size() >= static_cast<size_t>(nl)) {
                cfg.text.layer_types.resize(static_cast<size_t>(nl));
            } else {
                cfg.text.layer_types.clear();
            }
        }
        cfg.finalize();
        cfg.validate();
    }

    // ─── Quantization ───
    auto quant = params.quant_config;
    if (!quant.enabled()) {
        quant = weights::parse_quantization_config_from_env();
    }
    if (quant.enabled() && quant.group_size <= 0) {
        throw std::runtime_error("Quantization group_size must be > 0");
    }
    // Vision quantization disabled (same as load)
    weights::QuantizationConfig vision_quant{};

    // ─── IR paths ───
    const bool use_vl = params.enable_vision;
    std::string text_ir_stem = (use_vl ? "qwen3_5_text_vl" : "qwen3_5_text") + quant_cache_suffix(quant);
    if (params.num_layers.has_value()) {
        text_ir_stem += "_l" + std::to_string(*params.num_layers);
    }
    std::string vision_ir_stem = "qwen3_5_vision" + quant_cache_suffix(vision_quant);
    const auto text_xml = model_dir / (text_ir_stem + ".xml");
    const auto text_bin = model_dir / (text_ir_stem + ".bin");
    const auto vision_xml = model_dir / (vision_ir_stem + ".xml");
    const auto vision_bin = model_dir / (vision_ir_stem + ".bin");

    // ─── Check existing ───
    if (!params.force) {
        bool text_exists = has_ir_model_pair(text_xml, text_bin);
        bool vision_ok = !use_vl || has_ir_model_pair(vision_xml, vision_bin);
        if (text_exists && vision_ok) {
            std::cout << "[convert] IR already exists (use --force to overwrite):" << std::endl;
            std::cout << "  text:   " << text_xml << std::endl;
            if (use_vl) std::cout << "  vision: " << vision_xml << std::endl;
            return;
        }
    }

    // ─── Load weights ───
    std::cout << "[convert] Loading safetensors from " << model_dir << " ..." << std::endl;
    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    auto source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));

    // ─── Build and save vision model ───
    if (use_vl) {
        std::cout << "[convert] Building vision IR ..." << std::endl;
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant);
        auto vision_model = models::create_qwen3_5_vision_model(cfg, *source, finalizer);
        const std::string pe_name = resolve_pos_embed_name(*source);
        ov::Tensor pos_embed = source->get_tensor(pe_name);
        embed_pos_embed_in_vision_model(vision_model, pos_embed);
        ov::serialize(vision_model, vision_xml.string(), vision_bin.string());
        std::cout << "[convert] Saved vision IR: " << vision_xml << std::endl;
    }

    // ─── Build and save text model ───
    bool need_text = params.force || !has_ir_model_pair(text_xml, text_bin);
    if (need_text) {
        std::cout << "[convert] Building text IR (" << cfg.text.num_hidden_layers << " layers"
                  << (quant.enabled() ? ", quantized" : "") << ") ..." << std::endl;
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(quant);
        auto text_model = models::create_qwen3_5_text_model(cfg, *source, finalizer, false, use_vl);
        ov::serialize(text_model, text_xml.string(), text_bin.string());
        std::cout << "[convert] Saved text IR: " << text_xml << std::endl;
    }

    // ─── Summary ───
    auto file_size_mb = [](const std::filesystem::path& p) -> double {
        if (!std::filesystem::exists(p)) return 0;
        return static_cast<double>(std::filesystem::file_size(p)) / (1024.0 * 1024.0);
    };
    std::cout << "\n[convert] Done! IR files saved to " << model_dir << std::endl;
    std::cout << "  " << text_ir_stem << ".xml + .bin"
              << " (" << std::fixed << std::setprecision(1) << file_size_mb(text_bin) << " MB)" << std::endl;
    if (use_vl) {
        std::cout << "  " << vision_ir_stem << ".xml + .bin"
                  << " (" << file_size_mb(vision_bin) << " MB)" << std::endl;
    }
}

}  // namespace ov::genai::modeling
