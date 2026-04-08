// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/weights/quantization_config.hpp"

namespace ov::genai::modeling {

/// Parameters controlling model loading behavior.
struct LoadParams {
    std::string device = "GPU";
    weights::QuantizationConfig quant_config;   // defaults from env if empty
    bool cache_ir = true;                       // cache generated IR to disk
    bool enable_vision = true;                  // load vision model
    std::optional<int> num_layers;              // debug: override layer count
    int max_pixels = 0;                         // VL: max pixel count
};

/// Parameters controlling IR conversion (no GPU compile needed).
struct ConvertParams {
    weights::QuantizationConfig quant_config;   // defaults from env if empty
    bool enable_vision = false;                 // also convert vision model
    std::optional<int> num_layers;              // debug: override layer count
    bool force = false;                         // overwrite existing IR
};

/// Loads and compiles a Qwen3.5 model, ready for creating inference sessions.
///
/// RAII: constructor loads everything (config → weights → IR → compile).
/// Failure throws std::runtime_error.
class ModelLoader {
public:
    /// Load model from HuggingFace model directory.
    explicit ModelLoader(const std::filesystem::path& model_dir,
                         const LoadParams& params = {});

    ~ModelLoader();

    // Non-copyable
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Moveable
    ModelLoader(ModelLoader&&) noexcept;
    ModelLoader& operator=(ModelLoader&&) noexcept;

    /// Model configuration (text + vision + token IDs).
    const models::Qwen3_5Config& config() const;

    /// Vision preprocess configuration.
    const models::Qwen3_5VisionPreprocessConfig& preprocess_config() const;

    /// Compiled text model (used to create InferRequest).
    ov::CompiledModel& compiled_text();

    /// Compiled vision model (nullptr if vision not enabled).
    ov::CompiledModel* compiled_vision();

    /// Pos embed weight tensor (needed for vision preprocessing).
    const ov::Tensor& pos_embed_weight() const;

    /// Tokenizer (may be nullptr if tokenizer files are missing).
    ov::genai::Tokenizer* tokenizer();

    /// Model directory path.
    const std::filesystem::path& model_dir() const;

    /// Device name used for compilation.
    const std::string& device() const;

    /// Stop token IDs loaded from generation_config.json.
    const std::set<int64_t>& stop_token_ids() const;

    /// Convert HF model to OpenVINO IR files (no GPU compile needed).
    /// Saves .xml/.bin to model directory. Subsequent ModelLoader construction
    /// will reuse cached IR, skipping the expensive build step.
    static void convert(const std::filesystem::path& model_dir,
                        const ConvertParams& params = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ov::genai::modeling
