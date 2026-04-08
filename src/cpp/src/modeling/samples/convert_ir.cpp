// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// convert_ir — Convert HuggingFace Qwen3.5 model to OpenVINO IR files.
//
// Pre-converts safetensors weights to OpenVINO IR (.xml + .bin) so that
// subsequent ov_serve or modeling_qwen3_5 runs skip the expensive IR build step.
//
// Usage:
//   convert_ir --model <hf_model_dir> [--vl] [--force] [--num-layers N]
//
// Quantization is controlled by environment variables:
//   OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
//   OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
//
// Examples:
//   convert_ir --model C:\data\models\Qwen3.5-4B
//   convert_ir --model C:\data\models\Qwen3.5-35B-A3B --vl
//   set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym && convert_ir --model ...

#include <chrono>
#include <iostream>
#include <string>

#include "modeling/api/model_loader.hpp"

using namespace ov::genai::modeling;

void print_usage() {
    std::cerr << "Usage: convert_ir --model <path> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --model <path>     HuggingFace model directory (required)\n"
              << "  --vl               Also convert vision model for VL mode\n"
              << "  --force            Overwrite existing IR files\n"
              << "  --num-layers N     Only convert first N layers (debug)\n"
              << "  --help, -h         Show this help\n"
              << "\n"
              << "Quantization env vars:\n"
              << "  OV_GENAI_INFLIGHT_QUANT_MODE       int4_sym|int4_asym|int8_sym|int8_asym\n"
              << "  OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE  128 (default)\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    ConvertParams params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model") && i + 1 < argc) model_path = argv[++i];
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
        ModelLoader::convert(model_path, params);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[convert_ir] Total time: " << sec << "s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
