// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thin CLI frontend for the Qwen3.5 Modeling Library API.
// All inference logic lives in the library (api/); this file only does
// CLI parsing and result formatting.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <openvino/openvino.hpp>

#include "load_image.hpp"
#include "modeling/api/model_loader.hpp"
#include "modeling/api/session.hpp"
#include "modeling/api/types.hpp"

namespace {

struct CliOptions {
    std::filesystem::path model_dir;
    std::string mode = "text";
    std::filesystem::path image_path;
    std::string prompt;
    std::string device = "GPU";
    int max_new_tokens = 64;
    bool cache_model = false;
    std::optional<int> num_layers;
    int max_pixels = 0;

    // Sampling
    float temperature = 0.0f;
    float top_p = 0.95f;
    size_t top_k = 20;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    size_t rng_seed = 0;
    bool enable_thinking = true;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --model <path> [options]\n"
              << "\nRequired:\n"
              << "  --model <path>           HuggingFace model directory\n"
              << "\nOptional:\n"
              << "  --mode text|vl           Generation mode (default: text)\n"
              << "  --image <path>           Image file for VL mode\n"
              << "  --prompt <text>          Input prompt\n"
              << "  --device GPU|CPU         Device (default: GPU)\n"
              << "  --output-tokens <N>      Max tokens to generate (default: 64)\n"
              << "  --cache-model            Cache IR to disk\n"
              << "  --num-layers <N>         Override layer count (debug)\n"
              << "  --max-pixels <N>         Max pixel count for VL\n"
              << "  --temperature <F>        Sampling temperature (0=greedy)\n"
              << "  --top-p <F>              Top-P sampling\n"
              << "  --top-k <N>              Top-K sampling\n"
              << "  --repetition-penalty <F> Repetition penalty\n"
              << "  --frequency-penalty <F>  Frequency penalty\n"
              << "  --presence-penalty <F>   Presence penalty\n"
              << "  --rng-seed <N>           RNG seed (0=random)\n"
              << "  --think 0|1              Enable thinking mode (default: 1)\n";
}

CliOptions parse_cli(int argc, char* argv[]) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        auto take = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--model")               opts.model_dir = take("--model");
        else if (arg == "--mode")            opts.mode = take("--mode");
        else if (arg == "--image")           opts.image_path = take("--image");
        else if (arg == "--prompt")          opts.prompt = take("--prompt");
        else if (arg == "--device")          opts.device = take("--device");
        else if (arg == "--output-tokens")   opts.max_new_tokens = std::stoi(take("--output-tokens"));
        else if (arg == "--cache-model")     opts.cache_model = true;
        else if (arg == "--num-layers")      opts.num_layers = std::stoi(take("--num-layers"));
        else if (arg == "--max-pixels")      opts.max_pixels = std::stoi(take("--max-pixels"));
        else if (arg == "--temperature")     opts.temperature = std::stof(take("--temperature"));
        else if (arg == "--top-p")           opts.top_p = std::stof(take("--top-p"));
        else if (arg == "--top-k")           opts.top_k = static_cast<size_t>(std::stoi(take("--top-k")));
        else if (arg == "--repetition-penalty") opts.repetition_penalty = std::stof(take("--repetition-penalty"));
        else if (arg == "--frequency-penalty")  opts.frequency_penalty = std::stof(take("--frequency-penalty"));
        else if (arg == "--presence-penalty")   opts.presence_penalty = std::stof(take("--presence-penalty"));
        else if (arg == "--rng-seed")        opts.rng_seed = static_cast<size_t>(std::stoi(take("--rng-seed")));
        else if (arg == "--think")           opts.enable_thinking = (std::stoi(take("--think")) != 0);
        else throw std::runtime_error("Unknown option: " + arg);
    }

    if (opts.model_dir.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (opts.prompt.empty()) {
        opts.prompt = (opts.mode == "vl") ? "Describe the image." : "Write one sentence about OpenVINO.";
    }
    return opts;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    auto opts = parse_cli(argc, argv);
    const bool use_vl = (opts.mode == "vl");

    // ─── Load model via Library API ───
    ov::genai::modeling::LoadParams load_params;
    load_params.device = opts.device;
    load_params.cache_ir = opts.cache_model;
    load_params.enable_vision = use_vl;
    load_params.num_layers = opts.num_layers;
    load_params.max_pixels = opts.max_pixels;

    std::cout << "[cli] Loading model from " << opts.model_dir << " ..." << std::endl;
    auto model = ov::genai::modeling::ModelLoader(opts.model_dir, load_params);
    std::cout << "[cli] Model loaded." << std::endl;

    // ─── Create session ───
    auto session = ov::genai::modeling::Session(model);

    // ─── Build generation params ───
    ov::genai::modeling::GenerateParams gen_params;
    gen_params.max_new_tokens = opts.max_new_tokens;
    gen_params.enable_thinking = opts.enable_thinking;
    gen_params.sampling.temperature = opts.temperature;
    gen_params.sampling.top_p = opts.top_p;
    gen_params.sampling.top_k = opts.top_k;
    gen_params.sampling.repetition_penalty = opts.repetition_penalty;
    gen_params.sampling.frequency_penalty = opts.frequency_penalty;
    gen_params.sampling.presence_penalty = opts.presence_penalty;
    gen_params.sampling.rng_seed = opts.rng_seed;

    // ─── Generate ───
    ov::genai::modeling::GenerateResult result;
    if (use_vl) {
        if (opts.image_path.empty()) {
            std::cerr << "Error: --mode vl requires --image" << std::endl;
            return 1;
        }
        ov::Tensor image = utils::load_image(opts.image_path);
        result = session.generate_vl(opts.prompt, {image}, gen_params);
    } else {
        result = session.generate(opts.prompt, gen_params);
    }

    // ─── Output ───
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Prompt tokens: " << result.prompt_tokens << std::endl;
    std::cout << "Output tokens: " << result.generated_tokens << std::endl;
    std::cout << "TTFT: " << result.ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << result.decode_ms << " ms" << std::endl;
    if (result.generated_tokens > 1) {
        double tpot = result.decode_ms / static_cast<double>(result.generated_tokens - 1);
        std::cout << "TPOT: " << tpot << " ms/token" << std::endl;
        std::cout << "Throughput: " << result.throughput << " tokens/s" << std::endl;
    }
    if (!result.thinking_text.empty()) {
        std::cout << "Thinking Process:" << result.thinking_text << std::endl;
    }
    std::cout << result.text << std::endl;

    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
