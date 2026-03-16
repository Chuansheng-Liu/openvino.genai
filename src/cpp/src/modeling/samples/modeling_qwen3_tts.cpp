// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Qwen3-TTS End-to-End Sample
//
// This sample demonstrates the complete TTS pipeline:
//   Text -> Talker -> Code Predictor -> Speech Decoder -> WAV
//
// Usage:
//   ./modeling_qwen3_tts <model_path> "<text>" [output.wav] [device]
//
// Example:
//   ./modeling_qwen3_tts C:/models/Qwen3-TTS "Hello world" output.wav GPU
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "modeling/models/qwen3_tts/qwen3_tts_pipeline.hpp"

#include <openvino/runtime/properties.hpp>

using namespace ov::genai::modeling::models;

namespace {

void write_wav(const std::string& filename, const float* samples, size_t num_samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create WAV file: " << filename << "\n";
        return;
    }
    
    int32_t data_size = static_cast<int32_t>(num_samples * sizeof(int16_t));
    int32_t file_size = 36 + data_size;
    int16_t audio_format = 1;  // PCM
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * 2;
    int16_t block_align = num_channels * 2;
    int16_t bits_per_sample = 16;
    
    file.write("RIFF", 4);
    file.write(reinterpret_cast<char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    file.write(reinterpret_cast<char*>(&sample_rate), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    file.write(reinterpret_cast<char*>(&block_align), 2);
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);
    
    for (size_t i = 0; i < num_samples; ++i) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t sample = static_cast<int16_t>(s * 32767.0f);
        file.write(reinterpret_cast<char*>(&sample), 2);
    }
    
    file.close();
    std::cout << "Saved WAV file: " << filename << " (" << num_samples << " samples, "
              << std::fixed << std::setprecision(2) << static_cast<float>(num_samples) / sample_rate << "s)\n";
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> \"<text>\" [output.wav] [device]\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " C:/models/Qwen3-TTS \"我爱吃冰激凌。\" output.wav GPU\n";
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string input_text = argv[2];
    const std::string output_wav = (argc > 3) ? argv[3] : "output_qwen3_tts.wav";
    const std::string device = (argc > 4) ? argv[4] : "GPU";
    
    print_separator("Qwen3-TTS End-to-End Sample");
    
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Input text: \"" << input_text << "\"\n";
    std::cout << "Output file: " << output_wav << "\n";
    std::cout << "Device: " << device << "\n";
    
    // Initialize pipeline
    print_separator("Initialize Pipeline");
    
    auto load_start = std::chrono::high_resolution_clock::now();
    Qwen3TTSPipeline pipeline(model_path, device);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    
    std::cout << "Pipeline initialized in " << load_ms << " ms\n";
    std::cout << "Sample rate: " << pipeline.get_sample_rate() << " Hz\n";
    
    // Generate audio
    print_separator("Generate Audio");
    
    Qwen3TTSGenerationConfig config;
    config.temperature = 0.9f;
    config.top_k = 50;
    config.top_p = 1.0f;
    config.repetition_penalty = 1.05f;
    config.min_new_tokens = 2;
    config.max_new_tokens = 2048;
    config.seed = 42;
    
    std::cout << "Generation config:\n";
    std::cout << "  temperature: " << config.temperature << "\n";
    std::cout << "  top_k: " << config.top_k << "\n";
    std::cout << "  top_p: " << config.top_p << "\n";
    std::cout << "  repetition_penalty: " << config.repetition_penalty << "\n";
    std::cout << "  max_new_tokens: " << config.max_new_tokens << "\n";
    
    auto result = pipeline.generate(input_text, config);
    
    // Get audio data
    const float* audio_data = result.audio.data<float>();
    size_t num_samples = result.audio.get_size();
    float duration_sec = static_cast<float>(num_samples) / result.sample_rate;
    
    // Save WAV
    print_separator("Save Output");
    
    write_wav(output_wav, audio_data, num_samples, result.sample_rate);
    
    // Print summary
    print_separator("Summary");
    
    std::cout << "Input text: \"" << input_text << "\"\n";
    std::cout << "Generated frames: " << result.num_frames << "\n";
    std::cout << "Audio samples: " << num_samples << "\n";
    std::cout << "Audio duration: " << std::fixed << std::setprecision(2) << duration_sec << " seconds\n";
    std::cout << "Sample rate: " << result.sample_rate << " Hz\n";
    std::cout << "\nTiming:\n";
    std::cout << "  Model loading: " << load_ms << " ms\n";
    std::cout << "  Codec generation: " << std::fixed << std::setprecision(1) << result.generation_time_ms << " ms\n";
    std::cout << "  Speech decoding: " << result.decode_time_ms << " ms\n";
    std::cout << "  Total: " << (result.generation_time_ms + result.decode_time_ms) << " ms\n";
    
    float total_time_sec = (result.generation_time_ms + result.decode_time_ms) / 1000.0f;
    float rtf = total_time_sec / duration_sec;
    std::cout << "\nReal-time factor (RTF): " << std::fixed << std::setprecision(3) << rtf << "x\n";
    
    if (rtf < 1.0f) {
        std::cout << "  (faster than real-time)\n";
    } else {
        std::cout << "  (slower than real-time)\n";
    }
    
    print_separator("Complete");
    
    std::cout << "Output saved to: " << output_wav << "\n";
    
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
