// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <openvino/openvino.hpp>

#include "modeling/api/model_loader.hpp"
#include "modeling/api/types.hpp"

namespace ov::genai::modeling {

/// An independent inference session holding its own InferRequest, KV cache,
/// linear attention state, and decode position.
///
/// One ModelLoader can create multiple Sessions (memory permitting).
/// A single Session must NOT be used from multiple threads concurrently.
class Session {
public:
    /// Create a session bound to a loaded model.
    explicit Session(ModelLoader& model);
    ~Session();

    // Non-copyable
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Moveable
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    // ─── Core inference ───

    /// Text generation (blocking). Callback is optional for streaming.
    GenerateResult generate(const std::string& prompt,
                            const GenerateParams& params = {},
                            StreamCallback callback = nullptr);

    /// VL generation with an image tensor. Image should be HWC uint8.
    GenerateResult generate_vl(const std::string& prompt,
                               const ov::Tensor& image,
                               const GenerateParams& params = {},
                               StreamCallback callback = nullptr);

    // ─── Control ───

    /// Stop current generation (thread-safe, call from any thread).
    void stop();

    /// Reset all session state (KV cache + linear attention + positions).
    /// Next generate() starts from scratch.
    void reset();

    /// Whether generate() is currently executing.
    bool is_generating() const;

    // ─── Lifecycle ───

    /// Pre-warm GPU memory pool by running dummy inference at peak sequence length.
    /// This prevents CL_OUT_OF_RESOURCES on the first large request by forcing
    /// the GPU plugin to pre-allocate all intermediate buffers.
    /// @param max_seq_len  Target total sequence length (prefill + decode warmup).
    ///                     Should be >= max expected (prompt_tokens + max_new_tokens).
    void warmup(int max_seq_len);

    /// Recreate the underlying InferRequest from the existing CompiledModel.
    /// Use after GPU errors (CL_OUT_OF_RESOURCES) to recover from corrupted
    /// GPU driver state. Also re-creates all GPU-context-backed tensors.
    void recreate();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ov::genai::modeling
