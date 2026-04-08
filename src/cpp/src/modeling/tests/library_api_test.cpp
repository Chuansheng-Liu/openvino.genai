// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the library API streaming pipeline components:
//   - ThinkingTracker
//   - ToolCallParser
//   - Sampler (basic functionality)
//   - C API default parameter constructors

#include <set>
#include <gtest/gtest.h>

#include "modeling/api/thinking_tracker.hpp"
#include "modeling/api/tool_call_parser.hpp"
#include "modeling/api/sampler.hpp"
#include "modeling/api/types.hpp"
#include "modeling/api/ov_modeling_qwen3_5.h"

using namespace ov::genai::modeling;

// ═══════════════════════════════════════════════════════════════════
//  ThinkingTracker Tests
// ═══════════════════════════════════════════════════════════════════

class ThinkingTrackerTest : public ::testing::Test {
protected:
    ThinkingTracker tracker;
};

TEST_F(ThinkingTrackerTest, InitialState) {
    EXPECT_EQ(tracker.state(), ThinkingState::BEFORE_THINKING);
    EXPECT_FALSE(tracker.is_thinking());
}

TEST_F(ThinkingTrackerTest, FullThinkingBlockSingleChunk) {
    auto r = tracker.process("<think>I need to think about this</think>Here is the answer");
    EXPECT_EQ(r.thinking_text, "I need to think about this");
    EXPECT_EQ(r.content_text, "Here is the answer");
    EXPECT_EQ(tracker.state(), ThinkingState::AFTER_THINKING);
}

TEST_F(ThinkingTrackerTest, ThinkingBlockAcrossTwoChunks) {
    auto r1 = tracker.process("<think>thinking...");
    EXPECT_EQ(r1.thinking_text, "thinking...");
    EXPECT_TRUE(r1.content_text.empty());
    EXPECT_TRUE(tracker.is_thinking());

    auto r2 = tracker.process("more thinking</think>content");
    EXPECT_EQ(r2.thinking_text, "more thinking");
    EXPECT_EQ(r2.content_text, "content");
    EXPECT_EQ(tracker.state(), ThinkingState::AFTER_THINKING);
}

TEST_F(ThinkingTrackerTest, ThinkingBlockAcrossMultipleChunks) {
    auto r1 = tracker.process("<think>");
    EXPECT_TRUE(r1.thinking_text.empty());
    EXPECT_TRUE(r1.content_text.empty());

    auto r2 = tracker.process("step 1\n");
    EXPECT_EQ(r2.thinking_text, "step 1\n");

    auto r3 = tracker.process("step 2\n");
    EXPECT_EQ(r3.thinking_text, "step 2\n");

    auto r4 = tracker.process("</think>\n\n");
    EXPECT_TRUE(r4.thinking_text.empty());
    EXPECT_EQ(r4.content_text, "\n\n");
}

TEST_F(ThinkingTrackerTest, NoThinkingTag) {
    // Text without <think> goes directly to content
    auto r = tracker.process("Hello, world!");
    // Before seeing <think>, text goes to content in BEFORE_THINKING state
    // (depends on implementation: might buffer looking for <think>)
    // The important thing is that after processing, the content appears somewhere
    EXPECT_EQ(tracker.state(), ThinkingState::BEFORE_THINKING);
}

TEST_F(ThinkingTrackerTest, PartialOpenTagAcrossChunks) {
    auto r1 = tracker.process("<thi");
    // Buffer should hold partial tag
    auto r2 = tracker.process("nk>thinking text");
    EXPECT_EQ(tracker.state(), ThinkingState::IN_THINKING);
    EXPECT_TRUE(r2.thinking_text.find("thinking text") != std::string::npos);
}

TEST_F(ThinkingTrackerTest, PartialCloseTagAcrossChunks) {
    tracker.process("<think>inner");
    auto r = tracker.process("</thi");
    // Should be buffering the partial close tag
    EXPECT_TRUE(tracker.is_thinking());

    auto r2 = tracker.process("nk>content");
    EXPECT_EQ(r2.content_text, "content");
    EXPECT_FALSE(tracker.is_thinking());
}

TEST_F(ThinkingTrackerTest, ResetReturnsToInitial) {
    tracker.process("<think>thinking</think>done");
    EXPECT_EQ(tracker.state(), ThinkingState::AFTER_THINKING);

    tracker.reset();
    EXPECT_EQ(tracker.state(), ThinkingState::BEFORE_THINKING);
    EXPECT_FALSE(tracker.is_thinking());
}

TEST_F(ThinkingTrackerTest, EmptyString) {
    auto r = tracker.process("");
    EXPECT_TRUE(r.thinking_text.empty());
    EXPECT_TRUE(r.content_text.empty());
}

TEST_F(ThinkingTrackerTest, ThinkTagWithNewlines) {
    // Typical Qwen3.5 format: <think>\nthinking\n</think>\n\ncontent
    auto r = tracker.process("<think>\nI'm thinking carefully\n</think>\n\nThe answer is 42");
    EXPECT_TRUE(r.thinking_text.find("I'm thinking carefully") != std::string::npos);
    EXPECT_TRUE(r.content_text.find("The answer is 42") != std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════
//  ToolCallParser Tests
// ═══════════════════════════════════════════════════════════════════

class ToolCallParserTest : public ::testing::Test {
protected:
    ToolCallParser parser;
};

TEST_F(ToolCallParserTest, NoToolCall) {
    auto r = parser.process("Hello, world!");
    EXPECT_EQ(r.text, "Hello, world!");
    EXPECT_TRUE(r.tool_calls.empty());
    EXPECT_FALSE(r.in_tool_call);
}

TEST_F(ToolCallParserTest, SingleToolCallComplete) {
    std::string input = "<tool_call>\n"
                        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n"
                        "</tool_call>";
    auto r = parser.process(input);
    ASSERT_EQ(r.tool_calls.size(), 1);
    EXPECT_EQ(r.tool_calls[0].name, "get_weather");
    EXPECT_TRUE(r.tool_calls[0].arguments.find("Beijing") != std::string::npos);
    EXPECT_TRUE(r.text.empty());
}

TEST_F(ToolCallParserTest, ToolCallWithLeadingText) {
    std::string input = "Let me check the weather.\n"
                        "<tool_call>\n"
                        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\"}}\n"
                        "</tool_call>";
    auto r = parser.process(input);
    ASSERT_EQ(r.tool_calls.size(), 1);
    EXPECT_EQ(r.tool_calls[0].name, "get_weather");
    EXPECT_TRUE(r.text.find("Let me check the weather") != std::string::npos);
}

TEST_F(ToolCallParserTest, MultipleToolCalls) {
    std::string input = "<tool_call>\n"
                        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}\n"
                        "</tool_call>\n"
                        "<tool_call>\n"
                        "{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC+8\"}}\n"
                        "</tool_call>";
    auto r = parser.process(input);
    ASSERT_EQ(r.tool_calls.size(), 2);
    EXPECT_EQ(r.tool_calls[0].name, "get_weather");
    EXPECT_EQ(r.tool_calls[1].name, "get_time");
}

TEST_F(ToolCallParserTest, ToolCallAcrossChunks) {
    auto r1 = parser.process("<tool_call>\n{\"name\": \"calc\"");
    EXPECT_TRUE(r1.in_tool_call);
    EXPECT_TRUE(r1.tool_calls.empty());

    auto r2 = parser.process(", \"arguments\": {\"expr\": \"2+2\"}}\n</tool_call>");
    EXPECT_FALSE(r2.in_tool_call);
    ASSERT_EQ(r2.tool_calls.size(), 1);
    EXPECT_EQ(r2.tool_calls[0].name, "calc");
}

TEST_F(ToolCallParserTest, PartialOpenTag) {
    auto r1 = parser.process("text <tool");
    // Should buffer the partial tag
    auto r2 = parser.process("_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>");
    ASSERT_EQ(r2.tool_calls.size(), 1);
    EXPECT_EQ(r2.tool_calls[0].name, "f");
}

TEST_F(ToolCallParserTest, PartialCloseTag) {
    parser.process("<tool_call>\n{\"name\": \"f\", \"arguments\": {}}");
    auto r = parser.process("\n</tool");
    EXPECT_TRUE(r.in_tool_call);

    auto r2 = parser.process("_call>");
    ASSERT_EQ(r2.tool_calls.size(), 1);
    EXPECT_EQ(r2.tool_calls[0].name, "f");
}

TEST_F(ToolCallParserTest, MalformedJSON) {
    // Malformed JSON should still produce a ToolCall with empty name/arguments
    auto r = parser.process("<tool_call>\nnot valid json\n</tool_call>");
    ASSERT_EQ(r.tool_calls.size(), 1);
    // Name might be empty, but it shouldn't crash
}

TEST_F(ToolCallParserTest, NestedArguments) {
    std::string input = "<tool_call>\n"
                        "{\"name\": \"search\", \"arguments\": "
                        "{\"query\": \"hello\", \"filters\": {\"type\": \"web\", \"limit\": 10}}}\n"
                        "</tool_call>";
    auto r = parser.process(input);
    ASSERT_EQ(r.tool_calls.size(), 1);
    EXPECT_EQ(r.tool_calls[0].name, "search");
    EXPECT_TRUE(r.tool_calls[0].arguments.find("filters") != std::string::npos);
}

TEST_F(ToolCallParserTest, EmptyArguments) {
    auto r = parser.process("<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>");
    ASSERT_EQ(r.tool_calls.size(), 1);
    EXPECT_EQ(r.tool_calls[0].name, "noop");
}

TEST_F(ToolCallParserTest, CallCountIncreases) {
    EXPECT_EQ(parser.call_count(), 0);
    parser.process("<tool_call>\n{\"name\": \"f1\", \"arguments\": {}}\n</tool_call>");
    EXPECT_EQ(parser.call_count(), 1);
    parser.process("<tool_call>\n{\"name\": \"f2\", \"arguments\": {}}\n</tool_call>");
    EXPECT_EQ(parser.call_count(), 2);
}

TEST_F(ToolCallParserTest, ResetClearsState) {
    parser.process("<tool_call>\n{\"name\": \"f1\", \"arguments\": {}}\n</tool_call>");
    EXPECT_EQ(parser.call_count(), 1);
    parser.reset();
    EXPECT_EQ(parser.call_count(), 0);
}

TEST_F(ToolCallParserTest, FlushMidToolCall) {
    parser.process("<tool_call>\npartial content");
    auto r = parser.flush();
    // Flush should emit accumulated content as text
    EXPECT_TRUE(r.text.find("partial content") != std::string::npos ||
                r.text.find("<tool_call>") != std::string::npos);
}

TEST_F(ToolCallParserTest, EmptyInput) {
    auto r = parser.process("");
    EXPECT_TRUE(r.text.empty());
    EXPECT_TRUE(r.tool_calls.empty());
}

// ═══════════════════════════════════════════════════════════════════
//  Sampler Tests (free functions: argmax_f32, sample_fast)
// ═══════════════════════════════════════════════════════════════════

TEST(SamplerTest, ArgmaxPicksMax) {
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(argmax_f32(logits), 1);  // index 1 has value 5.0
}

TEST(SamplerTest, ArgmaxDifferentMax) {
    std::vector<float> logits = {10.0f, 1.0f, 2.0f, 3.0f};
    EXPECT_EQ(argmax_f32(logits), 0);  // index 0 has value 10.0
}

TEST(SamplerTest, ArgmaxDeterministic) {
    std::vector<float> logits = {1.0f, 3.0f, 7.0f, 2.0f, 5.0f};
    EXPECT_EQ(argmax_f32(logits), argmax_f32(logits));
    EXPECT_EQ(argmax_f32(logits), 2);  // index 2 has value 7.0
}

TEST(SamplerTest, ExtractLastLogitsF32) {
    // Shape [1, 3, 5] — last token's logits are the last 5 values
    std::vector<float> data(3 * 5, 0.0f);
    data[10] = 1.0f; data[11] = 2.0f; data[12] = 3.0f; data[13] = 4.0f; data[14] = 5.0f;
    ov::Tensor logits(ov::element::f32, {1, 3, 5}, data.data());

    std::vector<float> out;
    extract_last_logits_f32(logits, out);
    ASSERT_EQ(out.size(), 5u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[4], 5.0f);
}

TEST(SamplerTest, SampleFastWithLowTemperature) {
    // Very low temperature → effectively deterministic, picks highest
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f, 4.0f};
    std::mt19937 rng(42);
    SamplingContext ctx;

    auto token = sample_fast(logits.data(), logits.size(), 0.01f, 0.95f, 20, rng, ctx);
    EXPECT_EQ(token, 1);  // index 1 has value 5.0 (overwhelmingly likely)
}

TEST(SamplerTest, SampleFastTopK) {
    // With large separation, top_k=2 should only pick the two highest
    std::vector<float> logits = {0.1f, 10.0f, 0.2f, 9.0f, 0.3f};
    SamplingContext ctx;

    std::set<int64_t> seen;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 rng(i + 1);
        auto token = sample_fast(logits.data(), logits.size(), 1.0f, 1.0f, 2, rng, ctx);
        seen.insert(token);
    }
    EXPECT_TRUE(seen.count(1) > 0);
    EXPECT_TRUE(seen.count(3) > 0);
    EXPECT_EQ(seen.count(0), 0u);
    EXPECT_EQ(seen.count(2), 0u);
    EXPECT_EQ(seen.count(4), 0u);
}

// ═══════════════════════════════════════════════════════════════════
//  Types Tests
// ═══════════════════════════════════════════════════════════════════

TEST(TypesTest, SamplingParamsDefaults) {
    SamplingParams p;
    EXPECT_FLOAT_EQ(p.temperature, 0.0f);
    EXPECT_FLOAT_EQ(p.top_p, 0.95f);
    EXPECT_EQ(p.top_k, 20u);
    EXPECT_FLOAT_EQ(p.repetition_penalty, 1.0f);
}

TEST(TypesTest, GenerateParamsDefaults) {
    GenerateParams p;
    EXPECT_EQ(p.max_new_tokens, 256);
    EXPECT_TRUE(p.enable_thinking);
    EXPECT_TRUE(p.stop_strings.empty());
}

TEST(TypesTest, StreamChunkDefaults) {
    StreamChunk c;
    EXPECT_EQ(c.token_id, -1);
    EXPECT_TRUE(c.token_text.empty());
    EXPECT_FALSE(c.is_thinking);
}

TEST(TypesTest, GenerateResultDefaults) {
    GenerateResult r;
    EXPECT_TRUE(r.token_ids.empty());
    EXPECT_TRUE(r.text.empty());
    EXPECT_EQ(r.stop_reason, StopReason::EOS);
}

// ═══════════════════════════════════════════════════════════════════
//  C API Types Tests (no model needed)
// ═══════════════════════════════════════════════════════════════════

TEST(CApiTest, DefaultLoadParams) {
    auto p = ov_default_load_params();
    EXPECT_STREQ(p.device, "GPU");
    EXPECT_EQ(p.cache_ir, 1);
    EXPECT_EQ(p.enable_vision, 0);
    EXPECT_EQ(p.quant_mode, nullptr);
    EXPECT_EQ(p.quant_group_size, 0);
    EXPECT_EQ(p.num_layers, 0);
}

TEST(CApiTest, DefaultGenParams) {
    auto p = ov_default_gen_params();
    EXPECT_EQ(p.max_new_tokens, 256);
    EXPECT_FLOAT_EQ(p.temperature, 0.0f);
    EXPECT_FLOAT_EQ(p.top_p, 0.95f);
    EXPECT_EQ(p.top_k, 20);
    EXPECT_FLOAT_EQ(p.repetition_penalty, 1.0f);
    EXPECT_EQ(p.enable_thinking, 1);
}

TEST(CApiTest, NullModelLoad) {
    auto* model = ov_model_load(nullptr, nullptr);
    EXPECT_EQ(model, nullptr);
    const char* err = ov_get_last_error();
    EXPECT_NE(err, nullptr);
    EXPECT_TRUE(std::string(err).find("NULL") != std::string::npos);
}

TEST(CApiTest, NullSessionCreate) {
    auto* session = ov_session_create(nullptr);
    EXPECT_EQ(session, nullptr);
    const char* err = ov_get_last_error();
    EXPECT_NE(err, nullptr);
}

TEST(CApiTest, NullGenerate) {
    ov_gen_params_t params = ov_default_gen_params();
    ov_gen_result_t result{};
    auto status = ov_generate(nullptr, "hello", &params, &result);
    EXPECT_EQ(status, OV_ERROR_INVALID_PARAM);
}

TEST(CApiTest, NullGenerateStream) {
    auto status = ov_generate_stream(nullptr, "hello", nullptr, nullptr, nullptr);
    EXPECT_EQ(status, OV_ERROR_INVALID_PARAM);
}

TEST(CApiTest, OvFreeAcceptsNull) {
    // ov_free(NULL) should not crash
    ov_free(nullptr);
}

TEST(CApiTest, StopNullSession) {
    // ov_generate_stop(NULL) should not crash
    ov_generate_stop(nullptr);
}

TEST(CApiTest, ResetNullSession) {
    // ov_session_reset(NULL) should not crash
    ov_session_reset(nullptr);
}

TEST(CApiTest, FreeNullModel) {
    // ov_model_free(NULL) should not crash
    ov_model_free(nullptr);
}

TEST(CApiTest, FreeNullSession) {
    // ov_session_free(NULL) should not crash
    ov_session_free(nullptr);
}

TEST(CApiTest, InvalidPathModelLoad) {
    ov_load_params_t params = ov_default_load_params();
    auto* model = ov_model_load("/nonexistent/path/to/model", &params);
    EXPECT_EQ(model, nullptr);
    const char* err = ov_get_last_error();
    EXPECT_NE(err, nullptr);
}

TEST(CApiTest, GetModelInfoNull) {
    ov_model_info_t info{};
    auto status = ov_get_model_info(nullptr, &info);
    EXPECT_EQ(status, OV_ERROR_INVALID_PARAM);
}
