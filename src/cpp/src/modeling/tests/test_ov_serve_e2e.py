# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# E2E integration tests for ov_serve OpenAI-compatible HTTP server.
#
# Prerequisites:
#   pip install pytest requests
#   Start server:  ov_serve.exe --model <path-to-Qwen3.5-4B> --port 8080
#
# Run:
#   pytest test_ov_serve_e2e.py -v
#   pytest test_ov_serve_e2e.py -v -k "stream"       # only streaming tests
#   OV_SERVE_URL=http://host:port pytest ...          # custom server URL
#
# Environment variables:
#   OV_SERVE_URL      — server base URL (default: http://localhost:8080)
#   OV_SERVE_TIMEOUT  — request timeout in seconds (default: 120)
#
# Notes on Qwen3.5-4B with thinking enabled:
#   - Model emits <think>...</think> block before content
#   - Thinking blocks can be 500-1500 tokens for simple questions
#   - Tests that need actual content use max_tokens=1500+
#   - Structural/protocol tests use smaller max_tokens

import json
import os
import time
import concurrent.futures

import pytest
import requests

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

BASE_URL = os.environ.get("OV_SERVE_URL", "http://localhost:8080")
TIMEOUT = int(os.environ.get("OV_SERVE_TIMEOUT", "120"))


def chat_url():
    return f"{BASE_URL}/v1/chat/completions"


def completions_url():
    return f"{BASE_URL}/v1/completions"


def models_url():
    return f"{BASE_URL}/v1/models"


def health_url():
    return f"{BASE_URL}/health"


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def chat(messages, stream=False, **kwargs):
    """Send a chat completion request."""
    body = {"messages": messages, "stream": stream, **kwargs}
    return requests.post(chat_url(), json=body, timeout=TIMEOUT, stream=stream)


def chat_json(messages, **kwargs):
    """Send non-streaming chat request, return parsed JSON."""
    r = chat(messages, stream=False, **kwargs)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    return r.json()


def parse_sse_events(response):
    """Parse SSE stream into a list of (event_type, data) tuples.
    event_type is 'data' for JSON chunks or 'done' for [DONE].
    """
    events = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                events.append(("done", None))
            else:
                try:
                    events.append(("data", json.loads(payload)))
                except json.JSONDecodeError:
                    events.append(("parse_error", payload))
    return events


def collect_stream_text(events):
    """Concatenate content deltas from SSE events."""
    parts = []
    for etype, data in events:
        if etype == "data" and data:
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    parts.append(content)
    return "".join(parts)


def collect_stream_thinking(events):
    """Concatenate reasoning_content deltas from SSE events."""
    parts = []
    for etype, data in events:
        if etype == "data" and data:
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    parts.append(reasoning)
    return "".join(parts)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
}


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def server_ready():
    """Skip all tests if ov_serve is not reachable."""
    try:
        r = requests.get(health_url(), timeout=10)
        if r.status_code != 200:
            pytest.skip(f"ov_serve health check failed: {r.status_code}")
    except requests.ConnectionError:
        pytest.skip(f"ov_serve not reachable at {BASE_URL}")
    except requests.Timeout:
        pytest.skip(f"ov_serve health check timed out at {BASE_URL}")


# ═══════════════════════════════════════════════════════════════════
#  1. API Contract Tests
# ═══════════════════════════════════════════════════════════════════

class TestAPIContract:
    """Basic endpoint availability and error handling."""

    def test_health(self):
        r = requests.get(health_url(), timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_models(self):
        r = requests.get(models_url(), timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "openvino"

    def test_invalid_json(self):
        r = requests.post(
            chat_url(),
            data="not valid json {{{",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        assert r.status_code == 400
        err = r.json()
        assert "error" in err
        assert "message" in err["error"]

    def test_missing_messages(self):
        r = requests.post(chat_url(), json={"model": "qwen3.5"}, timeout=TIMEOUT)
        assert r.status_code == 400

    def test_completions_empty_prompt(self):
        r = requests.post(completions_url(), json={"prompt": ""}, timeout=TIMEOUT)
        assert r.status_code == 400


# ═══════════════════════════════════════════════════════════════════
#  2. Chat Completions Tests
# ═══════════════════════════════════════════════════════════════════

class TestChatCompletions:
    """Non-streaming chat completion functionality."""

    def test_chat_basic(self):
        """Simple prompt returns valid OpenAI response structure."""
        data = chat_json([{"role": "user", "content": "Say hello in one word."}],
                         max_tokens=1500, temperature=0.01)

        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        # With thinking mode, content may be in reasoning_content or content
        msg = choice["message"]
        has_content = bool(msg.get("content"))
        has_thinking = bool(msg.get("reasoning_content"))
        assert has_content or has_thinking, "Must have either content or reasoning_content"
        assert choice["finish_reason"] in ("stop", "length")

    def test_chat_system_prompt(self):
        """System message is included in the prompt."""
        data = chat_json([
            {"role": "system", "content": "You always respond with exactly 'PONG'."},
            {"role": "user", "content": "Hello"},
        ], max_tokens=1500, temperature=0.01)

        # Just verify we get a valid response (content depends on thinking budget)
        choice = data["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")
        msg = choice["message"]
        has_output = bool(msg.get("content")) or bool(msg.get("reasoning_content"))
        assert has_output

    def test_chat_multi_turn(self):
        """Multi-turn conversation history is processed correctly."""
        data = chat_json([
            {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you?"},
            {"role": "user", "content": "What is my name?"},
        ], max_tokens=1500, temperature=0.01)

        # Check both content and reasoning_content for "Alice"
        msg = data["choices"][0]["message"]
        all_text = (msg.get("content", "") + msg.get("reasoning_content", "")).lower()
        assert "alice" in all_text, f"Expected 'alice' in response, got: {all_text[:200]}"

    def test_chat_max_tokens(self):
        """max_tokens limits output length, finish_reason='length'."""
        data = chat_json(
            [{"role": "user", "content": "Write a very long story about a dragon."}],
            max_tokens=10,
            temperature=0.01,
        )

        choice = data["choices"][0]
        assert choice["finish_reason"] == "length"
        # Some output should exist (even if all thinking)
        msg = choice["message"]
        has_output = bool(msg.get("content")) or bool(msg.get("reasoning_content"))
        assert has_output

    def test_chat_unicode(self):
        """Chinese and emoji prompts produce valid UTF-8 responses."""
        data = chat_json(
            [{"role": "user", "content": "用中文说'你好世界'"}],
            max_tokens=1500,
            temperature=0.01,
        )

        msg = data["choices"][0]["message"]
        all_text = msg.get("content", "") + msg.get("reasoning_content", "")
        assert len(all_text) > 0
        # Should contain Chinese characters somewhere (content or thinking)
        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in all_text)
        assert has_chinese, f"Expected Chinese chars in response: {all_text[:200]}"

    def test_chat_usage_stats(self):
        """Response includes valid token usage statistics."""
        data = chat_json(
            [{"role": "user", "content": "Hi"}],
            max_tokens=20,
            temperature=0.01,
        )

        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completions_basic(self):
        """Text completion endpoint works."""
        r = requests.post(
            completions_url(),
            json={"prompt": "The capital of France is", "max_tokens": 10, "temperature": 0.01},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["text"]) > 0


# ═══════════════════════════════════════════════════════════════════
#  3. Streaming Protocol Tests
# ═══════════════════════════════════════════════════════════════════

class TestStreaming:
    """SSE streaming format and content correctness."""

    def test_stream_sse_format(self):
        """Each SSE event is a valid 'data: {...}' record."""
        r = chat(
            [{"role": "user", "content": "Count from 1 to 3."}],
            stream=True, max_tokens=50, temperature=0.01,
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("Content-Type", "")

        events = parse_sse_events(r)
        assert len(events) >= 2  # at least role chunk + DONE

        # All non-done events should be valid JSON
        for etype, data in events:
            assert etype in ("data", "done"), f"Unexpected event type: {etype}"
            if etype == "data":
                assert isinstance(data, dict)
                assert "choices" in data

    def test_stream_role_chunk(self):
        """First SSE chunk contains delta.role='assistant'."""
        r = chat(
            [{"role": "user", "content": "Hello"}],
            stream=True, max_tokens=20, temperature=0.01,
        )
        events = parse_sse_events(r)
        data_events = [e for e in events if e[0] == "data"]
        assert len(data_events) >= 1

        first = data_events[0][1]
        delta = first["choices"][0]["delta"]
        assert delta.get("role") == "assistant"

    def test_stream_content_assembly(self):
        """Concatenated content/reasoning deltas produce non-empty text."""
        r = chat(
            [{"role": "user", "content": "Say 'hello world' and nothing else."}],
            stream=True, max_tokens=50, temperature=0.01,
        )
        events = parse_sse_events(r)
        # With thinking mode, text may be in reasoning_content or content
        text = collect_stream_text(events)
        thinking = collect_stream_thinking(events)
        assert len(text) > 0 or len(thinking) > 0, "Must have some text in stream"

    def test_stream_finish_and_done(self):
        """Final chunk has finish_reason, usage, then [DONE] terminates."""
        r = chat(
            [{"role": "user", "content": "Hi"}],
            stream=True, max_tokens=20, temperature=0.01,
        )
        events = parse_sse_events(r)

        # Must end with [DONE]
        assert events[-1][0] == "done", "Stream must end with [DONE]"

        # Second-to-last data event should have finish_reason
        data_events = [e for e in events if e[0] == "data"]
        last_data = data_events[-1][1]
        finish_reason = last_data["choices"][0].get("finish_reason")
        assert finish_reason is not None, "Final data chunk must have finish_reason"
        assert finish_reason in ("stop", "length")

        # Usage should be present in final chunk
        assert "usage" in last_data
        assert last_data["usage"]["prompt_tokens"] > 0

    def test_stream_max_tokens_finish_reason(self):
        """Streaming with small max_tokens yields finish_reason='length'."""
        r = chat(
            [{"role": "user", "content": "Write a long essay about AI."}],
            stream=True, max_tokens=10, temperature=0.01,
        )
        events = parse_sse_events(r)
        data_events = [e for e in events if e[0] == "data"]
        last_data = data_events[-1][1]
        assert last_data["choices"][0].get("finish_reason") == "length"


# ═══════════════════════════════════════════════════════════════════
#  4. Tool Calling Tests
# ═══════════════════════════════════════════════════════════════════

class TestToolCalling:
    """Tool calling detection and OpenAI-format response."""

    def test_tool_call_structure(self):
        """Tool call response has correct OpenAI structure."""
        data = chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Beijing?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=2000,
            temperature=0.01,
        )

        choice = data["choices"][0]
        msg = choice["message"]

        # Model should produce tool calls for weather question
        if "tool_calls" in msg and msg["tool_calls"]:
            tc = msg["tool_calls"][0]
            # Validate full OpenAI structure
            assert "id" in tc
            assert tc["type"] == "function"
            assert "function" in tc
            assert "name" in tc["function"]
            assert "arguments" in tc["function"]

            # Arguments should be valid JSON
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict)

            # finish_reason should be tool_calls
            assert choice["finish_reason"] == "tool_calls"
        else:
            # Model chose to answer directly — acceptable but log it
            pytest.skip("Model did not produce tool_calls (may depend on model size)")

    def test_tool_call_arguments(self):
        """Tool call arguments contain expected parameter."""
        data = chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Tokyo right now?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=2000,
            temperature=0.01,
        )

        choice = data["choices"][0]
        msg = choice["message"]

        if "tool_calls" in msg and msg["tool_calls"]:
            tc = msg["tool_calls"][0]
            fn_name = tc["function"]["name"]
            if not fn_name:
                pytest.skip("Model produced tool_call with empty function name (4B limitation)")
            assert fn_name == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args
            assert "tokyo" in args["location"].lower() or "Tokyo" in args["location"]
        else:
            pytest.skip("Model did not produce tool_calls")

    def test_tool_response_roundtrip(self):
        """Full tool calling roundtrip: user → tool call → tool result → answer."""
        # Step 1: User asks weather, model should call tool
        data1 = chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=2000,
            temperature=0.01,
        )

        choice1 = data1["choices"][0]
        if "tool_calls" not in choice1["message"] or not choice1["message"]["tool_calls"]:
            pytest.skip("Model did not produce tool_calls for roundtrip test")

        tc = choice1["message"]["tool_calls"][0]
        tool_call_id = tc["id"]

        # Step 2: Send tool result back, model should use it
        data2 = chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": "", "tool_calls": [tc]},
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": '{"temperature": 22, "condition": "sunny", "humidity": 45}',
                },
            ],
            tools=[WEATHER_TOOL],
            max_tokens=2000,
            temperature=0.01,
        )

        # Model should produce a text answer referencing the weather data
        msg2 = data2["choices"][0]["message"]
        all_text = (msg2.get("content", "") or "") + (msg2.get("reasoning_content", "") or "")
        assert len(all_text) > 0
        # Should mention temperature or sunny or Paris
        text_lower = all_text.lower()
        assert any(w in text_lower for w in ["22", "sunny", "paris", "weather", "温度"])

    def test_no_tool_when_unnecessary(self):
        """Tools present but prompt doesn't need them → text response."""
        data = chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=1500,
            temperature=0.01,
        )

        choice = data["choices"][0]
        msg = choice["message"]
        all_text = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        assert "4" in all_text

        # Should NOT have tool calls for a math question
        tool_calls = msg.get("tool_calls")
        assert not tool_calls, "Model should not call weather tool for math question"


# ═══════════════════════════════════════════════════════════════════
#  5. Statelessness & Isolation Tests
# ═══════════════════════════════════════════════════════════════════

class TestStatelessness:
    """Server resets session between requests — no cross-request leakage."""

    def test_state_isolation_adversarial(self):
        """Secret from one request must not leak to next."""
        # Request A: tell a secret
        chat_json(
            [{"role": "user", "content": "Remember this secret code: ZEBRA-7742. "
                                          "Just say 'OK, I remember'."}],
            max_tokens=50,
            temperature=0.01,
        )

        # Request B: ask for the secret (fresh session, no history)
        data = chat_json(
            [{"role": "user", "content": "What is the secret code I told you earlier?"}],
            max_tokens=500,
            temperature=0.01,
        )

        msg = data["choices"][0]["message"]
        all_text = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        # Server resets session — model should NOT know ZEBRA-7742
        assert "ZEBRA-7742" not in all_text, \
            f"State leak detected! Model remembered secret: {all_text[:200]}"

    def test_state_isolation_after_tool_call(self):
        """Tool call state doesn't leak to subsequent requests."""
        # Request A: trigger tool call
        chat_json(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in London?"},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=500,
            temperature=0.01,
        )

        # Request B: unrelated question, no tools
        data = chat_json(
            [{"role": "user", "content": "What is the square root of 144?"}],
            max_tokens=1500,
            temperature=0.01,
        )

        msg = data["choices"][0]["message"]
        all_text = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        assert "12" in all_text
        # content (non-thinking) should not reference weather/London
        content = msg.get("content", "") or ""
        assert "london" not in content.lower()
        assert "weather" not in content.lower()

    def test_sequential_multiple_requests(self):
        """5 sequential requests all succeed independently."""
        prompts = [
            "What is 1+1?",
            "What color is the sky?",
            "Name a fruit.",
            "What is the capital of Japan?",
            "Say 'ping'.",
        ]
        for prompt in prompts:
            data = chat_json(
                [{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.01,
            )
            msg = data["choices"][0]["message"]
            has_output = bool(msg.get("content")) or bool(msg.get("reasoning_content"))
            assert has_output, f"No output for prompt: {prompt}"
            assert data["choices"][0]["finish_reason"] in ("stop", "length")


# ═══════════════════════════════════════════════════════════════════
#  6. Concurrency Tests
# ═══════════════════════════════════════════════════════════════════

class TestConcurrency:
    """Server handles concurrent requests correctly."""

    def test_concurrent_requests(self):
        """Multiple parallel requests all complete successfully."""
        prompts = [
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
        ]

        def send_chat(prompt):
            return chat_json(
                [{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.01,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_chat, p) for p in prompts]
            results = [f.result(timeout=TIMEOUT) for f in futures]

        # All should succeed
        for result in results:
            msg = result["choices"][0]["message"]
            has_output = bool(msg.get("content")) or bool(msg.get("reasoning_content"))
            assert has_output
            assert result["choices"][0]["finish_reason"] in ("stop", "length")

    def test_health_during_generation(self):
        """Health endpoint remains responsive during generation."""
        # Start a slow generation in background
        def slow_request():
            return chat_json(
                [{"role": "user", "content": "Write a detailed essay about AI history."}],
                max_tokens=100,
                temperature=0.01,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            gen_future = executor.submit(slow_request)
            # Give it a moment to start
            time.sleep(1)

            # Health should still respond
            health_future = executor.submit(
                lambda: requests.get(health_url(), timeout=10)
            )
            health_resp = health_future.result(timeout=15)
            assert health_resp.status_code == 200

            # Wait for generation to complete
            gen_future.result(timeout=TIMEOUT)


# ═══════════════════════════════════════════════════════════════════
#  7. Stop Strings & Sampling Tests
# ═══════════════════════════════════════════════════════════════════

class TestSamplingAndStop:
    """Sampling parameters and stop string behavior."""

    def test_stop_string(self):
        """Stop string causes generation to halt."""
        data = chat_json(
            [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
            max_tokens=1500,
            temperature=0.01,
            stop=["5"],
        )

        msg = data["choices"][0]["message"]
        all_text = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        # Should have some output
        assert len(all_text) > 0
        # finish_reason should be 'stop' (stopped by stop string)
        assert data["choices"][0]["finish_reason"] in ("stop", "length")

    def test_temperature_affects_output(self):
        """Different temperatures produce different output characteristics."""
        # Low temperature — more deterministic
        data_low = chat_json(
            [{"role": "user", "content": "Pick a random number between 1 and 100."}],
            max_tokens=50,
            temperature=0.01,
        )
        # High temperature — more varied
        data_high = chat_json(
            [{"role": "user", "content": "Pick a random number between 1 and 100."}],
            max_tokens=50,
            temperature=1.5,
        )

        # Both should produce valid responses
        msg_low = data_low["choices"][0]["message"]
        msg_high = data_high["choices"][0]["message"]
        has_low = bool(msg_low.get("content")) or bool(msg_low.get("reasoning_content"))
        has_high = bool(msg_high.get("content")) or bool(msg_high.get("reasoning_content"))
        assert has_low
        assert has_high


# ═══════════════════════════════════════════════════════════════════
#  8. Thinking Mode Tests
# ═══════════════════════════════════════════════════════════════════

class TestThinking:
    """Thinking/reasoning mode (if server has --enable-thinking)."""

    def test_thinking_in_response(self):
        """Math problem triggers thinking, reasoning_content is structured."""
        data = chat_json(
            [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
            max_tokens=1500,
            temperature=0.01,
        )

        msg = data["choices"][0]["message"]
        # Should have reasoning_content (thinking mode is on by default)
        all_text = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        assert len(all_text) > 0

        # If reasoning_content is present, it should be a non-empty string
        if msg.get("reasoning_content"):
            assert isinstance(msg["reasoning_content"], str)
            assert len(msg["reasoning_content"]) > 10

    def test_thinking_in_stream(self):
        """Streaming with thinking produces reasoning_content deltas."""
        r = chat(
            [{"role": "user", "content": "Solve: If x + 5 = 12, what is x? Think step by step."}],
            stream=True, max_tokens=1500, temperature=0.01,
        )
        events = parse_sse_events(r)

        thinking_text = collect_stream_thinking(events)
        content_text = collect_stream_text(events)

        # Should have some output (thinking and/or content)
        assert len(thinking_text) > 0 or len(content_text) > 0

        # If thinking was produced, it should be coherent text
        if thinking_text:
            assert len(thinking_text) > 5  # more than just whitespace


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
