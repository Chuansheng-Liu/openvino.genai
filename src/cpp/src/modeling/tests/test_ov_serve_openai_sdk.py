# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# OpenAI SDK integration tests for ov_serve.
#
# Prerequisites:
#   pip install pytest openai requests
#   Start server: ov_serve.exe --model <path-to-Qwen3.5> --port 8080 [--vl]
#
# Run:
#   pytest test_ov_serve_openai_sdk.py -v
#   OV_SERVE_URL=http://host:port pytest test_ov_serve_openai_sdk.py -v
#
# Environment variables:
#   OV_SERVE_URL      — server base URL (default: http://localhost:8080)
#   OV_SERVE_TIMEOUT  — request timeout in seconds (default: 120)
#   OV_SERVE_API_KEY  — API key for OpenAI SDK (default: dummy)
#   OV_SERVE_MODEL    — model name to send in requests (default: qwen3.5)

import base64
import os
import struct

import pytest
import requests
from openai import OpenAI


BASE_URL = os.environ.get("OV_SERVE_URL", "http://localhost:8080").rstrip("/")
SERVER_ROOT = BASE_URL[:-3] if BASE_URL.endswith("/v1") else BASE_URL
OPENAI_BASE_URL = BASE_URL if BASE_URL.endswith("/v1") else f"{BASE_URL}/v1"
TIMEOUT = int(os.environ.get("OV_SERVE_TIMEOUT", "120"))
API_KEY = os.environ.get("OV_SERVE_API_KEY", "dummy")
MODEL = os.environ.get("OV_SERVE_MODEL", "qwen3.5")


def health_url():
    return f"{SERVER_ROOT}/health"


def make_client():
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=API_KEY, timeout=TIMEOUT, max_retries=0)


def model_extra(obj):
    return getattr(obj, "model_extra", {}) or {}


def make_test_image_base64():
    """Create a valid 64x64 red BMP as base64 data URI.

    Tiny images can fail Qwen3.5 VL preprocess ("Height and width must be >= resize factor"),
    so keep this comfortably above the minimum.
    """
    width, height = 64, 64
    row_stride = ((width * 3 + 3) // 4) * 4
    image_size = row_stride * height
    file_size = 14 + 40 + image_size

    bmp = bytearray()
    bmp += b"BM"
    bmp += struct.pack("<IHHI", file_size, 0, 0, 54)
    bmp += struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, image_size, 2835, 2835, 0, 0)
    pixel_row = (b"\x00\x00\xff" * width) + (b"\x00" * (row_stride - width * 3))
    bmp += pixel_row * height

    return "data:image/bmp;base64," + base64.b64encode(bmp).decode("ascii")


def has_message_output(message):
    if getattr(message, "content", None):
        return True
    if model_extra(message).get("reasoning_content"):
        return True
    if getattr(message, "tool_calls", None):
        return True
    return False


def is_vl_enabled():
    try:
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": make_test_image_base64()}},
                            {"type": "text", "text": "hi"},
                        ],
                    }
                ],
                "max_tokens": 5,
            },
            timeout=min(TIMEOUT, 30),
        )
        if response.status_code == 400:
            msg = response.json().get("error", {}).get("message", "")
            if "--vl" in msg:
                return False
        return response.status_code == 200
    except Exception:
        return False


def require_vl_enabled():
    if not is_vl_enabled():
        pytest.skip("Server not started with --vl (vision not enabled)")


@pytest.fixture(scope="session", autouse=True)
def server_ready():
    try:
        response = requests.get(health_url(), timeout=10)
        if response.status_code != 200:
            pytest.skip(f"ov_serve health check failed: {response.status_code}")
    except requests.ConnectionError:
        pytest.skip(f"ov_serve not reachable at {SERVER_ROOT}")
    except requests.Timeout:
        pytest.skip(f"ov_serve health check timed out at {SERVER_ROOT}")


@pytest.fixture(scope="session")
def client():
    return make_client()


class TestOpenAISDKCompatibility:
    def test_models_list(self, client):
        models = client.models.list()
        assert len(models.data) >= 1
        assert any(model.id for model in models.data)

    def test_chat_completion_basic(self, client):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=256,
            temperature=0.01,
        )

        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert has_message_output(choice.message)
        assert choice.finish_reason in ("stop", "length", "tool_calls")
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens >= response.usage.prompt_tokens

    def test_chat_completion_streaming(self, client):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with a short greeting."}],
            max_tokens=256,
            temperature=0.01,
            stream=True,
        )

        roles = []
        content_parts = []
        reasoning_parts = []
        finish_reasons = []

        for chunk in stream:
            assert chunk.object == "chat.completion.chunk"
            assert len(chunk.choices) == 1

            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "role", None):
                roles.append(delta.role)
            if getattr(delta, "content", None):
                content_parts.append(delta.content)

            reasoning = model_extra(delta).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)

            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        assert roles and roles[0] == "assistant"
        assert content_parts or reasoning_parts
        assert finish_reasons
        assert finish_reasons[-1] in ("stop", "length", "tool_calls")

    def test_tool_call_structure(self, client):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Beijing?"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"}
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            max_tokens=512,
            temperature=0.01,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        if not tool_calls:
            pytest.skip("Model did not produce tool_calls for this prompt")

        tool_call = tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments
        assert choice.finish_reason == "tool_calls"

    def test_vl_chat_completion(self, client):
        require_vl_enabled()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": make_test_image_base64()}},
                        {"type": "text", "text": "What color is this image? Answer briefly."},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.01,
        )

        choice = response.choices[0]
        assert has_message_output(choice.message)
        assert response.usage.prompt_tokens > 50

    def test_vl_streaming(self, client):
        require_vl_enabled()
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": make_test_image_base64()}},
                        {"type": "text", "text": "Describe this image briefly."},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.01,
            stream=True,
        )

        content_parts = []
        reasoning_parts = []
        finish_reasons = []

        for chunk in stream:
            choice = chunk.choices[0]
            if getattr(choice.delta, "content", None):
                content_parts.append(choice.delta.content)

            reasoning = model_extra(choice.delta).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)

            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        assert content_parts or reasoning_parts
        assert finish_reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
