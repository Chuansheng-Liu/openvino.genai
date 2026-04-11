#!/usr/bin/env python3
"""
Comprehensive stress test for ov_serve.
Tests all combinations: text-only, VL, multi-image, streaming/non-streaming,
prefix cache hits, thinking tag filtering, long responses, and session resets.
"""
import json, http.client, base64, io, time, sys, traceback

SERVER = "127.0.0.1"
PORT = 8080

PASS = 0
FAIL = 0
ERRORS = []

def chat(messages, max_tokens=60, stream=False):
    payload = json.dumps({
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": stream,
    })
    conn = http.client.HTTPConnection(SERVER, PORT, timeout=300)
    conn.request("POST", "/v1/chat/completions",
                 body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    if resp.status != 200:
        return None, None, f"HTTP {resp.status}: {data[:200]}"

    if not stream:
        r = json.loads(data)
        content = r["choices"][0]["message"]["content"]
        perf = r["usage"].get("performance", {})
        return content, perf, None
    else:
        # Parse SSE stream
        content_parts = []
        perf = {}
        for line in data.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content_parts.append(delta["content"])
            if "usage" in chunk:
                perf = chunk["usage"].get("performance", {})
        return "".join(content_parts), perf, None


def make_image(width, height, color, shape="rect"):
    """Create test images with different sizes/shapes"""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    margin = min(width, height) // 10
    if shape == "rect":
        draw.rectangle([margin, margin, width-margin, height-margin],
                       fill=color, outline='black', width=2)
    elif shape == "circle":
        draw.ellipse([margin, margin, width-margin, height-margin],
                     fill=color, outline='black', width=2)
    elif shape == "triangle":
        draw.polygon([(width//2, margin), (margin, height-margin),
                      (width-margin, height-margin)],
                     fill=color, outline='black', width=2)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def fmt_perf(perf):
    if not perf:
        return "no perf data"
    c = perf.get("prefix_cached_tokens", 0)
    ttft = perf.get("ttft_ms", 0)
    tps = perf.get("throughput_tps", 0)
    return f"ttft={ttft:>8.1f}ms  cached={c:>5}  tps={tps:>5.1f}"


def check(condition, msg):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
    else:
        FAIL += 1
        ERRORS.append(msg)
        print(f"    ❌ FAIL: {msg}")


def run_turn(turn_num, label, messages, user_content, max_tokens=80,
             image_uris=None, stream=False, expect_cache=None,
             expect_no_think_tag=True, expect_min_len=5):
    """Run one conversation turn with comprehensive checks."""
    # Build user message
    if image_uris:
        content_parts = []
        for uri in (image_uris if isinstance(image_uris, list) else [image_uris]):
            content_parts.append({"type": "image_url", "image_url": {"url": uri}})
        content_parts.append({"type": "text", "text": user_content})
        user_msg = {"role": "user", "content": content_parts}
    else:
        user_msg = {"role": "user", "content": user_content}

    messages.append(user_msg)
    mode = "stream" if stream else "sync"
    t0 = time.time()
    try:
        content, perf, err = chat(messages, max_tokens, stream=stream)
    except Exception as e:
        print(f"  T{turn_num:>2} [{label:>12}] [{mode:>6}] EXCEPTION: {e}")
        messages.pop()
        check(False, f"T{turn_num} {label}: exception {e}")
        return False
    elapsed = time.time() - t0

    if err:
        print(f"  T{turn_num:>2} [{label:>12}] [{mode:>6}] ERROR: {err}")
        messages.pop()
        check(False, f"T{turn_num} {label}: {err}")
        return False

    cached = perf.get("prefix_cached_tokens", 0) if perf else 0
    short = (content or "")[:70].replace('\n', ' ')
    print(f"  T{turn_num:>2} [{label:>12}] [{mode:>6}]  {fmt_perf(perf)}  \"{short}...\"")

    # Checks
    check(content is not None and len(content) >= expect_min_len,
          f"T{turn_num} {label}: content too short ({len(content or '')} chars)")

    if expect_no_think_tag:
        check("</think>" not in (content or ""),
              f"T{turn_num} {label}: </think> tag leaked into content")
        check("<think>" not in (content or ""),
              f"T{turn_num} {label}: <think> tag leaked into content")

    if expect_cache is True:
        check(cached > 0,
              f"T{turn_num} {label}: expected cache hit but cached={cached}")
    elif expect_cache is False:
        pass  # Don't check, first turn or VL new image

    # Check for content duplication (thinking tag bug)
    if content and len(content) > 100:
        half = len(content) // 2
        first_half = content[:half]
        second_half = content[half:]
        # If > 80% of first half appears in second half, likely duplication
        overlap = sum(1 for a, b in zip(first_half, second_half) if a == b)
        dup_ratio = overlap / max(len(first_half), 1)
        check(dup_ratio < 0.7,
              f"T{turn_num} {label}: possible content duplication (ratio={dup_ratio:.2f})")

    messages.append({"role": "assistant", "content": content})
    return True


def test_text_only_conversation():
    """Test pure text multi-turn with prefix cache."""
    print("\n" + "═" * 90)
    print("TEST 1: Text-only multi-turn (prefix cache)")
    print("═" * 90)

    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    run_turn(1, "text-first", msgs, "What is 2+2?",
             expect_cache=False)
    run_turn(2, "text-cache", msgs, "And 3+3?",
             expect_cache=True)
    run_turn(3, "text-cache", msgs, "What about 10+10?",
             expect_cache=True)
    # Streaming
    run_turn(4, "text-stream", msgs, "Now 100+100?",
             stream=True, expect_cache=True)
    run_turn(5, "text-stream", msgs, "Summarize all our math.",
             stream=True, max_tokens=150, expect_cache=True)


def test_text_long_response():
    """Test that long responses (max_tokens=2048) don't leak </think>."""
    print("\n" + "═" * 90)
    print("TEST 2: Long response — thinking tag filter (max_tokens=2048)")
    print("═" * 90)

    msgs = [{"role": "system", "content": "You are a helpful assistant."}]

    # Non-streaming long
    run_turn(1, "long-sync", msgs, "Who are you? Introduce yourself in detail.",
             max_tokens=2048, expect_cache=False)

    # Streaming long
    run_turn(2, "long-stream", msgs, "Now tell me about your capabilities in detail.",
             max_tokens=2048, stream=True, expect_cache=True)


def test_vl_single_image():
    """Test VL with single image, then text follow-ups."""
    print("\n" + "═" * 90)
    print("TEST 3: VL single image + text follow-ups")
    print("═" * 90)

    img = make_image(300, 200, 'red', 'circle')
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    run_turn(1, "VL-single", msgs, "What shape and color is in this image?",
             image_uris=img, expect_cache=False)
    # Text follow-up after VL (should cache since same images)
    run_turn(2, "text-after-VL", msgs, "What was the background color?",
             expect_cache=True)
    run_turn(3, "text-after-VL", msgs, "Describe the image again briefly.",
             expect_cache=True)
    # Streaming follow-up
    run_turn(4, "stream-VL", msgs, "How many shapes were there?",
             stream=True, expect_cache=True)


def test_vl_multi_image():
    """Test VL with multiple images in one conversation."""
    print("\n" + "═" * 90)
    print("TEST 4: VL multi-image conversation")
    print("═" * 90)

    img1 = make_image(200, 200, 'red', 'rect')
    img2 = make_image(300, 300, 'blue', 'circle')
    img3 = make_image(250, 150, 'green', 'triangle')
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    # First image
    run_turn(1, "VL-img1", msgs, "What shape and color is in this image?",
             image_uris=img1, expect_cache=False)
    # Second image (different size, different shape)
    run_turn(2, "VL-img2", msgs, "Now describe this new image.",
             image_uris=img2)
    # Text follow-up referencing both
    run_turn(3, "text-2img", msgs, "Compare the two images.",
             expect_cache=True)
    # Third image
    run_turn(4, "VL-img3", msgs, "Describe this third image.",
             image_uris=img3)
    # Follow-up about all 3
    run_turn(5, "text-3img", msgs, "List all 3 images we looked at.",
             max_tokens=150, expect_cache=True)


def test_vl_multi_image_single_message():
    """Test multiple images in a single user message."""
    print("\n" + "═" * 90)
    print("TEST 5: Multiple images in single message")
    print("═" * 90)

    img1 = make_image(200, 200, 'orange', 'rect')
    img2 = make_image(300, 200, 'purple', 'triangle')
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    run_turn(1, "VL-2-in-1", msgs, "Compare these two images.",
             image_uris=[img1, img2], expect_cache=False)
    run_turn(2, "text-after", msgs, "Which had more sides?",
             expect_cache=True)


def test_text_to_vl_transition():
    """Test transition from text-only turns to VL turns."""
    print("\n" + "═" * 90)
    print("TEST 6: Text → VL → Text → VL transitions")
    print("═" * 90)

    img1 = make_image(200, 200, 'cyan', 'circle')
    img2 = make_image(300, 300, 'magenta', 'rect')
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    # Start text
    run_turn(1, "text-start", msgs, "Hello! What is Python?",
             expect_cache=False)
    run_turn(2, "text-cache", msgs, "What's it used for?",
             expect_cache=True)
    # Switch to VL
    run_turn(3, "VL-switch", msgs, "Now describe this image.",
             image_uris=img1)
    # Back to text
    run_turn(4, "text-back", msgs, "What color was the shape?",
             expect_cache=True)
    # Another VL
    run_turn(5, "VL-again", msgs, "And this one?",
             image_uris=img2)
    # Final text
    run_turn(6, "text-final", msgs, "Compare the two images.",
             expect_cache=True)


def test_session_reset():
    """Test that /clear properly resets and subsequent turns work."""
    print("\n" + "═" * 90)
    print("TEST 7: Session reset (clear history)")
    print("═" * 90)

    img = make_image(200, 200, 'yellow', 'triangle')
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    # Build up some history
    run_turn(1, "text", msgs, "Remember: the password is 'banana'.",
             expect_cache=False)
    run_turn(2, "text-cache", msgs, "What's the password?",
             expect_cache=True)

    # Reset
    print("  ── /clear ──")
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."}]

    run_turn(3, "text-reset", msgs, "What's the password?",
             expect_cache=False)
    run_turn(4, "text-cache", msgs, "Never mind. What is 5+5?",
             expect_cache=True)
    # VL after reset
    run_turn(5, "VL-reset", msgs, "Describe this image.",
             image_uris=img)
    run_turn(6, "text-VL", msgs, "What shape was it?",
             expect_cache=True)


def test_streaming_consistency():
    """Test that streaming and non-streaming give similar results."""
    print("\n" + "═" * 90)
    print("TEST 8: Streaming vs non-streaming consistency")
    print("═" * 90)

    # Non-streaming
    msgs1 = [{"role": "system", "content": "You are a helpful assistant. Be concise."},
              {"role": "user", "content": "What is 7 * 8?"}]
    content1, perf1, err1 = chat(msgs1, max_tokens=30, stream=False)
    print(f"  Non-stream: \"{(content1 or '')[:60]}\"")
    check(err1 is None, f"Non-streaming error: {err1}")
    check("56" in (content1 or ""), "Non-streaming: expected '56' in response")

    # Streaming
    msgs2 = [{"role": "system", "content": "You are a helpful assistant. Be concise."},
              {"role": "user", "content": "What is 7 * 8?"}]
    content2, perf2, err2 = chat(msgs2, max_tokens=30, stream=True)
    print(f"  Streaming:   \"{(content2 or '')[:60]}\"")
    check(err2 is None, f"Streaming error: {err2}")
    check("56" in (content2 or ""), "Streaming: expected '56' in response")


def test_thinking_tag_max_tokens():
    """Test thinking tag doesn't leak at various max_tokens values."""
    print("\n" + "═" * 90)
    print("TEST 9: Thinking tag filter at various max_tokens")
    print("═" * 90)

    for mt in [64, 128, 256, 512, 1024, 2048]:
        msgs = [{"role": "user", "content": "Who are you?"}]
        content, perf, err = chat(msgs, max_tokens=mt, stream=False)
        has_think = "</think>" in (content or "") or "<think>" in (content or "")
        status = "❌" if has_think else "✅"
        toks = perf.get("throughput_tps", 0) if perf else 0
        clen = len(content or "")
        print(f"  max_tokens={mt:>5}  content_len={clen:>5}  {status}")
        check(not has_think, f"max_tokens={mt}: thinking tag leaked")

    # Also test streaming
    for mt in [512, 1024, 2048]:
        msgs = [{"role": "user", "content": "你好，你是谁？"}]
        content, perf, err = chat(msgs, max_tokens=mt, stream=True)
        has_think = "</think>" in (content or "") or "<think>" in (content or "")
        status = "❌" if has_think else "✅"
        clen = len(content or "")
        print(f"  max_tokens={mt:>5}  stream  content_len={clen:>5}  {status}")
        check(not has_think, f"max_tokens={mt} stream: thinking tag leaked")


def test_rapid_fire():
    """Test rapid consecutive requests."""
    print("\n" + "═" * 90)
    print("TEST 10: Rapid-fire 10 turns")
    print("═" * 90)

    msgs = [{"role": "system", "content": "Be concise. Answer in one sentence."}]
    questions = [
        "What is 1+1?", "What color is the sky?", "Capital of France?",
        "Who wrote Romeo and Juliet?", "What is H2O?",
        "Largest planet?", "Speed of light?", "What is DNA?",
        "Who painted Mona Lisa?", "What is pi?",
    ]
    for i, q in enumerate(questions):
        run_turn(i+1, f"rapid-{i+1}", msgs, q, max_tokens=40,
                 expect_cache=(i > 0))


def main():
    global PASS, FAIL, ERRORS
    print("=" * 90)
    print("  COMPREHENSIVE STRESS TEST — ov_serve")
    print(f"  Server: {SERVER}:{PORT}")
    print("=" * 90)

    # Health check
    try:
        conn = http.client.HTTPConnection(SERVER, PORT, timeout=10)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        data = resp.read().decode()
        conn.close()
        print(f"\n  Health: {data}")
    except Exception as e:
        print(f"\n  ❌ Server not reachable: {e}")
        sys.exit(1)

    tests = [
        test_text_only_conversation,
        test_text_long_response,
        test_thinking_tag_max_tokens,
        test_vl_single_image,
        test_vl_multi_image,
        test_vl_multi_image_single_message,
        test_text_to_vl_transition,
        test_session_reset,
        test_streaming_consistency,
        test_rapid_fire,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            ERRORS.append(f"{test_fn.__name__}: EXCEPTION {e}")
            print(f"  ❌ EXCEPTION in {test_fn.__name__}: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 90)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if ERRORS:
        print(f"\n  FAILURES:")
        for e in ERRORS:
            print(f"    ❌ {e}")
    else:
        print("  ✅ ALL CHECKS PASSED")
    print("=" * 90)
    sys.exit(1 if FAIL > 0 else 0)

if __name__ == "__main__":
    main()
