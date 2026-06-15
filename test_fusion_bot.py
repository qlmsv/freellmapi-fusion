"""Offline functional test for fusion_bot — no network, upstream mocked."""
import asyncio
import os

# Minimal env so the module imports and validates.
os.environ.update({
    "UPSTREAM_BASE_URL": "http://x/v1", "UPSTREAM_API_KEY": "u",
    "PANEL_MODELS": "m1,m2,m3", "JUDGE_MODEL": "j1,j2",
    "FUSION_API_KEY": "secret",
})

import fusion_bot as fb

captured = []

async def fake_call(client, model, messages, sem, retries=2):
    captured.append(model)
    if model == "m2":
        return None              # simulate a dead panel member
    if model == "j1":
        return "FUSED: " + messages[-1]["content"][:20]
    return f"answer from {model}"

fb.call_upstream = fake_call


def _run_fuse(msgs):
    async def runner():
        return await fb.fuse(None, msgs, asyncio.Semaphore(4))
    return asyncio.run(runner())


def test_fuse_synthesizes():
    captured.clear()
    out = _run_fuse([{"role": "user", "content": "What is 2+2?"}])
    assert out.startswith("FUSED:"), out
    assert "m1" in captured and "m2" in captured and "m3" in captured  # all panel fanned out
    assert "j1" in captured                                            # judge ran
    print("PASS test_fuse_synthesizes ->", out)


def test_fuse_judge_fallback():
    captured.clear()
    async def judge_first_dies(client, model, messages, sem, retries=2):
        if model == "j1":
            return None
        if model == "j2":
            return "J2-FUSED"
        return f"panel {model}"
    fb.call_upstream = judge_first_dies
    out = _run_fuse([{"role": "user", "content": "hi"}])
    assert out == "J2-FUSED", out
    fb.call_upstream = fake_call
    print("PASS test_fuse_judge_fallback ->", out)


def test_all_panel_dead_falls_back_to_judge_alone():
    async def all_panel_dead(client, model, messages, sem, retries=2):
        if model.startswith("m"):
            return None
        return "JUDGE-ALONE" if model == "j1" else None
    fb.call_upstream = all_panel_dead
    out = _run_fuse([{"role": "user", "content": "hi"}])
    assert out == "JUDGE-ALONE", out
    fb.call_upstream = fake_call
    print("PASS test_all_panel_dead_falls_back_to_judge_alone ->", out)


def test_http_endpoint_and_auth():
    from fastapi.testclient import TestClient
    with TestClient(fb.app) as client:
        # missing/wrong key -> 401
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401, r.status_code
        # correct key -> OpenAI-shaped response
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "fusion", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, (r.status_code, r.text)
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"].startswith("FUSED:"), body
    print("PASS test_http_endpoint_and_auth")


def test_tools_passthrough_returns_tool_calls_and_skips_fusion():
    """A request carrying `tools` must bypass fuse() and forward raw to a tool model."""
    from fastapi.testclient import TestClient
    seen = {}

    async def fake_raw(client, model, messages, extra, sem, retries=2):
        seen["model"] = model
        seen["messages"] = messages
        seen["extra"] = extra
        return {
            "id": "x", "object": "chat.completion", "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
                "finish_reason": "tool_calls",
            }],
        }

    async def must_not_run(*a, **k):
        raise AssertionError("fuse() must not be called when tools are present")

    fb.call_upstream_raw = fake_raw
    orig_fuse = fb.fuse
    fb.fuse = must_not_run
    try:
        with TestClient(fb.app) as client:
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret"},
                json={
                    "model": "fusion",
                    "messages": [
                        {"role": "user", "content": "weather in Paris?"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [{"id": "c0", "type": "function",
                                         "function": {"name": "x", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "c0", "content": "sunny"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                    "tool_choice": "auto",
                },
            )
        assert r.status_code == 200, (r.status_code, r.text)
        body = r.json()
        assert body["choices"][0]["finish_reason"] == "tool_calls", body
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather", body
        assert body["model"] == "fusion", body
        # first model in the TOOL_MODELS chain was used
        assert seen["model"] == fb.TOOL_MODELS[0], seen["model"]
        # tools + tool_choice were forwarded
        assert seen["extra"].get("tools") and seen["extra"].get("tool_choice") == "auto", seen["extra"]
        # tool history fields preserved (assistant.tool_calls + tool.tool_call_id)
        roles = [m["role"] for m in seen["messages"]]
        assert roles == ["user", "assistant", "tool"], roles
        assert seen["messages"][1].get("tool_calls"), "assistant tool_calls dropped"
        assert seen["messages"][2].get("tool_call_id") == "c0", "tool_call_id dropped"
    finally:
        fb.fuse = orig_fuse
    print("PASS test_tools_passthrough_returns_tool_calls_and_skips_fusion")


def test_tools_passthrough_skips_garbage_and_falls_through():
    """A finish_reason='length' response with no tool_calls is garbage -> try next model."""
    from fastapi.testclient import TestClient
    used = []

    async def fake_raw(client, model, messages, extra, sem, retries=2):
        used.append(model)
        if model == fb.TOOL_MODELS[0]:
            # garbage: rambled to max_tokens, no tool_calls
            return {"choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": "随处可见 garbage \n ]["}}]}
        return {"choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Paris is 22C and sunny."}}]}

    fb.call_upstream_raw = fake_raw
    with TestClient(fb.app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "fusion",
                  "messages": [{"role": "user", "content": "weather?"}],
                  "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        )
    assert r.status_code == 200, (r.status_code, r.text)
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Paris is 22C and sunny.", body
    assert used == fb.TOOL_MODELS[:2], used  # skipped #1 (garbage), used #2
    print("PASS test_tools_passthrough_skips_garbage_and_falls_through")


def test_usable_tool_response_predicate():
    assert fb._usable_tool_response({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
    assert fb._usable_tool_response({"choices": [{"finish_reason": "tool_calls",
                                                  "message": {"tool_calls": [{"id": "x"}]}}]})
    assert not fb._usable_tool_response({"choices": [{"finish_reason": "length", "message": {"content": "ramble"}}]})
    assert not fb._usable_tool_response({"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]})
    assert not fb._usable_tool_response(None)
    print("PASS test_usable_tool_response_predicate")


def test_streaming_fusion_returns_sse():
    """stream=true must return an SSE stream (text/event-stream), not a JSON body.

    A JSON body reads as an empty response to streaming clients (the Hermes agent),
    which was the real cause of 'Empty response from model'.
    """
    from fastapi.testclient import TestClient
    fb.call_upstream = fake_call
    with TestClient(fb.app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "fusion", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.headers["content-type"].startswith("text/event-stream"), r.headers.get("content-type")
    body = r.text
    assert "chat.completion.chunk" in body, body
    assert "FUSED:" in body, body              # the computed answer rides in a delta
    assert "data: [DONE]" in body, body
    print("PASS test_streaming_fusion_returns_sse")


def test_streaming_tool_calls_sse():
    """Tool passthrough under stream=true must emit tool_calls deltas with an index."""
    from fastapi.testclient import TestClient

    async def fake_raw(client, model, messages, extra, sem, retries=2):
        return {"choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None,
                                         "tool_calls": [{"id": "c1", "type": "function",
                                                         "function": {"name": "get_weather", "arguments": "{}"}}]}}]}

    fb.call_upstream_raw = fake_raw
    with TestClient(fb.app) as client:
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "fusion", "messages": [{"role": "user", "content": "hi"}],
                  "tools": [{"type": "function", "function": {"name": "get_weather"}}], "stream": True},
        )
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.headers["content-type"].startswith("text/event-stream"), r.headers.get("content-type")
    body = r.text
    assert "tool_calls" in body and "get_weather" in body, body
    assert '"index": 0' in body, body
    assert "data: [DONE]" in body, body
    print("PASS test_streaming_tool_calls_sse")


if __name__ == "__main__":
    test_fuse_synthesizes()
    test_fuse_judge_fallback()
    test_all_panel_dead_falls_back_to_judge_alone()
    test_http_endpoint_and_auth()
    test_tools_passthrough_returns_tool_calls_and_skips_fusion()
    test_tools_passthrough_skips_garbage_and_falls_through()
    test_usable_tool_response_predicate()
    test_streaming_fusion_returns_sse()
    test_streaming_tool_calls_sse()
    print("\nALL TESTS PASSED")
