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


if __name__ == "__main__":
    test_fuse_synthesizes()
    test_fuse_judge_fallback()
    test_all_panel_dead_falls_back_to_judge_alone()
    test_http_endpoint_and_auth()
    print("\nALL TESTS PASSED")
