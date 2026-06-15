"""FreeLLMAPI Fusion layer — all-in-one.

One process exposing:
  * an OpenAI-compatible POST /v1/chat/completions endpoint (model = "fusion")
  * a Telegram long-poll bot

Both share the same fan-out + judge-synthesis logic, which mirrors
benchmark.py:run_systems exactly so the bot and the benchmark never diverge.

Run:
    ./venv/bin/python fusion_bot.py
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
import uvicorn


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _load_env_file(path):
    """Minimal .env loader — KEY=VALUE lines, no extra dependency."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.fusion"))

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
PANEL_MODELS = [m.strip() for m in os.environ.get("PANEL_MODELS", "").split(",") if m.strip()]
JUDGE_MODELS = [m.strip() for m in os.environ.get("JUDGE_MODEL", "").split(",") if m.strip()]
FUSION_API_KEY = os.environ.get("FUSION_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(",", " ").split() if x.strip().lstrip("-").isdigit()
}

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
PORT = int(os.environ.get("PORT", "8000"))

# Constants (no magic numbers scattered in the code).
UPSTREAM_RETRIES = 2
RATE_LIMIT_STATUS = 429
TELEGRAM_MAX_LEN = 4096
TELEGRAM_POLL_TIMEOUT = 30
HISTORY_LIMIT = 8  # messages kept per Telegram chat (user + assistant turns)
FUSION_MODEL_NAME = "fusion"

JUDGE_SYSTEM = (
    "You are the synthesizer in a model-fusion system. Several AI models each "
    "answered the same request. Find consensus, resolve contradictions, keep unique insights, "
    "cover blind spots, then write ONE best final answer. Keep the required answer format."
)


# --------------------------------------------------------------------------- #
# Core fusion logic (mirrors benchmark.py)                                     #
# --------------------------------------------------------------------------- #

def _content_to_text(content):
    """Normalize OpenAI message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts)
    return "" if content is None else str(content)


def _normalize_messages(messages):
    """Return a new list with all content coerced to text (immutable input)."""
    return [{"role": m["role"], "content": _content_to_text(m["content"])} for m in messages]


def _last_user_text(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


async def call_upstream(client, model, messages, sem, retries=UPSTREAM_RETRIES):
    """Single upstream call with 429-aware retries. Returns text or None."""
    payload = {"model": model, "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(retries + 1):
            try:
                r = await client.post(
                    f"{UPSTREAM_BASE_URL}/chat/completions",
                    json=payload, headers=headers, timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == RATE_LIMIT_STATUS and attempt < retries:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(attempt + 1)
                    continue
                return None


async def fuse(client, messages, sem):
    """Fan out to PANEL_MODELS, then synthesize via the JUDGE fallback chain.

    Returns the final fused answer string. Raises RuntimeError if nothing works.
    Mirrors benchmark.py:run_systems.
    """
    convo = _normalize_messages(messages)
    request_text = _last_user_text(convo)

    panel_answers = await asyncio.gather(*[call_upstream(client, m, convo, sem) for m in PANEL_MODELS])
    good = [(m, a) for m, a in zip(PANEL_MODELS, panel_answers) if a and a.strip()]

    if not good:
        # No panel member answered — fall back to the judge chain answering alone.
        for jm in JUDGE_MODELS:
            answer = await call_upstream(client, jm, convo, sem)
            if answer and answer.strip():
                return answer
        raise RuntimeError("All panel members and judges failed to respond.")

    panel_block = "\n\n".join(f"### Model {i+1} ({m})\n{a}" for i, (m, a) in enumerate(good))
    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"REQUEST:\n{request_text}\n\nPANEL ANSWERS:\n{panel_block}\n\n"
            "Write the single best final answer."
        )},
    ]
    for jm in JUDGE_MODELS:
        fusion = await call_upstream(client, jm, judge_messages, sem)
        if fusion and fusion.strip():
            return fusion

    # Judges all failed — degrade gracefully to the best available panel answer.
    return good[0][1]


# --------------------------------------------------------------------------- #
# OpenAI-compatible HTTP API                                                   #
# --------------------------------------------------------------------------- #

class ChatMessage(BaseModel):
    role: str
    content: object  # str, or OpenAI content-parts list


class ChatRequest(BaseModel):
    model: str = FUSION_MODEL_NAME
    messages: list[ChatMessage]


def _openai_response(text):
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": FUSION_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def require_fusion_key(authorization: str = Header(default="")):
    """Bearer auth against FUSION_API_KEY."""
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization.strip()
    if token != FUSION_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tg_task = None
    if TELEGRAM_TOKEN:
        tg_task = asyncio.create_task(telegram_loop(app.state.client, app.state.sem))
        print("[fusion] Telegram bot started")
    else:
        print("[fusion] TELEGRAM_TOKEN not set — Telegram bot disabled")
    try:
        yield
    finally:
        if tg_task:
            tg_task.cancel()
        await app.state.client.aclose()


app = FastAPI(title="FreeLLMAPI Fusion", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "panel": PANEL_MODELS, "judges": JUDGE_MODELS}


@app.get("/v1/models")
async def list_models(_=Depends(require_fusion_key)):
    return {"object": "list", "data": [
        {"id": FUSION_MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "fusion"},
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, _=Depends(require_fusion_key)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        text = await fuse(app.state.client, messages, app.state.sem)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _openai_response(text)


# --------------------------------------------------------------------------- #
# Telegram long-poll bot                                                       #
# --------------------------------------------------------------------------- #

_tg_history: dict[int, list[dict]] = {}


def _tg_url(method):
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


async def _tg_send(client, chat_id, text):
    for start in range(0, max(len(text), 1), TELEGRAM_MAX_LEN):
        chunk = text[start:start + TELEGRAM_MAX_LEN]
        try:
            await client.post(_tg_url("sendMessage"), json={"chat_id": chat_id, "text": chunk}, timeout=30)
        except Exception as exc:
            print(f"[telegram] send failed: {exc}")
            return


def _is_allowed(user_id):
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def _handle_update(client, sem, update):
    message = update.get("message") or update.get("edited_message")
    if not message:
        return
    chat_id = message["chat"]["id"]
    user_id = (message.get("from") or {}).get("id", 0)
    text = (message.get("text") or "").strip()
    if not text:
        return
    if not _is_allowed(user_id):
        await _tg_send(client, chat_id, "Not authorized.")
        return
    if text in ("/start", "/help"):
        await _tg_send(client, chat_id, "FreeLLMAPI Fusion bot. Send a prompt; I fan it out and synthesize one answer. /reset clears history.")
        return
    if text == "/reset":
        _tg_history.pop(chat_id, None)
        await _tg_send(client, chat_id, "History cleared.")
        return

    history = _tg_history.get(chat_id, [])
    convo = (history + [{"role": "user", "content": text}])[-HISTORY_LIMIT:]
    try:
        answer = await fuse(client, convo, sem)
    except RuntimeError as exc:
        await _tg_send(client, chat_id, f"Upstream error: {exc}")
        return
    _tg_history[chat_id] = (convo + [{"role": "assistant", "content": answer}])[-HISTORY_LIMIT:]
    await _tg_send(client, chat_id, answer)


async def telegram_loop(client, sem):
    """Long-poll getUpdates and dispatch each message through fuse()."""
    offset = 0
    while True:
        try:
            r = await client.get(
                _tg_url("getUpdates"),
                params={"offset": offset, "timeout": TELEGRAM_POLL_TIMEOUT},
                timeout=TELEGRAM_POLL_TIMEOUT + 10,
            )
            updates = r.json().get("result", [])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[telegram] poll error: {exc}")
            await asyncio.sleep(3)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                await _handle_update(client, sem, update)
            except Exception as exc:
                print(f"[telegram] handler error: {exc}")


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #

def _validate_config():
    missing = []
    if not UPSTREAM_BASE_URL:
        missing.append("UPSTREAM_BASE_URL")
    if not UPSTREAM_API_KEY:
        missing.append("UPSTREAM_API_KEY")
    if not PANEL_MODELS:
        missing.append("PANEL_MODELS")
    if not JUDGE_MODELS:
        missing.append("JUDGE_MODEL")
    if not FUSION_API_KEY:
        missing.append("FUSION_API_KEY")  # required — this proxy must not run unauthenticated
    if missing:
        raise SystemExit("Missing required env vars: " + ", ".join(missing))


def main():
    _validate_config()
    if not ALLOWED_USER_IDS and TELEGRAM_TOKEN:
        print("[fusion] WARNING: ALLOWED_USER_IDS empty — Telegram bot replies to anyone.")
    print(f"[fusion] panel={PANEL_MODELS}")
    print(f"[fusion] judges={JUDGE_MODELS}")
    print(f"[fusion] listening on :{PORT}  (model='{FUSION_MODEL_NAME}')")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
