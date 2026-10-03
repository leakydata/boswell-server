"""A language model on this computer, for the phone's assistant.

  GET  /v1/llm                  no key: is the local model there, and which one
  POST /v1/chat/completions     (key) OpenAI-style chat, tools included, answered by Ollama

The phone speaks the OpenAI chat-completions API to OpenRouter; pointed here
instead, the same request is answered by a model running in Ollama on this
computer, so questions, titles and summaries never leave the house. Ollama
speaks the same API (http://127.0.0.1:11434/v1), so this mostly passes the
request through: it picks the model, drops what only OpenRouter understands,
and reports the cost as 0.

  BOSWELL_LLM_MODEL       the Ollama model (default gemma4:e4b; `ollama list` shows what's installed)
  BOSWELL_LLM_THINK       hidden thinking: none (default), low, medium or high
  BOSWELL_LLM_KEEP_ALIVE  how long it stays in graphics memory after the last call (default 10m),
                          so it doesn't hold memory the other programs on this GPU need all day
  BOSWELL_OLLAMA          where Ollama is (default http://127.0.0.1:11434)

Why gemma4:e4b with thinking off (measured 2026-10-02 on the RTX 4090 with ~12.7 GB already
taken by desktop Boswell and this server's own models): it fits whole on the GPU (~5 GB at
Ollama's 32k context) and answers a tool call in ~2 s with a fresh prompt (0.2-0.7 s when the
prompt's start is cached), 15 of 15 tool calls right, titles and the watcher's JSON 6 of 6.
Loading it takes 13-25 s, minutes the very first time when it must come off the disk. Thinking on (Ollama's default for it), the same calls took
1-4 s and short JSON replies came back empty: the thinking used up their 200-400 tokens.
gpt-oss:20b was as accurate but spills 76% onto the CPU next to the other programs (4-12 s a
call, 3+ minutes to load); gemma4:12b-it-q8_0 and qwen3.6:27b spill too (25-220 s a call);
phi4 can't call tools in Ollama.

Ollama's context window is its own setting (OLLAMA_CONTEXT_LENGTH; 32k here): the
assistant's tool list alone is ~2,900 tokens, so keep it at 16k or more.
"""
import os
import time

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from . import auth

OLLAMA = os.environ.get("BOSWELL_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("BOSWELL_LLM_MODEL", "gemma4:e4b")
THINK = os.environ.get("BOSWELL_LLM_THINK", "none")
KEEP_ALIVE = os.environ.get("BOSWELL_LLM_KEEP_ALIVE", "10m")

# Loading a model takes seconds and a long answer with tools can take a minute;
# not being able to connect at all should fail at once, so the phone can fall back.
TIMEOUT = httpx.Timeout(300.0, connect=3.0)

# OpenRouter's own extensions; Ollama doesn't know them (web search can only be done there).
OPENROUTER_ONLY = ("usage", "reasoning", "plugins", "provider", "transforms", "models", "route", "web_search_options")

router = APIRouter()


def _note(msg: str):
    """A line in the terminal screen's log, when there is one."""
    try:
        from .api import note
        note(msg)
    except Exception:
        pass


def _key(authorization: str | None) -> dict:
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    phone = auth.check(token)
    if phone is None:
        raise HTTPException(401, "not paired")
    return phone


def _installed(name: str, tags: dict) -> bool:
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    return name in names or (":" not in name and f"{name}:latest" in names)


def request_for_ollama(body: dict) -> dict:
    """The phone's request as Ollama wants it: our model, no streaming, nothing OpenRouter-only."""
    out = {k: v for k, v in body.items() if k not in OPENROUTER_ONLY}
    out["model"] = MODEL
    out["stream"] = False
    # The phone asks OpenRouter for little thinking ({"reasoning": {"effort": "low"}}); a small
    # local model does better with none at all (see above), so this computer's setting decides.
    out["reasoning_effort"] = THINK
    return out


async def _keep_for(client: httpx.AsyncClient):
    """Ollama's OpenAI API ignores keep_alive, so set it with an empty native request (loads nothing new)."""
    try:
        await client.post(f"{OLLAMA}/api/generate", json={"model": MODEL, "keep_alive": KEEP_ALIVE}, timeout=10.0)
    except Exception:
        pass


@router.get("/v1/llm")
async def llm_status():
    """Whether the local model can answer: Ollama up and the model installed (and whether it's loaded now)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            tags = (await c.get(f"{OLLAMA}/api/tags")).json()
            ps = (await c.get(f"{OLLAMA}/api/ps")).json()
    except Exception:
        return {"available": False, "model": MODEL, "reason": "Ollama isn't running on this computer"}
    if not _installed(MODEL, tags):
        return {"available": False, "model": MODEL, "reason": f"{MODEL} isn't installed: run  ollama pull {MODEL}"}
    loaded = next((m for m in ps.get("models", []) if m.get("name") == MODEL or m.get("model") == MODEL), None)
    return {"available": True, "model": MODEL, "loaded": loaded is not None,
            "vram_gb": round(loaded.get("size_vram", 0) / 1e9, 1) if loaded else None,
            "think": THINK, "keep_alive": KEEP_ALIVE}


@router.post("/v1/chat/completions")
async def chat(request: Request, authorization: str | None = Header(None)):
    phone = _key(authorization)
    try:
        body = await request.json()
        assert isinstance(body, dict) and isinstance(body.get("messages"), list)
    except Exception:
        raise HTTPException(400, "expected an OpenAI-style chat request with messages")
    if body.get("stream"):
        raise HTTPException(400, "streaming isn't supported here")
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{OLLAMA}/v1/chat/completions", json=request_for_ollama(body))
            if r.status_code < 400:
                await _keep_for(c)
    except httpx.ConnectError:
        _note("AI: Ollama isn't running")
        raise HTTPException(503, "the AI on this computer (Ollama) isn't running")
    except httpx.TimeoutException:
        _note("AI: Ollama took too long")
        raise HTTPException(504, "the AI on this computer took too long to answer")
    if r.status_code == 404 or (r.status_code >= 400 and "not found" in r.text and "model" in r.text):
        _note(f"AI: {MODEL} isn't installed")
        raise HTTPException(503, f"{MODEL} isn't installed on this computer: run  ollama pull {MODEL}")
    if r.status_code >= 500:
        _note(f"AI: Ollama failed ({r.status_code})")
        raise HTTPException(503, f"the AI on this computer failed: {r.text[:300]}")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Ollama refused the request: {r.text[:300]}")
    try:
        out = r.json()
    except Exception:
        raise HTTPException(502, "Ollama sent something that isn't JSON")
    usage = out.setdefault("usage", {}) or {}
    usage["cost"] = 0.0                     # it's your own electricity
    out["usage"] = usage
    out["model"] = out.get("model") or MODEL
    calls = len(((out.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
    _note(f"AI: {phone.get('device', 'phone')} · {MODEL} · {usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tokens"
          f"{f' · {calls} tool call' + ('s' if calls != 1 else '') if calls else ''} · {time.time() - t0:.1f} s")
    return out
