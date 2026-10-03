"""The HTTP API the phone talks to.

  GET  /v1/health                  no key: is it up, what can it do
  POST /v1/pair    {code, device}  a pairing code -> this phone's key
  POST /v1/analyze ?voice_model=…  (key) one recording's audio -> its ingredients
  GET  /v1/local/status            this computer only: what the terminal screen shows
"""
import collections
import time

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from . import __version__, auth
from .audio import decode
from .config import VOICE_MODELS
from .pipeline import Engine

engine = Engine()
# What the terminal screen shows: the latest jobs, and anything worth a line in its log.
recent: collections.deque = collections.deque(maxlen=200)
log: collections.deque = collections.deque(maxlen=500)
state = {"ready": "loading models…"}
echo = False   # headless (`serve`): log lines go to stdout too, so the journal has them


def note(msg: str):
    log.append((time.time(), msg))
    if echo:
        print(msg, flush=True)


def warm():
    """Load every model, and say how it went."""
    t = time.time()
    try:
        engine.warm(("wespeaker-resnet34-lm", "redimnet2-b6-vb2vox2-lm"))
        state["ready"] = f"ready (models loaded in {time.time() - t:.0f} s)"
    except Exception as e:
        state["ready"] = f"[red]models failed: {e}[/red]"
    note(state["ready"])


def gpu() -> str:
    """The biggest NVIDIA GPU: name, how busy, memory used."""
    try:
        import pynvml
        pynvml.nvmlInit()
        best = None
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            if best is None or mem.total > best[1].total:
                best = (h, mem)
        h, mem = best
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        name = pynvml.nvmlDeviceGetName(h)
        return f"{name} · {util}% busy · {mem.used / 2**30:.1f} of {mem.total / 2**30:.0f} GB"
    except Exception:
        return "no NVIDIA GPU found"


app = FastAPI(title="Boswell Server", version=__version__)
from .llm import router as llm_router  # noqa: E402  (AI on this computer: Ollama behind /v1/chat/completions)
app.include_router(llm_router)


def _key(authorization: str | None) -> dict:
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    phone = auth.check(token)
    if phone is None:
        raise HTTPException(401, "not paired")
    return phone


@app.get("/v1/health")
def health():
    import torch
    return {"name": "Boswell Server", "version": __version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "voice_models": list(VOICE_MODELS), "asr": "parakeet-tdt-0.6b-v3", "diarization": "pyannote-community-1"}


LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
PROXY_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-real-ip", "via")


def from_this_computer(request: Request) -> bool:
    """Asked directly on this computer: a loopback client, addressed to localhost, and not through a
    proxy. A proxy here (`tailscale serve`) connects from 127.0.0.1 too, but adds forwarding headers
    (and Tailscale-User-*) and keeps the outside Host."""
    if request.client is None or request.client.host not in LOOPBACK:
        return False
    host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        return False
    return not any(k.lower() in PROXY_HEADERS or k.lower().startswith("tailscale-") for k in request.headers.keys())


@app.get("/v1/local/status")
def local_status(request: Request):
    """For a terminal screen on this computer showing a server already running (the service).
    Recordings' names and phones' names are private, so nothing proxied or remote gets them."""
    if not from_this_computer(request):
        raise HTTPException(403, "only from this computer")
    return {"name": "Boswell Server", "version": __version__, "ready": state["ready"], "gpu": gpu(),
            "recent": list(recent), "log": list(log)}


@app.post("/v1/pair")
def pair(body: dict = Body(...)):
    token = auth.pair(str(body.get("code", "")), str(body.get("device", "phone")))
    if token is None:
        note("pairing refused: wrong or expired code")
        raise HTTPException(403, "wrong or expired code")
    note(f"paired {body.get('device', 'a phone')}")
    return {"token": token, "server": "Boswell Server", "version": __version__}


@app.post("/v1/analyze")
async def analyze(request: Request, voice_model: str = "wespeaker-resnet34-lm", clip: str = "",
                  authorization: str | None = Header(None), x_boswell_hotwords: str | None = Header(None)):
    phone = _key(authorization)
    if voice_model not in VOICE_MODELS:
        raise HTTPException(400, f"unknown voice model {voice_model}")
    data = await request.body()
    if not data:
        raise HTTPException(400, "no audio")
    t0 = time.time()
    try:
        audio = await run_in_threadpool(decode, data)
    except Exception as e:
        raise HTTPException(400, f"couldn't read the audio: {e}")
    # The phone's "words Boswell should know" (names, its owner's terms), boosted while decoding.
    try:
        import json
        hotwords = tuple(str(w) for w in json.loads(x_boswell_hotwords)) if x_boswell_hotwords else ()
    except Exception:
        hotwords = ()
    result = await run_in_threadpool(engine.analyze, audio, voice_model, hotwords)
    secs = len(audio) / 16_000
    recent.append({"at": time.time(), "phone": phone["device"], "clip": clip, "seconds": round(secs, 1),
                   "ms": int((time.time() - t0) * 1000), "words": len(result["words"]), "speakers": len(result["speakers"])})
    return result
