"""The HTTP API the phone talks to.

  GET  /v1/health                  no key: is it up, what can it do
  POST /v1/pair    {code, device}  a pairing code -> this phone's key
  POST /v1/analyze ?voice_model=…  (key) one recording's audio -> its ingredients
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


def note(msg: str):
    log.append((time.time(), msg))


app = FastAPI(title="Boswell Server", version=__version__)


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
                  authorization: str | None = Header(None)):
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
    result = await run_in_threadpool(engine.analyze, audio, voice_model)
    secs = len(audio) / 16_000
    recent.append({"at": time.time(), "phone": phone["device"], "clip": clip, "seconds": round(secs, 1),
                   "ms": int((time.time() - t0) * 1000), "words": len(result["words"]), "speakers": len(result["speakers"])})
    return result
