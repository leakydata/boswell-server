"""The HTTP API the phone talks to.

  GET  /v1/health                  no key: is it up, what can it do
  POST /v1/pair    {code, device}  a pairing code -> this phone's key
  POST /v1/analyze ?voice_model=…  (key) one recording's audio -> its ingredients
  POST /v1/backup                  (key) a whole backup zip from the phone, streamed to disk (older phones)
  POST /v1/backup/start {files}    (key) an incremental backup: the phone's file list -> the hashes missing here
  POST /v1/backup/blobs            (key) those files, streamed (see backups.BlobReader)
  POST /v1/backup/finish {session} (key) the backup's manifest, written once every file is here
  GET  /v1/backups                 (key) this phone's backups: name, bytes, created; the newest 7 kept
  GET  /v1/backups/{name}          (key) one of them as a backup zip, to restore from
  GET  /v1/local/status            this computer only: what the terminal screen shows
"""
import collections
import re
import time

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from . import __version__, auth, backups as store, config
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
            "voice_models": list(VOICE_MODELS), "asr": "parakeet-tdt-0.6b-v3", "diarization": "pyannote-community-1",
            "backup": 2}   # 2: incremental backups (POST /v1/backup/start)


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


KEEP_BACKUPS = store.KEEP
BACKUP_NAME = store.ZIP_NAME


def backups_dir(phone: dict):
    """This phone's backup folder, named after it. Pairing again from the same phone keeps its
    name (and gives it a new key), so its name is what stays the same."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", phone["device"]).strip("-")[:80] or "phone"
    return config.DATA / "backups" / safe


@app.post("/v1/backup")
async def backup(request: Request, authorization: str | None = Header(None)):
    """The phone's whole backup (hundreds of MB), written to disk as it arrives: never all in memory."""
    phone = _key(authorization)
    d = backups_dir(phone)
    d.mkdir(parents=True, exist_ok=True)
    name = time.strftime("boswell-backup-%Y-%m-%d-%H%M%S.zip")
    part = d / (name + ".part")
    size = 0
    try:
        with open(part, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                size += len(chunk)
        if size == 0:
            raise HTTPException(400, "no backup")
        with open(part, "rb") as f:
            if f.read(4) != b"PK\x03\x04":
                raise HTTPException(400, "not a zip")
        part.replace(d / name)
    except BaseException as e:
        part.unlink(missing_ok=True)
        if not isinstance(e, HTTPException):
            note(f"backup from {phone['device']} failed: {e}")
        raise
    store.keep(d, KEEP_BACKUPS)
    note(f"backup from {phone['device']}: {size / 2**20:.0f} MB")
    return {"name": name, "bytes": size}


def _refused(e: store.Refused):
    return HTTPException(e.status, str(e))


@app.post("/v1/backup/start")
def backup_start(body: dict = Body(...), authorization: str | None = Header(None)):
    """An incremental backup begins: every file the phone would put in its zip, in that order
    (path, size, sha256). The answer is a session and the sha256s this server doesn't have."""
    phone = _key(authorization)
    d = backups_dir(phone)
    d.mkdir(parents=True, exist_ok=True)
    try:
        r = store.start(d, body.get("files"))
    except store.Refused as e:
        raise _refused(e)
    note(f"backup from {phone['device']}: {r['files']} files, {len(r['missing'])} new ({r['missing_bytes'] / 2**20:.0f} MB to send)")
    return r


@app.post("/v1/backup/blobs")
async def backup_blobs(request: Request, authorization: str | None = Header(None)):
    """Files for a backup, one after another (length, bytes, sha256), each checked and stored as it ends."""
    phone = _key(authorization)
    d = backups_dir(phone)
    reader = store.BlobReader(store.blobs_dir(d))
    store.blobs_dir(d).mkdir(parents=True, exist_ok=True)
    try:
        async for chunk in request.stream():
            reader.feed(chunk)
        reader.close()
    except store.Refused as e:
        reader.close_quietly()
        note(f"backup from {phone['device']}: {e}")
        raise _refused(e)
    except BaseException:
        reader.close_quietly()
        raise
    return {"stored": reader.stored, "received": reader.received, "bytes": reader.bytes}


@app.post("/v1/backup/finish")
def backup_finish(body: dict = Body(...), authorization: str | None = Header(None)):
    """The backup is all here: write its manifest, keep the newest 7, and let go of files none lists."""
    phone = _key(authorization)
    d = backups_dir(phone)
    try:
        r = store.finish(d, body.get("session"), body.get("changed"), body.get("drop"),
                         name=time.strftime("boswell-backup-%Y-%m-%d-%H%M%S.zip").removesuffix(".zip") + ".json")
    except store.Refused as e:
        note(f"backup from {phone['device']} refused: {e}")
        raise _refused(e)
    note(f"backup from {phone['device']}: {r['bytes'] / 2**20:.0f} MB, {r['files']} files")
    return r


@app.get("/v1/backups")
def backups(authorization: str | None = Header(None)):
    phone = _key(authorization)
    found = [{"name": n, "bytes": size, "created": created} for n, _, created, size in store.listing(backups_dir(phone))]
    note(f"{phone['device']} listed its backups ({len(found)})")
    return {"backups": found}


@app.get("/v1/backups/{name}")
def backup_file(name: str, authorization: str | None = Header(None)):
    phone = _key(authorization)
    # Only a name this server gave out, in this phone's own folder: nothing else can be reached.
    d = backups_dir(phone)
    found = {n: f for n, f, _, _ in store.listing(d)} if BACKUP_NAME.fullmatch(name) else {}
    f = found.get(name)
    if f is None:
        raise HTTPException(404, "no such backup")
    note(f"{phone['device']} downloaded {name}")
    if f.suffix == ".zip":
        return FileResponse(f, media_type="application/zip", filename=name)
    # Incremental: the zip is put together from its files as it's sent.
    m = store.manifest(f)
    return StreamingResponse(store.zip_stream(d, m), media_type="application/zip",
                             headers={"Content-Length": str(m["bytes"]),
                                      "Content-Disposition": f'attachment; filename="{name}"'})
