"""Pairing: a short code shown on this computer, traded once by a phone for its own key.

Keys are stored only as SHA-256 hashes; a phone's key never leaves the phone
again. A code lasts ten minutes and works once.
"""
import hashlib
import json
import secrets
import threading
import time

from .config import TOKENS

CODE_SECONDS = 600
_lock = threading.Lock()
CODES = TOKENS.with_name("pairing-codes.json")   # a file, so `boswell-server pair` works beside a running server


def _codes() -> dict[str, float]:
    try:
        return {c: t for c, t in json.loads(CODES.read_text()).items() if t > time.time()}
    except Exception:
        return {}


def _save_codes(codes: dict[str, float]):
    CODES.parent.mkdir(parents=True, exist_ok=True)
    tmp = CODES.with_suffix(".tmp")
    tmp.write_text(json.dumps(codes))
    tmp.chmod(0o600)
    tmp.replace(CODES)


def _load() -> dict:
    try:
        return json.loads(TOKENS.read_text())
    except Exception:
        return {"phones": []}


def _save(d: dict):
    TOKENS.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKENS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.chmod(0o600)
    tmp.replace(TOKENS)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_code() -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    with _lock:
        codes = _codes()
        codes[code] = time.time() + CODE_SECONDS
        _save_codes(codes)
    return code


def pair(code: str, device: str) -> str | None:
    """A valid, unused code -> a new key for this phone; otherwise None."""
    with _lock:
        codes = _codes()
        ok = codes.pop(code, 0) > time.time()
        _save_codes(codes)
        if not ok:
            return None
        token = secrets.token_urlsafe(32)
        d = _load()
        d["phones"].append({"device": device[:80] or "phone", "hash": _hash(token), "paired": time.time(), "last": None})
        _save(d)
    return token


def check(token: str | None) -> dict | None:
    if not token:
        return None
    h = _hash(token)
    with _lock:
        d = _load()
        for p in d["phones"]:
            if secrets.compare_digest(p["hash"], h):
                p["last"] = time.time()
                _save(d)
                return p
    return None


def phones() -> list[dict]:
    return _load()["phones"]


def forget(device: str):
    with _lock:
        d = _load()
        d["phones"] = [p for p in d["phones"] if p["device"] != device]
        _save(d)
