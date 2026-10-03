"""A Boswell Server already running on this computer (the service, or another terminal): is there one, and what's it doing."""
import json
import socket
import urllib.error
import urllib.request

from .config import PORT


def _get(path: str, port: int, timeout: float) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def running(port: int = PORT) -> str | None:
    """"boswell" if Boswell Server answers on this port, "other" if something else holds it, None if it's free."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except OSError:
        return None
    try:
        return "boswell" if _get("/v1/health", port, 10).get("name") == "Boswell Server" else "other"
    except Exception:
        return "other"


def status(port: int = PORT) -> dict | None:
    """What the running server's screen would show (GET /v1/local/status); None if it can't say.
    A server older than this endpoint answers 404: {"old": True}."""
    try:
        return _get("/v1/local/status", port, 5)
    except urllib.error.HTTPError as e:
        return {"old": True} if e.code == 404 else None
    except Exception:
        return None
