"""boswell-server [screen | serve | pair | install-service | uninstall-service | fetch-models | doctor]"""
import argparse
import json
import sys
import tarfile
import time
import urllib.request

from . import __version__
from .config import ARCHIVES, FILES, MODELS, PORT, base_url, tailscale_name


def _download(url: str, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {url.rsplit('/', 1)[-1]} …", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def fetch_models():
    """Everything the server runs, into ~/.local/share/boswell-server/models."""
    for name, url in FILES.items():
        if not (MODELS / name).exists():
            _download(url, MODELS / name)
    for folder, url in ARCHIVES.items():
        if (MODELS / folder).exists():
            continue
        arc = MODELS / url.rsplit("/", 1)[-1]
        _download(url, arc)
        with tarfile.open(arc) as t:
            top = t.getnames()[0].split("/")[0]
            t.extractall(MODELS, filter="data")
        (MODELS / top).rename(MODELS / folder)
        arc.unlink()
    from huggingface_hub import snapshot_download
    print("  nvidia/parakeet-tdt-0.6b-v3 (NeMo) …", flush=True)
    snapshot_download("nvidia/parakeet-tdt-0.6b-v3")
    print("pyannote/speaker-diarization-community-1 comes from Hugging Face's cache; if it isn't there, accept its terms on "
          "huggingface.co and run `huggingface-cli login` once.")
    print("done")


def doctor():
    """What's in place, what isn't."""
    import torch
    print(f"Boswell Server {__version__}")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (everything will run on the CPU)")
    print("Tailscale:", tailscale_name() or "off -- the phone can only reach this computer on the same network")
    print("Address:", base_url())
    from huggingface_hub import try_to_load_from_cache
    nemo = try_to_load_from_cache("nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo")
    print(f"  {'ok     ' if isinstance(nemo, str) else 'MISSING'} nvidia/parakeet-tdt-0.6b-v3 (Hugging Face cache)")
    for p in ["voiceprint.onnx", "speaker-id-redimnet2-b6.onnx", "ced-mini/model.onnx"]:
        print(f"  {'ok     ' if (MODELS / p).exists() else 'MISSING'} {p}")
    from . import service
    from .local import running
    print(f"Service: {service.state()}")
    print(f"Port {PORT}:", {"boswell": "Boswell Server is running", "other": "in use by another program",
                            None: "free (no server running)"}[running()])


def pair():
    """Without the screen: print a code and its QR code here."""
    from . import auth
    from .tui import qr_text
    code = auth.new_code()
    print(qr_text(json.dumps({"boswell": 1, "server": base_url(), "code": code})))
    print(f"\n{base_url()}   code {code}   (once, for 10 minutes; the server must be running)")


def main():
    ap = argparse.ArgumentParser(prog="boswell-server", description="Home processing server for Boswell Phone")
    ap.add_argument("command", nargs="?", default="screen",
                    choices=["screen", "serve", "pair", "install-service", "uninstall-service", "fetch-models", "doctor"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--unit", default="boswell-server", help=argparse.SUPPRESS)   # another unit name, for testing
    a = ap.parse_args()
    if a.command in ("serve", "screen"):
        from .local import running
        busy = running()
        if busy == "other":
            sys.exit(f"Port {PORT} is in use by another program; set BOSWELL_PORT to use a different one.")
        if busy == "boswell" and a.command == "serve":
            sys.exit(f"Boswell Server is already running on port {PORT}. `boswell-server` shows it.")
    if a.command == "install-service":
        from . import service
        service.install(a.host, a.unit)
    elif a.command == "uninstall-service":
        from . import service
        service.uninstall(a.unit)
    elif a.command == "fetch-models":
        fetch_models()
    elif a.command == "doctor":
        doctor()
    elif a.command == "pair":
        pair()
    elif a.command == "serve":
        import logging
        import threading
        import uvicorn
        from . import api
        api.echo = True
        threading.Thread(target=api.warm, daemon=True).start()
        print(f"Boswell Server on {base_url()}", flush=True)
        config = uvicorn.Config(api.app, host=a.host, port=PORT, log_level="info")
        # An open screen asks for the status every two seconds; that's not news for the log.
        logging.getLogger("uvicorn.access").addFilter(lambda r: "/v1/local/status" not in r.getMessage())
        uvicorn.Server(config).run()
    elif busy == "boswell":
        # The service (or another terminal) is serving: show it, don't load a second copy of everything.
        from .tui import ServerApp
        ServerApp(a.host, attach=True).run()
        print(f"Boswell Server is still running on port {PORT} (systemctl --user status boswell-server, if it's the service).")
    else:
        from . import api
        from .tui import ServerApp
        # The tagging helper is a separate process; start it before the screen takes over
        # this terminal's file descriptors (spawning afterwards fails: "bad value(s) in fds_to_keep").
        print("starting…", flush=True)
        api.engine.speech()
        ServerApp(a.host).run()


if __name__ == "__main__":
    sys.exit(main())
