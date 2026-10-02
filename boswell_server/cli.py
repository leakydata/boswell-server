"""boswell-server [screen | serve | pair | fetch-models | doctor]"""
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
    if not (MODELS / "parakeet-v3-fp32" / "encoder-model.onnx").exists():
        from huggingface_hub import snapshot_download
        print("  parakeet-tdt-0.6b-v3 (full precision) …", flush=True)
        snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx", local_dir=MODELS / "parakeet-v3-fp32")
    print("pyannote/speaker-diarization-3.1 comes from Hugging Face's cache; if it isn't there, accept its terms on "
          "huggingface.co and run `huggingface-cli login` once.")
    print("done")


def doctor():
    """What's in place, what isn't."""
    import torch
    print(f"Boswell Server {__version__}")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (everything will run on the CPU)")
    print("Tailscale:", tailscale_name() or "off -- the phone can only reach this computer on the same network")
    print("Address:", base_url())
    for p in ["voiceprint.onnx", "speaker-id-redimnet2-b6.onnx", "parakeet-v3-fp32/encoder-model.onnx", "ced-mini/model.onnx"]:
        print(f"  {'ok     ' if (MODELS / p).exists() else 'MISSING'} {p}")


def pair():
    """Without the screen: print a code and its QR code here."""
    from . import auth
    from .tui import qr_text
    code = auth.new_code()
    print(qr_text(json.dumps({"boswell": 1, "server": base_url(), "code": code})))
    print(f"\n{base_url()}   code {code}   (once, for 10 minutes; the server must be running)")


def main():
    ap = argparse.ArgumentParser(prog="boswell-server", description="Home processing server for Boswell Phone")
    ap.add_argument("command", nargs="?", default="screen", choices=["screen", "serve", "pair", "fetch-models", "doctor"])
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    if a.command == "fetch-models":
        fetch_models()
    elif a.command == "doctor":
        doctor()
    elif a.command == "pair":
        pair()
    elif a.command == "serve":
        import threading
        import uvicorn
        from . import api
        threading.Thread(target=api.engine.warm, args=(("wespeaker-resnet34-lm", "redimnet2-b6-vb2vox2-lm"),), daemon=True).start()
        print(f"Boswell Server on {base_url()}", flush=True)
        uvicorn.run(api.app, host=a.host, port=PORT, log_level="info")
    else:
        from .tui import ServerApp
        ServerApp(a.host).run()


if __name__ == "__main__":
    sys.exit(main())
