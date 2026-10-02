"""Where Boswell Server keeps things, and the models it runs."""
import json
import os
import shutil
import subprocess
from pathlib import Path

PORT = int(os.environ.get("BOSWELL_PORT", "8765"))
DATA = Path(os.environ.get("BOSWELL_SERVER_DATA", Path.home() / ".local" / "share" / "boswell-server"))
MODELS = DATA / "models"
TOKENS = DATA / "phones.json"
SR = 16_000

# The phone's own models (from its models-v1 release) give voiceprints the phone can
# match exactly; Parakeet and pyannote are the server's.
PHONE_RELEASE = "https://github.com/leakydata/boswell-phone/releases/download/models-v1/"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
# Single files from the phone's release, and archives from sherpa-onnx's.
FILES = {
    "voiceprint.onnx": PHONE_RELEASE + "voiceprint.onnx",
    "speaker-id-redimnet2-b6.onnx": PHONE_RELEASE + "speaker-id-redimnet2-b6.onnx",
}
ARCHIVES = {   # folder name -> archive (unpacked into MODELS/<folder>)
    "parakeet-v3": SHERPA + "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
    "ced-mini": SHERPA + "audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2",
}

# Voiceprint models the phone may ask for, by the id it files them under (VoiceModel.kt).
VOICE_MODELS = {
    "wespeaker-resnet34-lm": {"file": "voiceprint.onnx", "input": "audio", "max_s": None, "min_s": 0.025},
    "redimnet2-b6-vb2vox2-lm": {"file": "speaker-id-redimnet2-b6.onnx", "input": "waveform", "max_s": 8.0, "min_s": 0.5},
}


def tailscale_name() -> str | None:
    """This computer's address on the tailnet (MagicDNS), if Tailscale is up."""
    if not shutil.which("tailscale"):
        return None
    try:
        out = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5).stdout
        name = json.loads(out)["Self"]["DNSName"].rstrip(".")
        return name or None
    except Exception:
        return None


def base_url() -> str:
    return f"http://{tailscale_name() or 'localhost'}:{PORT}"
