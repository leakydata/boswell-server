"""Which speech recognizer is most accurate on the owner's own recordings?

The answer key is the owner's corrections: Boswell Phone marks a line fixed by hand
(`edited`), and a recording with one is taken as right as it now reads. Give this a
Boswell backup (Device -> Storage -> Back up); it scores every engine's word error rate
against those recordings.

    uv run --with faster-whisper python tools/asr_bench.py boswell-backup-….zip
    uv run --with faster-whisper python tools/asr_bench.py --speed        # timing only, on desktop audio
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boswell_server.audio import decode   # noqa: E402

MODELS = os.path.expanduser("~/.local/share/boswell-server/models")
SR = 16000


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split()


def edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


def keys_from_backup(path):
    """(name, audio, reference) for each recording with a hand-corrected line."""
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    out = []
    for n in names:
        if not (n.startswith("files/transcripts/") and n.endswith(".json")):
            continue
        t = json.loads(z.read(n))
        segs = t.get("segments", [])
        if not any(s.get("edited") for s in segs):
            continue
        base = os.path.basename(n)[:-5]
        for ext in (".ogg", ".wav"):
            a = f"files/clips/{base}{ext}"
            if a in names:
                out.append((base, decode(z.read(a)), " ".join(s["text"] for s in segs)))
                break
    return out


def engines():
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    import onnx_asr
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    out = {}
    out["Parakeet TDT 0.6B v3 (now)"] = (lambda m: lambda a: m.recognize(a, sample_rate=SR))(
        onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", os.path.join(MODELS, "parakeet-v3-fp32"), providers=gpu))
    from huggingface_hub import snapshot_download

    def local(repo):
        # A plain folder: onnxruntime refuses external weights behind the hub cache's symlinks.
        return snapshot_download(repo, local_dir=os.path.join(MODELS, "eval", repo.split("/")[-1]))
    out["Parakeet TDT 0.6B v2 (English)"] = (lambda m: lambda a: m.recognize(a, sample_rate=SR))(
        onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", local("istupakov/parakeet-tdt-0.6b-v2-onnx"), providers=gpu))
    out["Canary 1B v2"] = (lambda m: lambda a: m.recognize(a, sample_rate=SR, language="en"))(
        onnx_asr.load_model("nemo-canary-1b-v2", local("istupakov/canary-1b-v2-onnx"), providers=gpu))
    from faster_whisper import WhisperModel
    w = WhisperModel("large-v3", device="cuda", compute_type="float16")
    out["Whisper large-v3"] = lambda a: " ".join(s.text for s in w.transcribe(a, language="en", vad_filter=False)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backup", nargs="?")
    ap.add_argument("--speed", action="store_true", help="time each engine on desktop recordings, no scoring")
    a = ap.parse_args()
    if a.speed:
        import soundfile as sf
        files = sorted(glob.glob(os.path.expanduser("~/Documents/electronics/nRF52840/data/omi_*.wav")))[-200::20]
        keys = [(os.path.basename(f), sf.read(f, dtype="float32")[0], None) for f in files]
    else:
        keys = keys_from_backup(a.backup)
        print(f"{len(keys)} corrected recordings, {sum(len(words(k[2])) for k in keys)} words")
        if not keys:
            return
    for name, run in engines().items():
        run(keys[0][1])   # warm up
        e = n = 0
        t = time.time()
        for _, audio, ref in keys:
            hyp = run(audio)
            hyp = hyp if isinstance(hyp, str) else getattr(hyp, "text", str(hyp))
            if ref is not None:
                r = words(ref); e += edits(r, words(hyp)); n += len(r)
        ms = (time.time() - t) / len(keys) * 1000
        print(f"{name:32s} " + (f"{e / n * 100:5.1f}% words wrong  " if n else "") + f"{ms:.0f} ms per recording", flush=True)


if __name__ == "__main__":
    main()
