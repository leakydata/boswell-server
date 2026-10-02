"""Speech detection: does a recording have speech worth transcribing?

Reference: desktop Boswell's transcripts. A recording where its transcriber found at
least 5 words counts as speech; one where it found nothing counts as no speech
(recordings in between are left out as ambiguous). Each detector says how many seconds
of speech it hears; the question is how well that separates the two groups, and what
it costs.

    uv run --with silero-vad python tools/vad_bench.py [--n 400]
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import soundfile as sf

DATA = os.path.expanduser("~/Documents/electronics/nRF52840/data")
SR = 16000


def pick(n, seed):
    speech, empty = [], []
    files = sorted(glob.glob(os.path.join(DATA, "transcripts", "omi_*.json")))
    random.Random(seed).shuffle(files)
    for f in files:
        try:
            t = json.load(open(f))
        except Exception:
            continue
        words = sum(len(s.get("text", "").split()) for s in t.get("segments", []))
        wav = os.path.join(DATA, os.path.basename(f)[:-5] + ".wav")
        if not os.path.exists(wav):
            continue
        if words >= 5 and len(speech) < n // 2:
            speech.append(wav)
        elif words == 0 and len(empty) < n // 2:
            empty.append(wav)
        if len(speech) >= n // 2 and len(empty) >= n // 2:
            break
    return speech, empty


def best_split(pos, neg):
    """The threshold (seconds of speech) that gets the most recordings right, and how many."""
    best = (0, 0.0)
    for t in sorted(set(pos + neg)):
        right = sum(p > t for p in pos) + sum(n <= t for n in neg)
        best = max(best, (right, t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    speech, empty = pick(a.n, 5)
    print(f"{len(speech)} recordings with speech, {len(empty)} without")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    seg = Model.from_pretrained("pyannote/segmentation-3.0").to(torch.device("cuda"))
    vad = VoiceActivityDetection(segmentation=seg)
    vad.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    def pyannote_secs(x):
        return vad({"waveform": torch.from_numpy(x)[None, :], "sample_rate": SR}).get_timeline().support().duration()

    from silero_vad import get_speech_timestamps, load_silero_vad
    sil = load_silero_vad()

    def silero_secs(x):
        return sum(s["end"] - s["start"] for s in get_speech_timestamps(torch.from_numpy(x), sil, sampling_rate=SR)) / SR

    for name, f in (("pyannote segmentation-3.0", pyannote_secs), ("Silero VAD v5", silero_secs)):
        t = time.time()
        pos = [f(sf.read(w, dtype="float32")[0]) for w in speech]
        neg = [f(sf.read(w, dtype="float32")[0]) for w in empty]
        ms = (time.time() - t) / (len(pos) + len(neg)) * 1000
        right, thr = best_split(pos, neg)
        at_03 = sum(p > 0.3 for p in pos) + sum(n <= 0.3 for n in neg)
        print(f"{name:26s} best: {right}/{len(pos) + len(neg)} right at > {thr:.2f} s of speech;  at 0.3 s: {at_03} right "
              f"(misses {sum(p <= 0.3 for p in pos)} with speech, keeps {sum(n > 0.3 for n in neg)} empty);  {ms:.0f} ms per recording", flush=True)


if __name__ == "__main__":
    main()
