"""Which diarization model separates speakers best, on Omi audio with a known answer?

Desktop Boswell's archive has no conversations with checked speaker times, so this
builds them: speech of people the owner named (each voiceprint points at a clip and a
diarizer label, so the exact speech can be cut out), stitched into 30 s exchanges of
2-3 people with short gaps and occasional overlaps. The answer is then exact, in the
Omi's real recording conditions.

Caveat: stitched speech comes from different recordings (rooms, distances), which makes
speakers somewhat easier to tell apart than in one real conversation. It ranks models
fairly, but flatters all of them.

    uv run python tools/diar_bench.py [--n 80] [--models pyannote-3.1,community-1]
"""
import argparse
import json
import os
import random
import sqlite3
import time

import numpy as np
import soundfile as sf

DATA = os.path.expanduser("~/Documents/electronics/nRF52840/data")
SR = 16000


def speech_by_person(min_seg=1.5, max_per_person=60):
    """person -> list of speech pieces (float32), each one diarized turn of theirs."""
    db = sqlite3.connect(os.path.join(DATA, "speakers.db"))
    rows = db.execute("""SELECT p.id, v.clip, v.speaker FROM voiceprints v JOIN people p ON p.id = v.person_id
                         WHERE p.name IS NOT NULL AND (p.kind IS NULL OR p.kind != 'media') AND v.clip IS NOT NULL""").fetchall()
    out = {}
    for pid, clip, spk in rows:
        if len(out.get(pid, [])) >= max_per_person:
            continue
        base = os.path.splitext(clip)[0]
        wav, tr = os.path.join(DATA, base + ".wav"), os.path.join(DATA, "transcripts", base + ".json")
        if not (os.path.exists(wav) and os.path.exists(tr)):
            continue
        t = json.load(open(tr))
        segs = [(s["start"], s["end"]) for s in t.get("segments", []) if s.get("speaker") == spk and s["end"] - s["start"] >= min_seg]
        if not segs:
            continue
        a, sr = sf.read(wav, dtype="float32")
        if sr != SR:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        for s, e in segs:
            out.setdefault(pid, []).append(a[int(s * SR):int(min(e, s + 6) * SR)])
    return {p: v for p, v in out.items() if len(v) >= 5}


def make_conversation(people_speech, rnd, seconds=30.0):
    """A stitched exchange: (audio, [(start, end, speaker)])."""
    n = rnd.choice([2, 2, 3])
    who = rnd.sample(list(people_speech), n)
    audio = np.zeros(int(seconds * SR), np.float32)
    ref = []
    t = rnd.uniform(0.2, 1.0)
    last = None
    while t < seconds - 1.5:
        p = rnd.choice([w for w in who if w != last] or who)
        piece = rnd.choice(people_speech[p])
        piece = piece[: int((seconds - t) * SR)]
        if len(piece) < SR:
            break
        i = int(t * SR)
        audio[i:i + len(piece)] += piece
        ref.append((t, t + len(piece) / SR, str(p)))
        last = p
        # Usually a short gap; sometimes the next one starts before this ends.
        t += len(piece) / SR + (rnd.uniform(-0.6, -0.2) if rnd.random() < 0.15 else rnd.uniform(0.15, 1.0))
    return np.clip(audio, -1, 1), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", default="pyannote-3.1,community-1")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    people = speech_by_person()
    print(f"{len(people)} people with enough speech; {sum(len(v) for v in people.values())} pieces")
    rnd = random.Random(a.seed)
    convs = [make_conversation(people, rnd) for _ in range(a.n)]
    names = {"pyannote-3.1": "pyannote/speaker-diarization-3.1", "community-1": "pyannote/speaker-diarization-community-1"}
    for key in a.models.split(","):
        p = Pipeline.from_pretrained(names[key]).to(torch.device("cuda"))
        der = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        right_count, ms = 0, []
        for audio, ref in convs:
            r = Annotation()
            for s, e, spk in ref:
                r[Segment(s, e)] = spk
            t = time.time()
            out = p({"waveform": torch.from_numpy(audio)[None, :], "sample_rate": SR})
            ms.append((time.time() - t) * 1000)
            hyp = getattr(out, "speaker_diarization", out)
            der(r, hyp)
            right_count += len(hyp.labels()) == len(set(x[2] for x in ref))
        d = abs(der)
        comp = der[:]
        print(f"{key:14s} DER {d * 100:5.1f}%  (missed {comp['missed detection'] / comp['total'] * 100:4.1f}%, "
              f"false alarm {comp['false alarm'] / comp['total'] * 100:4.1f}%, confusion {comp['confusion'] / comp['total'] * 100:4.1f}%)  "
              f"right number of speakers {right_count}/{len(convs)}  {np.median(ms):.0f} ms per 30 s", flush=True)


if __name__ == "__main__":
    main()
