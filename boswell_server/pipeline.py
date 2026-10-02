"""One recording in, the phone's raw ingredients out.

The phone assembles transcripts itself (Lines.build, its vocabulary, matching
against its people), so the server returns exactly what the phone's own
models would have produced, from bigger ones:

  speech    seconds anyone speaks (pyannote); under SPEECH_MIN_S the rest is skipped
  words     [{text, start, end}] -- Parakeet TDT 0.6B v3, full precision on the GPU
  speakers  [{index, turns: [{start, end}], seconds, voiceprint}] -- pyannote 3.1 on
            the GPU; speakers numbered by first turn, as the phone's Diarizer does;
            voiceprints in the model the phone asked for (VOICE_MODELS), on the GPU
  sounds    [{label, score, at}] -- CED-Mini, windowed exactly as the phone's SoundTagger

Each step mirrors a phone function, named in its docstring.
"""
import os
import threading
import time

import numpy as np

from .config import MODELS, SR, VOICE_MODELS

SPEECH_MIN_S = 0.3          # Diarizer.SPEECH_MIN_S
MIN_VOICEPRINT_S = 0.8      # Diarizer.MIN_VOICEPRINT_S
# Sounds.kt
WINDOW_S, HOP_S, WINDOW_MIN, WINDOW_KEEP, WHOLE_KEEP, FLOOR, KEEP = 10.0, 5.0, 0.25, 0.35, 0.20, 0.02, 10


def words_from_tokens(tokens, timestamps, audio_seconds, max_word=1.0):
    """Words.fromTokens: a token starting with a space starts a word; a word ends at
    the next word's start, capped so one before a silence doesn't swallow it."""
    acc = []
    for i, tok in enumerate(tokens):
        t = float(timestamps[i]) if i < len(timestamps) else (float(timestamps[-1]) if len(timestamps) else 0.0)
        if not tok.strip():
            acc.append(["", t])
            continue
        if tok.startswith(" ") or not acc:
            acc.append([tok.lstrip(), t])
        elif acc[-1][0] == "":
            acc[-1] = [tok, acc[-1][1]]
        else:
            acc[-1][0] += tok
    words = [w for w in acc if w[0].strip()]
    out = []
    for i, (text, start) in enumerate(words):
        nxt = words[i + 1][1] if i + 1 < len(words) else audio_seconds
        out.append({"text": text, "start": round(start, 3), "end": round(max(start, min(nxt, start + max_word)), 3)})
    return out



def _speech_main(conn, provider, threads):
    """The sound-tagging helper process: CED through sherpa-onnx's CUDA build.

    Its own process because sherpa-onnx bundles an ONNX Runtime (1.28) that
    refuses to start once PyTorch or onnxruntime-gpu (1.30) is loaded in the
    same process -- and the main process needs both for pyannote and voiceprints.
    """
    import sherpa_onnx
    c = MODELS / "ced-mini"
    tagger = sherpa_onnx.AudioTagging(sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(ced=str(c / "model.onnx"), num_threads=2, provider=provider),
        labels=str(c / "class_labels_indices.csv"), top_k=20))
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        kind, audio = msg
        try:
            if kind == "tag":
                s = tagger.create_stream(); s.accept_waveform(SR, audio)
                conn.send(("ok", {e.name: float(e.prob) for e in tagger.compute(s, 20)}))
        except Exception as e:
            conn.send(("error", repr(e)))


def _nvidia_libs() -> str:
    """CUDA and cuDNN as PyTorch installed them (nvidia-* wheels), for sherpa-onnx's CUDA provider."""
    import glob, site
    dirs = []
    for sp in site.getsitepackages():
        dirs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    return ":".join(dirs)


class SpeechHelper:
    def __init__(self, provider, threads):
        import multiprocessing as mp
        # The helper is a fresh interpreter: its loader reads this when it starts.
        libs = _nvidia_libs()
        if libs and libs not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = libs + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_speech_main, args=(child, provider, threads), daemon=True)
        self.proc.start()
        kind, _ = self.conn.recv()
        if kind != "ready":
            raise RuntimeError("speech helper didn't start")

    def ask(self, kind, audio):
        self.conn.send((kind, np.ascontiguousarray(audio, dtype=np.float32)))
        status, value = self.conn.recv()
        if status != "ok":
            raise RuntimeError(value)
        return value

class Engine:
    """The models, loaded once. One recording at a time on the GPU."""

    def __init__(self, asr_threads: int = 8, provider: str = "cuda"):
        self.lock = threading.Lock()
        self.provider = provider   # sherpa-onnx's CUDA build runs Parakeet and CED on the GPU
        self.asr_threads = asr_threads
        self._diar = self._asr = self._helper = None
        self._vp = {}

    # ------------------------------------------------------------ models

    def diarizer(self):
        if self._diar is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from pyannote.audio import Pipeline
            p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            p.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._diar = p
        return self._diar

    def speech(self) -> SpeechHelper:
        """CED, in the helper process (started first, before torch loads here)."""
        if self._helper is None:
            self._helper = SpeechHelper(self.provider, 2)
        return self._helper

    def asr(self):
        """Parakeet TDT 0.6B v3, full precision on the GPU (onnx-asr): ~0.17 s per 30 s clip on a 4090."""
        if self._asr is None:
            import onnx_asr
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except Exception:
                pass
            self._asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MODELS / "parakeet-v3-fp32"),
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]).with_timestamps()
        return self._asr

    def voiceprinter(self, model_id: str):
        if model_id not in self._vp:
            import onnxruntime as ort
            try:
                ort.preload_dlls()
            except Exception:
                pass
            spec = VOICE_MODELS[model_id]
            self._vp[model_id] = ort.InferenceSession(str(MODELS / spec["file"]),
                                                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return self._vp[model_id]

    def warm(self, voice_models=("wespeaker-resnet34-lm",)):
        self.speech(); self.diarizer(); self.asr()
        for m in voice_models:
            self.voiceprinter(m)

    # ------------------------------------------------------------ steps

    def diarize(self, audio):
        """Turns per speaker, numbered by first turn (Diarizer.run's ordering)."""
        import torch
        out = self.diarizer()({"waveform": torch.from_numpy(audio)[None, :], "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)
        by = {}
        for seg, _, label in ann.itertracks(yield_label=True):
            by.setdefault(label, []).append((float(seg.start), float(seg.end)))
        ordered = sorted(by.values(), key=lambda turns: turns[0][0])
        return [sorted(t) for t in ordered]

    def transcribe(self, audio):
        r = self.asr().recognize(audio, sample_rate=SR)
        return words_from_tokens(list(r.tokens or []), list(r.timestamps or []), len(audio) / SR)

    def voiceprint(self, audio, model_id):
        spec = VOICE_MODELS[model_id]
        if len(audio) < spec["min_s"] * SR:
            return None
        if spec["max_s"]:
            audio = audio[: int(spec["max_s"] * SR)]
        s = self.voiceprinter(model_id)
        v = s.run(None, {spec["input"]: audio[None, :].astype(np.float32)})[0].reshape(-1)
        if not np.all(np.isfinite(v)):
            return None
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32).tolist() if n > 0 else None

    def _look(self, audio):
        return self.speech().ask("tag", audio[:160_000])

    def tag(self, audio):
        """SoundTagger.tag: the first 10 s whole, plus 10 s windows every 5 s."""
        if len(audio) < 3_200:
            return []
        whole = self._look(audio)
        n, step = int(WINDOW_S * SR), int(HOP_S * SR)
        windows, i = [], 0
        while i == 0 or i + n <= len(audio):
            seg = audio[i:min(i + n, len(audio))]
            if len(seg) < SR:
                break
            windows.append((i / SR, self._look(seg)))
            i += step
            if i + SR > len(audio):
                break
        labels = set(whole) | {k for _, w in windows for k in w}
        out = []
        for label in labels:
            hits = [(t, w[label]) for t, w in windows if w.get(label, 0) >= WINDOW_MIN]
            w = whole.get(label, 0.0)
            if not (sum(1 for _, s in hits if s >= WINDOW_KEEP) >= 2 or w >= WHOLE_KEEP):
                continue
            best = max(hits, key=lambda h: h[1]) if hits else None
            at, score = best if best and best[1] > w else (0.0, w)
            out.append({"label": label, "score": round(score, 4), "at": at})
        return sorted([o for o in out if o["score"] >= FLOOR], key=lambda o: -o["score"])[:KEEP]

    # ------------------------------------------------------------ the whole thing

    def analyze(self, audio: np.ndarray, voice_model: str = "wespeaker-resnet34-lm") -> dict:
        if voice_model not in VOICE_MODELS:
            raise ValueError(f"unknown voice model {voice_model}")
        t0 = time.time()
        with self.lock:
            speakers = self.diarize(audio)
            speech = sum(e - s for turns in speakers for s, e in turns)
            sounds = self.tag(audio)
            if speech <= SPEECH_MIN_S:
                return {"speech": round(speech, 3), "words": [], "speakers": [], "sounds": sounds,
                        "engine": "pyannote-3.1 (home) + ced-mini", "ms": int((time.time() - t0) * 1000)}
            words = self.transcribe(audio)
            out = []
            for i, turns in enumerate(speakers):
                secs = sum(e - s for s, e in turns)
                vp = None
                if secs >= MIN_VOICEPRINT_S:
                    clip = np.concatenate([audio[int(s * SR):int(e * SR)] for s, e in turns])
                    vp = self.voiceprint(clip, voice_model)
                out.append({"index": i, "turns": [{"start": round(s, 3), "end": round(e, 3)} for s, e in turns],
                            "seconds": round(secs, 3), "voiceprint": vp})
        return {"speech": round(speech, 3), "words": words, "speakers": out, "sounds": sounds,
                "engine": f"parakeet-tdt-0.6b-v3 (home) + pyannote-3.1 (home) + {voice_model} + ced-mini",
                "voice_model": voice_model, "ms": int((time.time() - t0) * 1000)}
