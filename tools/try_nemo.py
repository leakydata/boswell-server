"""NeMo Parakeet v3 in the server's environment: word timings, boosting, speed."""
import copy, glob, json, os, sys, time
import numpy as np, soundfile as sf

if __name__ == "__main__":
    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").cuda().eval()
    files = [f for f in sorted(glob.glob(os.path.expanduser("~/Documents/electronics/nRF52840/data/omi_*.wav")))[-400:]]
    pick = []
    for f in reversed(files):
        t = os.path.join(os.path.dirname(f), "transcripts", os.path.basename(f)[:-4] + ".json")
        if os.path.exists(t):
            words = sum(len(x.get("text", "").split()) for x in json.load(open(t)).get("segments", []))
            if words >= 40:
                pick.append(f)
        if len(pick) == 5:
            break
    audio = [sf.read(f, dtype="float32")[0] for f in pick]

    def run(label):
        m.transcribe([audio[0]], timestamps=True, verbose=False)
        t = time.time()
        out = m.transcribe(audio, timestamps=True, verbose=False, batch_size=1)
        ms = (time.time() - t) / len(audio) * 1000
        h = out[0]
        with open("/tmp/claude-1000/-home-scholyx-Documents-electronics-boswell-phone/e83dba6d-3f28-4c34-8bb0-cf6d25f75e8e/scratchpad/nemo.txt", "a") as log:
            log.write(f"{label}: {ms:.0f} ms each; timestamp keys {list((h.timestamp or {}).keys())}; word[:3] {(h.timestamp or {}).get('word', [])[:3]}\n")
            for o in out:
                log.write(f"    TEXT {o.text[:120]}\n")

    run("plain")
    cfg = copy.deepcopy(m.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.greedy.boosting_tree = {"key_phrases_list": ["Lindsey", "Boswell", "Omi", "Tacky", "Nexium", "OpenRouter", "open router"],
                                    "context_score": 1.0, "depth_scaling": 2.0}
        cfg.greedy.boosting_tree_alpha = 0.5
    t = time.time(); m.change_decoding_strategy(cfg, verbose=False); print(f"boosting set up in {(time.time() - t) * 1000:.0f} ms")
    run("boosted")
    import os as _o; _o.environ["HF_HUB_OFFLINE"] = "1"
    from pyannote.audio import Pipeline
    p = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1").to(torch.device("cuda"))
    print("audio seconds:", [round(len(a) / 16000, 1) for a in audio], "rms", [round(float((a ** 2).mean() ** 0.5), 4) for a in audio])
    print("pyannote alongside:", len(p({"waveform": torch.from_numpy(audio[0])[None, :], "sample_rate": 16000}).speaker_diarization.labels()), "speakers")
