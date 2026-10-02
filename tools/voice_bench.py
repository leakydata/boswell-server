"""Voiceprint models on desktop Boswell's named speakers: the phone's two, and WeSpeaker's
w2v-BERT 2.0 (Apache-2.0, ~590M parameters, GPU).

The same test as boswell_phone/tools/calibrate_voice.py: hand-named voices and a sample
of automatically matched ones, at most 8 s of each speaker's speech, scored by EER and by
picking the right person (best voiceprint per person, as Boswell matches).

    uv run --with transformers --with peft --with silero-vad python tools/voice_bench.py   (ONLY=w2v to run one)
"""
import os
import sys
import time

import numpy as np

PHONE_TOOLS = os.path.expanduser("~/Documents/electronics/boswell_phone/tools")
WESPEAKER = os.path.expanduser("~/.local/share/boswell-server/eval/wespeaker")
MODELS = os.path.expanduser("~/.local/share/boswell-server/models")
sys.path[:0] = [PHONE_TOOLS, WESPEAKER]
CAP = 8 * 16000


def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def onnx(path, input_name):
    import onnxruntime as ort
    try:
        ort.preload_dlls()   # cuDNN as PyTorch installed it
    except Exception:
        pass
    s = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return lambda x: unit(s.run(None, {input_name: x[None, :CAP].astype(np.float32)})[0])


def w2vbert():
    import torch
    from wespeaker.frontend.w2vbert import W2VBertFrontend
    from wespeaker.models.speaker_model import get_speaker_model
    model = get_speaker_model("W2VBert_Adapter_MFA")(feat_dim=1024, embed_dim=256, pooling_func="ASP", dropout=0.0, n_mfa_layers=-1,
                                                      adapter_dim=128, num_frontend_hidden_layers=24)
    frontend = W2VBertFrontend(model_name="facebook/w2v-bert-2.0", frozen=True, use_lora=False)
    model.add_module("frontend", frontend)
    sd = torch.load(os.path.join(MODELS, "eval", "wespeaker_w2vbert2_LM.pt"), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"w2v-BERT: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
    model = model.cuda().eval()

    def embed(x):
        with torch.no_grad():
            w = torch.from_numpy(x[:CAP]).float()[None, :].cuda()
            feats, _ = model.frontend(w, torch.tensor([w.shape[1]]).cuda())
            out = model(feats)
            e = out[-1] if isinstance(out, tuple) else out
            return unit(e.cpu().numpy())
    return embed


def main():
    from embed_bench import load_samples, evaluate
    sets = load_samples(400, 3)
    samples = sets["hand"] + sets["auto"]
    persons = [s["person"] for s in samples]
    nh = len(sets["hand"])
    models = {
        "WeSpeaker ResNet34-LM (phone, standard)": lambda: onnx(os.path.join(MODELS, "voiceprint.onnx"), "audio"),
        "ReDimNet2-B6 (phone, better)": lambda: onnx(os.path.join(MODELS, "speaker-id-redimnet2-b6.onnx"), "waveform"),
        "w2v-BERT 2.0 MFA (GPU only)": w2vbert,
    }
    only = os.environ.get("ONLY")
    for name, make in models.items():
        if only and only not in name:
            continue
        embed = make()
        t = time.time()
        E = [embed(s["full"]) for s in samples]
        ms = (time.time() - t) / len(samples) * 1000
        h = evaluate(E[:nh], persons[:nh]); a = evaluate(E, persons)
        print(f"{name:42s} hand: right person {h['ident'] * 100:5.1f}%  EER {h['eer'] * 100:5.2f}%   "
              f"all: right person {a['ident'] * 100:5.1f}%  EER {a['eer'] * 100:5.2f}%   {ms:.0f} ms/voice", flush=True)


if __name__ == "__main__":
    main()
