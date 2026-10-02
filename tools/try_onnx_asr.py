import sys, time, numpy as np, onnxruntime as ort, onnx_asr
from boswell_server.audio import decode
if __name__ == "__main__":
    try: ort.preload_dlls()
    except Exception: pass
    a = decode(open(sys.argv[1], "rb").read())
    t = time.time()
    import os; m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", os.path.expanduser("~/.local/share/boswell-server/models/parakeet-v3-fp32"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]).with_timestamps()
    print("load %.1f s" % (time.time() - t))
    for _ in range(3):
        t = time.time(); r = m.recognize(a, sample_rate=16000); dt = time.time() - t
    print("transcribe %.0f ms" % (dt * 1000))
    print("text", r.text[:120])
    print("tokens", list(zip(r.tokens[:8], [round(x, 2) for x in r.timestamps[:8]])))
