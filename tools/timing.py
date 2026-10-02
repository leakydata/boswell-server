"""Time each step of the pipeline on one recording: uv run python tools/timing.py <audio>"""
import sys
import time

from boswell_server.audio import decode
from boswell_server.pipeline import Engine

if __name__ == "__main__":
    a = decode(open(sys.argv[1], "rb").read())
    e = Engine()
    t = time.time(); e.warm(("wespeaker-resnet34-lm", "redimnet2-b6-vb2vox2-lm")); print("warm %.1f s" % (time.time() - t))
    for _ in range(3):
        t = time.time(); e.diarize(a); t1 = time.time(); w = e.transcribe(a); t2 = time.time(); e.tag(a); t3 = time.time()
    print("step diarize %.0f ms, transcribe %.0f ms, tag %.0f ms" % ((t1 - t) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000))
    for vm in ("wespeaker-resnet34-lm", "redimnet2-b6-vb2vox2-lm"):
        r = e.analyze(a, vm); print("total", vm, r["ms"], "ms")
    print("text", " ".join(x["text"] for x in w)[:120])
