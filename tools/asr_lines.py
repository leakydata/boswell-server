"""Speech recognizers scored on the owner's corrected lines, line by line.

Only the lines someone corrected are an answer key (the rest of those recordings were
never checked), so each corrected line's audio is cut out and every engine transcribes
just that. The phone's own first attempt ("original") is the baseline.

    LD_LIBRARY_PATH=<nvidia libs> uv run --with faster-whisper python tools/asr_lines.py backup.zip keys.json
keys.json: [{"clip": …, "start": …, "end": …, "heard": …, "truth": …}, …]
"""
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boswell_server.audio import decode          # noqa: E402
from tools.asr_bench import edits, engines, words  # noqa: E402

SR = 16000


def main():
    z = zipfile.ZipFile(sys.argv[1])
    keys = json.load(open(sys.argv[2]))
    names = set(z.namelist())
    clips = {}
    for k in keys:
        if k["clip"] not in clips:
            a = next(f"files/clips/{k['clip']}{x}" for x in (".ogg", ".wav") if f"files/clips/{k['clip']}{x}" in names)
            clips[k["clip"]] = decode(z.read(a))
        audio = clips[k["clip"]]
        k["audio"] = audio[max(0, int((k["start"] - 0.3) * SR)): int((k["end"] + 0.3) * SR)]
    ref_words = sum(len(words(k["truth"])) for k in keys)
    print(f"{len(keys)} corrected lines, {ref_words} words")
    e = sum(edits(words(k["truth"]), words(k["heard"])) for k in keys)
    print(f"{'Phone (Nemotron, as first heard)':34s} {e / ref_words * 100:5.1f}% words wrong")
    rows = []
    for name, run in engines().items():
        run(keys[0]["audio"])
        t = time.time(); e = 0; outs = []
        for k in keys:
            hyp = run(k["audio"])
            hyp = hyp if isinstance(hyp, str) else getattr(hyp, "text", str(hyp))
            outs.append(hyp); e += edits(words(k["truth"]), words(hyp))
        print(f"{name:34s} {e / ref_words * 100:5.1f}% words wrong   {(time.time() - t) / len(keys) * 1000:.0f} ms per line", flush=True)
        rows.append((name, outs))
    print("\nline by line:")
    for i, k in enumerate(keys):
        print(f"\nTRUTH  {k['truth']}\nphone  {k['heard']}")
        for name, outs in rows:
            print(f"{name.split(' (')[0][:12]:12s} {outs[i].strip()}")


if __name__ == "__main__":
    main()
