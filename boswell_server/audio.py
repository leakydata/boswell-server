"""Whatever the phone sends (Ogg Opus as it stores clips, or WAV) as 16 kHz mono float32."""
import io

import av
import numpy as np

from .config import SR


def decode(data: bytes) -> np.ndarray:
    with av.open(io.BytesIO(data)) as box:
        stream = box.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        parts = []
        for frame in box.decode(stream):
            for f in resampler.resample(frame):
                parts.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            parts.append(f.to_ndarray().reshape(-1))
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
