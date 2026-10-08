import os

__version__ = "0.4.0"

# NeMo calls torch.compile when it's imported (transformer_encoder.py), which makes PyTorch start
# a pool of compile workers, one per core up to 32, ~380 MB each. Nothing here compiles anything,
# so compiling in-process is plenty. It has to be set before torch is imported.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
