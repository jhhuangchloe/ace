"""Check the PyTorch port against the JAX reference outputs (run in the ``fme`` env).

python check_against_jax.py v3_baseline.pt ref.npz
"""

import sys

import numpy as np
import torch

from fme.downscaling.v3_baseline import V3Baseline, V3Config

ckpt = torch.load(sys.argv[1], map_location="cpu")
model = V3Baseline(V3Config(**ckpt["config"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
ref = np.load(sys.argv[2])
src = torch.from_numpy(ref["source"]).float()
enc = model.encode(src)
fine = model(src)
for name, got, want in (
    ("encoding", enc, ref["encoding"]),
    ("fine output", fine, ref["fine"]),
):
    err = np.abs(got.numpy() - want)
    print(
        f"{name}: max abs err {err.max():.2e}, mean abs err {err.mean():.2e}, "
        f"ref std {want.std():.3f}"
    )
