"""Convert the npz written by ``export_params_npz.py`` to a PyTorch checkpoint for
``fme.downscaling.v3_baseline.V3Baseline`` (run in the ``fme`` env).

    python convert_checkpoint.py params.npz v3_baseline.pt
"""

import sys

import numpy as np
import torch

from fme.downscaling.v3_baseline import V3Baseline, V3Config


def unflatten(flat: dict) -> dict:
    tree: dict = {}
    for key, value in flat.items():
        if key.startswith("__"):
            continue
        node = tree
        *parents, leaf = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return tree


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    data = dict(np.load(src))
    model = V3Baseline.from_jax_params(unflatten(data), V3Config())
    torch.save(
        {
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "source_step": int(data["__step__"]),
            "normalization": {
                "names": [str(n) for n in data["__names__"]],
                "means": [float(m) for m in data["__means__"]],
                "stds": [float(s) for s in data["__stds__"]],
                "log_scales": {
                    str(n): float(s)
                    for n, s in zip(data["__log_names__"], data["__log_scales__"])
                },
            },
        },
        dst,
    )
    print(f"saved {dst} (jax step {int(data['__step__'])})")
