"""Export a climate-V3 JAX checkpoint (``~/Projects/v3``) to a flat npz of numpy arrays.

Run in the ``v3`` conda env (it needs jax to unpickle). The pickle is read with a
restricted
unpickler that only allows array / dict classes.

    python export_params_npz.py /data/pmd/runs/v3-climate/best.pkl \\
        /data/pmd/v3_baseline/params.npz
"""

import os
import pickle
import sys

import numpy as np

ALLOWED = {
    ("collections", "OrderedDict"),
    ("builtins", "dict"),
    ("flax.core.frozen_dict", "FrozenDict"),
    ("jax._src.array", "_reconstruct_array"),
    ("jax._src.numpy.lax_numpy", "_reconstruct"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED or module.startswith(("numpy", "optax.")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def flatten(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from flatten(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", np.asarray(value)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    with open(src, "rb") as fh:
        ckpt = SafeUnpickler(fh).load()
    flat = dict(flatten(ckpt["params"]))
    flat["__args_json__"] = np.array(str(ckpt["args"]))
    flat["__step__"] = np.array(ckpt["step"])
    # normalization the model was trained with (climate_data.py of the V3 checkout)
    sys.path.insert(0, os.environ.get("V3_DIR", os.path.expanduser("~/Projects/v3")))
    import climate_data as cd

    flat["__names__"] = np.array(cd.OUT_NAMES)
    flat["__means__"] = np.asarray(cd.MEANS, dtype=np.float64)
    flat["__stds__"] = np.asarray(cd.STDS, dtype=np.float64)
    flat["__log_names__"] = np.array(list(cd.LOG_CHANNELS))
    flat["__log_scales__"] = np.array(list(cd.LOG_CHANNELS.values()), dtype=np.float64)
    np.savez(dst, **flat)
    print(f"wrote {len(flat)} arrays from step {ckpt['step']} to {dst}")
