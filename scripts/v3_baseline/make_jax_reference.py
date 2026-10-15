"""Reference outputs of the JAX climate-V3 for ``check_against_jax.py`` (run in the
``v3`` env).

python make_jax_reference.py /data/pmd/runs/v3-climate/best.pkl windows.npz ref.npz
"""

import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/ubuntu/Projects/v3")
import climate_data as cd  # noqa: E402
import climate_model as cm  # noqa: E402

ckpt_path, windows_path, out_path = sys.argv[1:4]
jax.config.update(
    "jax_default_matmul_precision", "highest"
)  # no TF32: reference for the port
ck = pickle.load(open(ckpt_path, "rb"))
m = ck["args"]
args = cm.default_args(
    num_basis=m["num_basis"],
    init_scale=m["init_scale"],
    t_init_scale=m["t_init_scale"],
    embed_dims=m["embed_dims"],
    num_blocks=m["num_blocks"],
    depths=m["depths"],
    attention_heads=m["attention_heads"],
    deformable_groups=m["deformable_groups"],
    output_dims=m["output_dims"],
)
data = np.load(windows_path)
coarse = data["coarse"][:3]  # (3, 9, C, 44, 72) physical
src = np.stack([cd.to_model_space(np.moveaxis(c[[0, 8]], 1, -1)) for c in coarse])
hyper, _, phi = cm.build_models(
    jax.random.PRNGKey(0), args, 5, jnp.zeros((1, 2, 44, 72, 5))
)
params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
enc = np.asarray(cm.encode(hyper, params, jnp.asarray(src)))
fine = np.asarray(cm.decode_full(hyper, phi, params, jnp.asarray(src)))
np.savez(out_path, source=src, encoding=enc, fine=fine)
print("saved", out_path, src.shape, enc.shape, fine.shape)
