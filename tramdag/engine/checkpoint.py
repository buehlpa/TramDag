"""
Parameter (de)serialisation.

Node parameters are stored as ``.npz`` files whose keys are the pytree paths, e.g.
``int/0/w`` or ``shifts/1/2/b``. :func:`params_from_torch_state_dict` converts a
``state_dict`` of the original PyTorch ``TramModel`` into the same pytree, which is
used by the parity tests and to load experiments trained with the legacy engine.
"""

import os
import re
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np


def _path_key(path):
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts)


def flatten_params(params):
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {_path_key(path): np.asarray(v) for path, v in leaves}


def unflatten_params(flat, template):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    out = []
    for path, ref in leaves:
        k = _path_key(path)
        if k not in flat:
            raise KeyError(f"parameter '{k}' missing in checkpoint")
        v = np.asarray(flat[k], dtype=np.float32)
        if v.shape != tuple(ref.shape):
            raise ValueError(f"parameter '{k}' has shape {v.shape}, expected {tuple(ref.shape)}")
        out.append(jnp.asarray(v))
    return jax.tree_util.tree_unflatten(treedef, out)


def save_params(path, params):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **flatten_params(params))
    os.replace(tmp, path)


def load_params(path, template):
    with np.load(path) as f:
        return unflatten_params(dict(f), template)


# --------------------------------------------------------------------------- torch interop


def _linear_layers(state_dict, prefix):
    """Ordered list of (weight, bias|None) of all Linear layers below ``prefix``."""
    layers = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        m = re.match(r"^(.*)\.(weight|bias)$", rest)
        if not m:
            continue
        mod, kind = m.groups()
        layers.setdefault(mod, {})[kind] = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v)
    return [(d["weight"], d.get("bias")) for d in layers.values()]


def params_from_torch_state_dict(state_dict, template):
    """Map a torch ``TramModel.state_dict()`` onto the JAX pytree ``template`` of one node."""

    def fill(layers_tmpl, torch_layers, what):
        if len(layers_tmpl) != len(torch_layers):
            raise ValueError(f"{what}: {len(torch_layers)} torch layers vs {len(layers_tmpl)} expected")
        out = []
        for t, (w, b) in zip(layers_tmpl, torch_layers):
            layer = {"w": jnp.asarray(w.T, jnp.float32)}
            if "b" in t:
                if b is None:
                    raise ValueError(f"{what}: torch layer has no bias but JAX layer expects one")
                layer["b"] = jnp.asarray(b, jnp.float32)
            out.append(layer)
        return out

    params = {"int": fill(template["int"], _linear_layers(state_dict, "nn_int."), "intercept")}
    params["shifts"] = [
        fill(t, _linear_layers(state_dict, f"nn_shift.{i}."), f"shift {i}") for i, t in enumerate(template["shifts"])
    ]
    return params


def params_to_torch_state_dict(params):
    """Inverse of :func:`params_from_torch_state_dict` (keys follow ``nn.Sequential``-free naming
    is not recoverable, so this returns ``{prefix: [(W, b), ...]}`` for manual copying)."""
    return {
        "nn_int": [(np.asarray(l["w"]).T, np.asarray(l["b"]) if "b" in l else None) for l in params["int"]],
        "nn_shift": [
            [(np.asarray(l["w"]).T, np.asarray(l["b"]) if "b" in l else None) for l in sh] for sh in params["shifts"]
        ],
    }
