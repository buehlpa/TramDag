"""
Neural network building blocks for the JAX engine.

Every term of a transformation function (simple/complex intercept, linear/complex
shift) is a small network with parameters stored as a plain pytree::

    [{"w": (n_in, n_hidden), "b": (n_hidden,)}, ..., {"w": (n_last, n_out)}]

Networks are looked up by the class names that the configuration file uses
(e.g. ``"LinearShift"``, ``"ComplexShiftDefaultTabular"``). Trailing digits, as in
``"ComplexShiftDefaultTabular12"`` (interaction terms), are ignored.

Custom networks no longer require editing the package source::

    from tramdag import register_model
    register_model("MyShift", hidden=(32, 32), activation="tanh")

or, for full control, ``register_model("MyNet", init=init_fn, apply=apply_fn)`` with
``init_fn(key, n_in, n_out) -> params`` and ``apply_fn(params, x) -> (n, n_out)``.
An inline spec such as ``"MLP(64,64,relu)"`` can also be used directly as model name.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "leaky_relu": jax.nn.leaky_relu,
}


@dataclass(frozen=True)
class MLPSpec:
    """Fully connected network. Hidden layers use a bias, the output layer does not
    (same convention as the original PyTorch models)."""

    hidden: Tuple[int, ...] = ()
    activation: str = "relu"
    out_bias: bool = False
    role: str = "any"  # "shift", "intercept" or "any" (only used for UI hints)
    description: str = ""

    def init(self, key, n_in, n_out):
        sizes = [n_in, *self.hidden, n_out]
        layers = []
        keys = jax.random.split(key, len(sizes) - 1)
        for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            kw, kb = jax.random.split(k)
            # PyTorch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias
            bound = 1.0 / jnp.sqrt(a)
            layer = {"w": jax.random.uniform(kw, (a, b), jnp.float32, -bound, bound)}
            if (not last) or self.out_bias:
                layer["b"] = jax.random.uniform(kb, (b,), jnp.float32, -bound, bound)
            layers.append(layer)
        return layers

    def apply(self, params, x):
        act = ACTIVATIONS[self.activation]
        n = len(params)
        for i, layer in enumerate(params):
            x = x @ layer["w"]
            if "b" in layer:
                x = x + layer["b"]
            if i < n - 1:
                x = act(x)
        return x


@dataclass(frozen=True)
class CustomSpec:
    init: Callable
    apply: Callable
    role: str = "any"
    description: str = ""


_REGISTRY = {
    # simple terms
    "SimpleIntercept": MLPSpec((), role="intercept", description="theta per class / Bernstein coefficient (no inputs)"),
    "LinearShift": MLPSpec((), role="shift", description="beta * x (interpretable log-odds shift)"),
    # defaults used by the original package
    "ComplexShiftDefaultTabular": MLPSpec((64, 128, 64), "relu", role="shift", description="MLP 64-128-64, ReLU"),
    "ComplexInterceptDefaultTabular": MLPSpec((8, 8), "relu", role="intercept", description="MLP 8-8, ReLU"),
    # custom examples from the original package
    "ComplexShiftCustomTabular": MLPSpec((128, 128, 128, 64, 32, 16), "tanh", role="shift", description="MLP 128-128-128-64-32-16, tanh"),
    "ComplexInterceptCustomTabular": MLPSpec((128, 128, 64, 64, 32, 16), "relu", role="intercept", description="MLP 128-128-64-64-32-16, ReLU"),
}

_INLINE = re.compile(r"^MLP\(\s*([\d\s,]*?)\s*(?:,\s*([a-z_]+))?\s*\)$", re.IGNORECASE)


def register_model(name, hidden=None, activation="relu", init=None, apply=None, role="any", description=""):
    """Register a network under ``name`` so it can be used in the configuration.

    Either give ``hidden`` (tuple of layer widths, an MLP is built) or ``init``/``apply``.
    """
    if not re.match(r"^[A-Za-z_][A-Za-z_]*$", name):
        raise ValueError("model names may only contain letters and '_' (digits are reserved for interaction groups)")
    if init is not None or apply is not None:
        if init is None or apply is None:
            raise ValueError("custom models need both init and apply")
        _REGISTRY[name] = CustomSpec(init, apply, role, description)
    else:
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}', choose from {sorted(ACTIVATIONS)}")
        _REGISTRY[name] = MLPSpec(tuple(hidden or ()), activation, role=role, description=description)
    return _REGISTRY[name]


def base_model_name(class_name: str) -> str:
    """Strip interaction-group digits: ``'ComplexShiftDefaultTabular12' -> 'ComplexShiftDefaultTabular'``."""
    if class_name.upper().startswith("MLP("):
        return class_name
    m = re.match(r"^(.*?)(\d*)$", class_name)
    return m.group(1)


def get_model_spec(class_name: str):
    name = base_model_name(class_name)
    if name in _REGISTRY:
        return _REGISTRY[name]
    m = _INLINE.match(name)
    if m:
        hidden = tuple(int(v) for v in re.split(r"[\s,]+", m.group(1).strip()) if v)
        act = (m.group(2) or "relu").lower()
        if act not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{act}' in '{class_name}'")
        return MLPSpec(hidden, act)
    if "image" in name.lower():
        raise NotImplementedError(
            f"'{name}' is an image model. The JAX engine currently supports tabular data only; "
            "use tramdag.legacy_torch for image experiments."
        )
    raise KeyError(
        f"Unknown model '{class_name}'. Known models: {sorted(_REGISTRY)}. "
        "Register your own with tramdag.register_model(...) or use an inline spec like 'MLP(32,32,relu)'."
    )


def available_models(role=None):
    """Names of registered models, optionally filtered by role ('shift' / 'intercept')."""
    return {
        k: getattr(v, "description", "")
        for k, v in _REGISTRY.items()
        if role is None or v.role in (role, "any")
    }
