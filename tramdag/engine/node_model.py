"""
Forward pass and likelihood of a single node model ``h(x_j | pa(x_j))``.

Continuous targets (CONTRAM)::

    h(y | pa) = h_I(y) + sum(shifts),      U = h(y | pa) ~ standard logistic
    NLL       = -[ log f_logistic(h) + log h'(y) ]

Ordinal targets (ONTRAM)::

    P(Y <= k | pa) = sigmoid(theta_k - sum(shifts))

Note the opposite sign convention of the shift terms (taken over from the original code).
"""

import jax
import jax.numpy as jnp

from .nets import get_model_spec
from .transforms import (
    h_dash_extrapolated,
    h_extrapolated,
    ordinal_cutpoints,
    transform_intercepts_continuous,
)


def init_node_params(key, st):
    keys = jax.random.split(key, 1 + len(st.shifts))
    params = {
        "int": get_model_spec(st.intercept_model).init(keys[0], st.n_int_features, st.n_thetas),
        "shifts": [get_model_spec(g.model_name).init(k, g.n_features, 1) for g, k in zip(st.shifts, keys[1:])],
    }
    return params


def node_forward(st, params, inputs):
    """Returns ``(int_out (n, n_thetas), shift_sum (n,), [shift_term_i (n,)])``."""
    int_out = get_model_spec(st.intercept_model).apply(params["int"], inputs["int"])
    terms = [
        get_model_spec(g.model_name).apply(p, x)[:, 0]
        for g, p, x in zip(st.shifts, params["shifts"], inputs["shifts"])
    ]
    n = int_out.shape[0]
    shift_sum = sum(terms) if terms else jnp.zeros((n,), int_out.dtype)
    return int_out, shift_sum, terms


def continuous_nll_terms(int_out, shift_sum, y, k_min, k_max):
    """Per-observation NLL and latent ``h`` for a continuous target."""
    thetas = transform_intercepts_continuous(int_out)
    h = h_extrapolated(thetas, y, k_min, k_max) + shift_sum
    log_latent = -h - 2.0 * jax.nn.softplus(-h)
    h_dash = h_dash_extrapolated(thetas, y, k_min, k_max)
    log_hdash = jnp.log(jnp.abs(h_dash)) - jnp.log(k_max - k_min)
    return -(log_latent + log_hdash), h


def ordinal_bounds(int_out, shift_sum, y):
    """Latent interval ``(U_lower, U_upper)`` of class ``y`` (``-inf``/``+inf`` at the ends)."""
    cut = ordinal_cutpoints(int_out) - shift_sum[:, None]  # (n, K-1)
    K1 = cut.shape[-1]
    lo = jnp.take_along_axis(cut, jnp.clip(y - 1, 0, K1 - 1)[:, None], axis=1)[:, 0]
    up = jnp.take_along_axis(cut, jnp.clip(y, 0, K1 - 1)[:, None], axis=1)[:, 0]
    lo = jnp.where(y <= 0, -jnp.inf, lo)
    up = jnp.where(y >= K1, jnp.inf, up)
    return lo, up


def ordinal_nll_terms(int_out, shift_sum, y):
    cut = ordinal_cutpoints(int_out) - shift_sum[:, None]
    K1 = cut.shape[-1]
    lo = jnp.take_along_axis(cut, jnp.clip(y - 1, 0, K1 - 1)[:, None], axis=1)[:, 0]
    up = jnp.take_along_axis(cut, jnp.clip(y, 0, K1 - 1)[:, None], axis=1)[:, 0]
    cdf_lo = jnp.where(y <= 0, 0.0, jax.nn.sigmoid(lo))
    cdf_up = jnp.where(y >= K1, 1.0, jax.nn.sigmoid(up))
    return -jnp.log(cdf_up - cdf_lo + 1e-16)


def node_nll_terms(st, params, inputs, y, minmax):
    int_out, shift_sum, _ = node_forward(st, params, inputs)
    if st.target == "continuous":
        nll, _ = continuous_nll_terms(int_out, shift_sum, y, minmax[0], minmax[1])
        return nll
    return ordinal_nll_terms(int_out, shift_sum, y)
