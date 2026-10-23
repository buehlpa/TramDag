"""
Copyright 2025 Zurich University of Applied Sciences (ZHAW)
Pascal Buehler, Beate Sick, Oliver Duerr

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

JAX versions of the transformation-function building blocks.

The formulas intentionally mirror ``tramdag/legacy_torch/utils/continous.py`` and
``ordinal.py`` one-to-one (including the ``mean`` in :func:`h_dag` and the
un-normalised derivative in :func:`h_dag_dash`) so that a model trained with the
JAX engine is numerically interchangeable with one trained by the original
PyTorch code. ``tests/test_parity_torch.py`` checks this.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln as _np_gammaln

# Inside [L_START, R_START] (on the min-max scaled axis) h is a Bernstein polynomial,
# outside it is extrapolated linearly.
L_START = 0.0001
R_START = 1.0 - L_START


# --------------------------------------------------------------------------- continuous


def bernstein_basis(t, M):
    """Bernstein basis of degree ``M`` evaluated at ``t``; returns shape ``t.shape + (M+1,)``."""
    t = jnp.asarray(t)
    eps = jnp.finfo(t.dtype).eps
    t = jnp.clip(t[..., None], eps, 1.0 - eps)
    k_np = np.arange(M + 1, dtype=np.float64)
    # M is static: compute log(M choose k) on the host (gammaln is expensive to compile on GPU)
    log_binom = jnp.asarray(_np_gammaln(M + 1.0) - _np_gammaln(k_np + 1.0) - _np_gammaln(M - k_np + 1.0), t.dtype)
    k = jnp.asarray(k_np, t.dtype)
    log_pow = k * jnp.log(t) + (M - k) * jnp.log(1.0 - t)
    return jnp.exp(log_binom + log_pow)


def h_dag(t, thetas):
    """Bernstein polynomial on the scaled axis: ``mean_k B_k(t) * theta_k``.

    ``t`` has shape ``(n,)``; ``thetas`` has shape ``(n, b)`` or ``(b,)``.
    """
    b = thetas.shape[-1]
    return jnp.mean(bernstein_basis(t, b - 1) * thetas, axis=-1)


def h_dag_dash(t, thetas):
    """Derivative helper used by the original code: ``sum_k B^{b-2}_k(t) * (theta_{k+1}-theta_k)``."""
    b = thetas.shape[-1]
    dtheta = thetas[..., 1:] - thetas[..., :-1]
    return jnp.sum(bernstein_basis(t, b - 2) * dtheta, axis=-1)


def scale(y, k_min, k_max):
    return (y - k_min) / (k_max - k_min)


def h_extrapolated_scaled(thetas, t):
    """h on the scaled axis ``t``; linear extrapolation outside ``[L_START, R_START]``."""
    L = jnp.asarray(L_START, dtype=t.dtype)
    R = jnp.asarray(R_START, dtype=t.dtype)
    b0 = h_dag(jnp.full_like(t, L), thetas)
    s0 = h_dag_dash(jnp.full_like(t, L), thetas)
    b1 = h_dag(jnp.full_like(t, R), thetas)
    s1 = h_dag_dash(jnp.full_like(t, R), thetas)
    center = h_dag(t, thetas)
    h = jnp.where(t < L, s0 * (t - L) + b0, center)
    return jnp.where(t > R, s1 * (t - R) + b1, h)


def h_dash_extrapolated_scaled(thetas, t):
    L = jnp.asarray(L_START, dtype=t.dtype)
    R = jnp.asarray(R_START, dtype=t.dtype)
    s0 = h_dag_dash(jnp.full_like(t, L), thetas)
    s1 = h_dag_dash(jnp.full_like(t, R), thetas)
    center = h_dag_dash(t, thetas)
    hd = jnp.where(t < L, s0, center)
    return jnp.where(t > R, s1, hd)


def h_extrapolated(thetas, y, k_min, k_max):
    """Intercept part of the transformation function ``h_I(y)`` on the original axis."""
    return h_extrapolated_scaled(thetas, scale(y, k_min, k_max))


def h_dash_extrapolated(thetas, y, k_min, k_max):
    return h_dash_extrapolated_scaled(thetas, scale(y, k_min, k_max))


def transform_intercepts_continuous(theta_tilde):
    """Unconstrained network output -> strictly increasing Bernstein coefficients."""
    b = theta_tilde.shape[-1]
    shift = jnp.log(2.0) * b / 2
    widths = jnp.concatenate(
        [theta_tilde[..., :1], jax.nn.softplus(theta_tilde[..., 1:])], axis=-1
    )
    return jnp.cumsum(widths, axis=-1) - shift


def inverse_transform_intercepts_continuous(theta, eps=1e-12):
    """Inverse of :func:`transform_intercepts_continuous` (NumPy, float64 internally)."""
    theta = np.asarray(theta, dtype=np.float64)
    b = theta.shape[-1]
    shifted = theta + np.log(2.0) * b / 2
    widths = np.empty_like(shifted)
    widths[..., 0] = shifted[..., 0]
    widths[..., 1:] = shifted[..., 1:] - shifted[..., :-1]
    out = np.empty_like(shifted)
    out[..., 0] = widths[..., 0]
    out[..., 1:] = np.log(np.expm1(np.clip(widths[..., 1:], eps, None)))
    return out


# --------------------------------------------------------------------------- ordinal


def ordinal_cutpoints(int_in):
    """Network output ``(..., K-1)`` -> increasing finite cut-points ``theta_1 < ... < theta_{K-1}``."""
    first = int_in[..., :1]
    rest = first + jnp.cumsum(jnp.exp(int_in[..., 1:]), axis=-1)
    return jnp.concatenate([first, rest], axis=-1)


def transform_intercepts_ordinal(int_in):
    """Same as the torch version: cut-points padded with ``-inf`` / ``+inf`` -> shape ``(..., K+1)``."""
    cut = ordinal_cutpoints(int_in)
    pad_shape = cut.shape[:-1] + (1,)
    return jnp.concatenate(
        [jnp.full(pad_shape, -jnp.inf, cut.dtype), cut, jnp.full(pad_shape, jnp.inf, cut.dtype)],
        axis=-1,
    )


def inverse_transform_intercepts_ordinal(cutpoints, eps=1e-12):
    """Finite increasing cut-points ``(..., K-1)`` -> unconstrained network output (NumPy)."""
    cut = np.asarray(cutpoints, dtype=np.float64)
    diffs = np.diff(cut, axis=-1)
    return np.concatenate([cut[..., :1], np.log(np.clip(diffs, eps, None))], axis=-1)


# --------------------------------------------------------------------------- latent distribution


def logistic_cdf(x):
    return jax.nn.sigmoid(x)


def logistic_quantile(p):
    return jnp.log(p) - jnp.log1p(-p)
