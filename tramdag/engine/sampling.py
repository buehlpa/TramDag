"""
Sampling from a fitted TRAM-DAG (observational, interventional and counterfactual).

Nodes are visited in causal order. For every node a latent ``U`` is either drawn
from the standard logistic distribution, taken from a user supplied latent frame
(counterfactuals, ``{node}_U``), or - for ordinal nodes with abducted latent
intervals ``{node}_U_lower``/``{node}_U_upper`` - drawn ``n_cf`` times from the
truncated logistic distribution. In the latter case the node and all its
descendants are represented by ``(n_obs, n_cf)`` draws ("distribution mode"),
exactly like the original implementation (which streamed the same draws through
per-observation CSV files).

Continuous values are recovered by inverting ``h`` (closed form in the linear
tails, bisection on the Bernstein part), ordinal values by counting the shifted
cut-points below ``U``.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .node_model import node_forward
from .transforms import (
    L_START,
    R_START,
    h_dag,
    h_dag_dash,
    ordinal_cutpoints,
    transform_intercepts_continuous,
)


def invert_h(thetas, shift, u, k_min, k_max, n_iter=48):
    """Solve ``h_I(x) + shift = u`` for ``x`` (vectorised over rows)."""
    target = u - shift
    dt = target.dtype
    n = target.shape[0]
    L = jnp.full((n,), L_START, dt)
    R = jnp.full((n,), R_START, dt)
    hL, sL = h_dag(L, thetas), h_dag_dash(L, thetas)
    hR, sR = h_dag(R, thetas), h_dag_dash(R, thetas)

    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        below = h_dag(mid, thetas) < target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (L, R))
    t = jnp.where(target < hL, L + (target - hL) / sL, jnp.where(target > hR, R + (target - hR) / sR, 0.5 * (lo + hi)))
    return t * (k_max - k_min) + k_min


def ordinal_from_latent(int_out, shift_sum, u):
    cut = ordinal_cutpoints(int_out) - shift_sum[:, None]
    return jnp.sum(u[:, None] > cut, axis=-1)


def truncated_logistic(rng, lower, upper, size):
    """Draw ``size`` samples per row from the standard logistic truncated to ``[lower, upper]``."""
    lower = np.asarray(lower, np.float64)[:, None]
    upper = np.asarray(upper, np.float64)[:, None]
    f_lo = 1.0 / (1.0 + np.exp(-lower))
    f_up = 1.0 / (1.0 + np.exp(-upper))
    v = rng.uniform(size=(lower.shape[0], size))
    p = np.clip(f_lo + v * (f_up - f_lo), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


class DagSampler:
    def __init__(self, structures, order, params, minmax, chunk=262144):
        self.structures = structures
        self.order = list(order)
        self.params = params
        self.minmax = minmax
        self.chunk = chunk
        self._fns = {}

    def _fn(self, node):
        if node not in self._fns:
            st = self.structures[node]
            mm = tuple(float(v) for v in self.minmax.get(node, (0.0, 1.0)))

            if st.target == "continuous":

                def f(params, inputs, u):
                    int_out, shift_sum, _ = node_forward(st, params, inputs)
                    thetas = transform_intercepts_continuous(int_out)
                    return invert_h(thetas, shift_sum, u, mm[0], mm[1])

            else:

                def f(params, inputs, u):
                    int_out, shift_sum, _ = node_forward(st, params, inputs)
                    return ordinal_from_latent(int_out, shift_sum, u).astype(jnp.float32)

            self._fns[node] = jax.jit(f)
        return self._fns[node]

    def _evaluate(self, node, columns, u):
        """Run the node model on flat inputs (len m) in chunks."""
        st = self.structures[node]
        m = u.shape[0]
        out = np.empty((m,), np.float32)
        fn = self._fn(node)
        for s in range(0, m, self.chunk):
            e = min(m, s + self.chunk)
            k = e - s
            size = _bucket(k, self.chunk)  # pad to a few fixed sizes -> no recompilation per n
            cols = {p: _pad(v[s:e], size) for p, v in columns.items()}
            inputs = st.encode_inputs(cols, size)
            uu = jnp.asarray(_pad(u[s:e].astype(np.float32), size))
            out[s:e] = np.asarray(fn(self.params[node], inputs, uu))[:k]
        return out

    def sample(self, n, do=None, latents_df=None, n_cf=10_000, seed=None, verbose=False):
        """Returns ``(sampled, latents, draws)``.

        ``sampled[node]`` is a 1d array, or for distribution-mode nodes either the class
        frequencies ``(n, levels)`` (ordinal) or ``{"vals", "counts"}`` (continuous, as in the
        original package). ``draws`` holds the raw ``(n, n_cf)`` draws of distribution-mode nodes.
        """
        do = dict(do or {})
        rng = np.random.default_rng(seed)
        if latents_df is not None:
            n = len(latents_df)
        values, sampled, latents, dist = {}, {}, {}, set()

        for node in self.order:
            st = self.structures[node]
            if node in do:
                v = np.full((n,), do[node], np.float32)
                values[node] = sampled[node] = v
                latents[node] = np.full((n,), np.nan, np.float32)
                if verbose:
                    print(f"[INFO] {node}: do({node}={do[node]})")
                continue

            has_interval = (
                latents_df is not None
                and f"{node}_U_lower" in latents_df.columns
                and f"{node}_U_upper" in latents_df.columns
            )
            if has_interval:
                u = truncated_logistic(
                    rng, latents_df[f"{node}_U_lower"].to_numpy(), latents_df[f"{node}_U_upper"].to_numpy(), n_cf
                ).astype(np.float32)
                mode = "interval latents (truncated logistic)"
            elif latents_df is not None and f"{node}_U" in latents_df.columns:
                u = latents_df[f"{node}_U"].to_numpy().astype(np.float32)
                mode = f"predefined latents {node}_U"
            else:
                u = rng.logistic(size=n).astype(np.float32)
                mode = "standard logistic latents"

            dist_parents = [p for p in st.parents if p in dist]
            distribution_mode = has_interval or bool(dist_parents)
            if verbose:
                extra = f", distribution over parents {dist_parents}" if dist_parents else ""
                print(f"[INFO] {node}: {mode}{extra}")

            if distribution_mode:
                M = n_cf
                cols = {}
                for p in st.parents:
                    v = values[p]
                    cols[p] = (v if v.ndim == 2 else np.repeat(v[:, None], M, axis=1)).reshape(-1)
                uu = u if u.ndim == 2 else np.repeat(u[:, None], M, axis=1)
                draws = self._evaluate(node, cols, uu.reshape(-1)).reshape(n, M)
                values[node] = draws
                latents[node] = uu
                dist.add(node)
                if st.target == "ordinal":
                    idx = draws.astype(np.int64)
                    freq = np.zeros((n, st.levels), np.float32)
                    np.add.at(freq, (np.repeat(np.arange(n), M), idx.reshape(-1)), 1.0)
                    sampled[node] = freq / M
                else:
                    sampled[node] = _unique_counts(draws)
            else:
                cols = {p: values[p] for p in st.parents}
                v = self._evaluate(node, cols, u)
                values[node] = sampled[node] = v
                latents[node] = u
        draws = {k: values[k] for k in dist}
        return sampled, latents, draws


def _bucket(k, cap):
    size = 1024
    while size < k:
        size *= 2
    return min(size, cap) if k <= cap else k


def _pad(a, size):
    a = np.asarray(a)
    if a.shape[0] == size:
        return a
    pad = np.zeros((size - a.shape[0],) + a.shape[1:], a.dtype)
    return np.concatenate([a, pad])


def _unique_counts(draws):
    rows = [np.unique(r, return_counts=True) for r in draws]
    width = max(len(v) for v, _ in rows)
    vals = np.full((len(rows), width), np.nan, np.float32)
    counts = np.zeros((len(rows), width), np.int64)
    for i, (v, c) in enumerate(rows):
        vals[i, : len(v)] = v
        counts[i, : len(c)] = c
    return {"vals": vals, "counts": counts}
