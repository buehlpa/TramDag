"""
Joint, fully compiled training of all node models.

Why this is fast compared to the original implementation:

* the whole data set is encoded once and lives on the accelerator (no per-row
  ``Dataset.__getitem__`` / ``DataLoader`` overhead),
* all nodes are optimised in the same compiled step (their parameters are
  disjoint, so summing the node losses gives each node exactly its own gradient;
  Adam is element-wise, so this is equivalent to one Adam per node),
* mini-batches and several epochs run inside ``lax.scan`` -> one device call per
  chunk of epochs, best-model tracking and the per-epoch parameter histories are
  computed on device as well.
"""

import math
import time
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .node_model import node_nll_terms
from .transforms import ordinal_cutpoints, transform_intercepts_continuous


def to_device(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def n_rows(data: Dict):
    for d in data.values():
        return int(d["int"].shape[0])
    return 0


def _pad_rows(a, n_total):
    pad = n_total - a.shape[0]
    if pad <= 0:
        return a
    return jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)], axis=0)


def make_per_sample_nll(structures, nodes, minmax, chunk=131072):
    """jit-compiled ``params, data -> {node: per-sample NLL}`` evaluated in memory-bounded chunks."""

    def _single(params, data):
        return {
            n: node_nll_terms(structures[n], params[n], data[n], data[n]["y"], minmax.get(n, (0.0, 1.0)))
            for n in nodes
        }

    @jax.jit
    def per_sample(params, data):
        n = n_rows(data)
        if n <= chunk:
            return _single(params, data)
        nc = math.ceil(n / chunk)
        padded = jax.tree_util.tree_map(lambda a: _pad_rows(a, nc * chunk).reshape((nc, chunk) + a.shape[1:]), data)
        out = jax.lax.map(lambda d: _single(params, d), padded)
        return {k: v.reshape(-1)[:n] for k, v in out.items()}

    return per_sample


def linear_shift_weights(structures, params, node):
    st = structures[node]
    return [p[0]["w"][:, 0] for g, p in zip(st.shifts, params[node]["shifts"]) if g.is_linear]


def simple_intercept_thetas(structures, params, node):
    st = structures[node]
    if not st.has_simple_intercept:
        return None
    w = params[node]["int"][0]["w"][0]  # (n_thetas,)
    if st.target == "ordinal":
        return ordinal_cutpoints(w)
    return transform_intercepts_continuous(w)


class JointTrainer:
    """Compiled trainer for a set of node models that share one data set."""

    def __init__(self, structures, nodes, minmax, batch_size, learning_rates, optimizer=None, seed=0):
        self.structures = structures
        self.nodes = list(nodes)
        self.minmax = {n: tuple(float(v) for v in minmax[n]) for n in nodes if n in minmax}
        self.batch_size = int(batch_size)
        self.lrs = {n: float(learning_rates[n]) for n in self.nodes}
        self.custom_tx = optimizer
        self.tx = optimizer if optimizer is not None else optax.scale_by_adam()
        self.key = jax.random.PRNGKey(seed)
        self.per_sample = make_per_sample_nll(structures, self.nodes, self.minmax)
        self._chunk_fns = {}

    # ------------------------------------------------------------------ pieces

    def _batch_loss(self, params, batch, mask):
        losses = {}
        for n in self.nodes:
            st = self.structures[n]
            l = node_nll_terms(st, params[n], batch[n], batch[n]["y"], self.minmax.get(n, (0.0, 1.0)))
            losses[n] = jnp.sum(l * mask) / jnp.sum(mask)
        return sum(losses.values()), losses

    def _records(self, params):
        ls = {n: linear_shift_weights(self.structures, params, n) for n in self.nodes}
        si = {n: simple_intercept_thetas(self.structures, params, n) for n in self.nodes}
        si = {n: v for n, v in si.items() if v is not None}
        return ls, si

    def _make_chunk_fn(self, n_train, has_val, n_epochs):
        bs = min(self.batch_size, n_train)
        nb = math.ceil(n_train / bs)
        pad = nb * bs - n_train
        nodes = self.nodes
        node_index = {n: i for i, n in enumerate(nodes)}
        grad_fn = jax.value_and_grad(self._batch_loss, has_aux=True)

        def scale_updates(updates, active):
            out = {}
            for n in nodes:
                f = active[node_index[n]].astype(jnp.float32)
                if self.custom_tx is None:
                    f = f * (-self.lrs[n])
                out[n] = jax.tree_util.tree_map(lambda u: u * f, updates[n])
            return out

        def one_epoch(carry, epoch_local, train_data, val_data):
            params, opt_state, key, best_params, best_score, remaining = carry
            active = epoch_local < remaining
            key, sk = jax.random.split(key)
            perm = jax.random.permutation(sk, n_train)
            idx = jnp.concatenate([perm, jnp.zeros((pad,), perm.dtype)]).reshape(nb, bs)
            mask = jnp.concatenate([jnp.ones((n_train,), jnp.float32), jnp.zeros((pad,), jnp.float32)]).reshape(nb, bs)

            def step(c, xs):
                p, s = c
                idx_b, m_b = xs
                batch = jax.tree_util.tree_map(lambda a: a[idx_b], train_data)
                (_, per_node), grads = grad_fn(p, batch, m_b)
                upd, s = self.tx.update(grads, s, p)
                p = optax.apply_updates(p, scale_updates(upd, active))
                return (p, s), per_node

            (params, opt_state), per_batch = jax.lax.scan(step, (params, opt_state), (idx, mask))
            train_loss = jnp.stack([jnp.mean(per_batch[n]) for n in nodes])
            if has_val:
                ps = self.per_sample(params, val_data)
                val_loss = jnp.stack([jnp.mean(ps[n]) for n in nodes])
                score = val_loss
            else:
                val_loss = jnp.full((len(nodes),), jnp.nan)
                score = train_loss
            improved = active & (score < best_score)
            best_params = {
                n: jax.tree_util.tree_map(
                    lambda new, old: jnp.where(improved[node_index[n]], new, old), params[n], best_params[n]
                )
                for n in nodes
            }
            best_score = jnp.where(improved, score, best_score)
            ls, si = self._records(params)
            rec = {"train": train_loss, "val": val_loss, "ls": ls, "si": si}
            return (params, opt_state, key, best_params, best_score, remaining), rec

        @jax.jit
        def run(carry, train_data, val_data, epoch_ids):
            return jax.lax.scan(lambda c, e: one_epoch(c, e, train_data, val_data), carry, epoch_ids)

        return run

    # ------------------------------------------------------------------ public

    def fit(self, params, train_data, val_data, epochs, best_params=None, best_score=None,
            verbose=True, epochs_per_call=None, log_every=None):
        """Train for ``epochs[node]`` additional epochs per node.

        Returns ``params, best_params, best_score, history`` where history holds numpy arrays
        ``train``/``val`` of shape ``(E, n_nodes)`` and the per-epoch ``ls``/``si`` records.
        """
        nodes = self.nodes
        n_train = n_rows(train_data)
        has_val = val_data is not None and n_rows(val_data) > 0
        remaining = jnp.asarray([int(epochs[n]) for n in nodes], jnp.int32)
        total = int(max(epochs[n] for n in nodes)) if nodes else 0
        if best_params is None:
            best_params = params
        if best_score is None:
            best_score = jnp.full((len(nodes),), jnp.inf)
        else:
            best_score = jnp.asarray([best_score[n] for n in nodes], jnp.float32)
        opt_state = self.tx.init(params)
        chunk = int(epochs_per_call or max(1, min(total, 25)))
        log_every = log_every or chunk

        carry = (params, opt_state, self.key, best_params, best_score, remaining)
        val_arg = val_data if has_val else None
        records = []
        done = 0
        t0 = time.perf_counter()
        while done < total:
            e = min(chunk, total - done)
            key = (n_train, has_val, e)
            if key not in self._chunk_fns:
                self._chunk_fns[key] = self._make_chunk_fn(n_train, has_val, e)
            carry, rec = self._chunk_fns[key](carry, train_data, val_arg, jnp.arange(done, done + e))
            rec = jax.device_get(rec)
            records.append(rec)
            done += e
            if verbose and (done % log_every == 0 or done == total):
                tr, va = rec["train"][-1], rec["val"][-1]
                parts = " | ".join(
                    f"{n}: {tr[i]:.4f}/{va[i]:.4f}" for i, n in enumerate(nodes) if done <= int(epochs[n]) or done == total
                )
                print(f"[epoch {done:>5}/{total}] train/val NLL  {parts}   ({time.perf_counter()-t0:.1f}s)")
        params, _, self.key, best_params, best_score = carry[:5]

        def cat(path):
            arrs = [path(r) for r in records]
            return np.concatenate(arrs, axis=0) if arrs else np.zeros((0, len(nodes)))

        history = {
            "train": cat(lambda r: r["train"]),
            "val": cat(lambda r: r["val"]),
            "ls": {n: [np.concatenate([np.asarray(r["ls"][n][i]) for r in records], 0) for i in range(len(records[0]["ls"][n]))] if records else [] for n in nodes},
            "si": {n: np.concatenate([np.asarray(r["si"][n]) for r in records], 0) for n in nodes if records and n in records[0]["si"]},
            "seconds": time.perf_counter() - t0,
        }
        best_score = {n: float(v) for n, v in zip(nodes, np.asarray(best_score))}
        return params, best_params, best_score, history
