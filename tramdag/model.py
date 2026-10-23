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

:class:`TramDagModel` - the user facing model class, running on the JAX engine.

The public methods mirror the original PyTorch class (``from_config``,
``from_directory``, ``fit``, ``sample``, ``get_latent``, ``nll``, ``summary`` and the
``plot_*`` diagnostics) and the experiment directory keeps the same layout and
JSON files, so existing notebooks keep working. Parameters are stored as
``.npz`` files (``initial_params.npz``, ``best_params.npz``, ``last_params.npz``);
legacy ``*.pt`` checkpoints are converted on the fly when torch is installed.
"""

import json
import os
import shutil
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .TramDagConfig import TramDagConfig
from .engine.checkpoint import load_params, params_from_torch_state_dict, save_params
from .engine.nets import base_model_name, get_model_spec
from .engine.node_model import continuous_nll_terms, init_node_params, node_forward, ordinal_bounds
from .engine.sampling import DagSampler
from .engine.structure import build_node_structure, topological_order
from .engine.training import JointTrainer, make_per_sample_nll, to_device
from .engine.transforms import (
    h_extrapolated,
    inverse_transform_intercepts_continuous,
    inverse_transform_intercepts_ordinal,
    ordinal_cutpoints,
    transform_intercepts_continuous,
)

_STATE_FILES = {"best": "best_params.npz", "last": "last_params.npz", "init": "initial_params.npz"}
_LEGACY_FILES = {"best": "best_model.pt", "last": "last_model.pt", "init": "initial_model.pt"}

# keyword arguments of the original torch fit() that have no meaning for the JAX engine
_IGNORED_FIT_KWARGS = {
    "callbacks", "schedulers", "use_scheduler", "train_mode", "num_workers",
    "persistent_workers", "prefetch_factor", "overwrite_inital_weights", "optimizers",
    "use_dataloader", "use_precomputed", "pin_memory", "shuffle",
}


def _resolve_device(device):
    if device in (None, "auto"):
        return jax.devices()[0]
    if isinstance(device, str):
        kind = {"cuda": "gpu"}.get(device.split(":")[0].lower(), device.split(":")[0].lower())
        idx = int(device.split(":")[1]) if ":" in device else 0
        try:
            return jax.devices(kind)[idx]
        except RuntimeError:
            warnings.warn(f"device '{device}' not available, falling back to {jax.devices()[0]}")
            return jax.devices()[0]
    return device


class TramDagModel:
    """Probabilistic causal model: one transformation model per node of the DAG."""

    DEFAULTS_CONFIG = {
        "set_initial_weights": False,
        "debug": False,
        "verbose": False,
        "device": "auto",
        "initial_data": None,
        "overwrite_initial_weights": True,
        "seed": 0,
    }

    DEFAULTS_FIT = {
        "epochs": 100,
        "train_list": None,
        "learning_rate": 0.01,
        "batch_size": 1000,
        "device": "auto",
        "optimizer": None,
        "save_linear_shifts": True,
        "save_simple_intercepts": True,
        "debug": False,
        "verbose": True,
        "return_history": False,
        "seed": None,
        "epochs_per_call": None,
        "resume": True,
    }

    def __init__(self):
        self.debug = False
        self.verbose = False
        self.device = _resolve_device("auto")
        self.params = {}

    # ================================================================== construction

    @classmethod
    def from_config(cls, cfg, **kwargs):
        """Build one model per node from a :class:`TramDagConfig`.

        Keyword arguments (scalars or ``{node: value}`` dicts where it makes sense):
        ``device``, ``seed``, ``verbose``, ``debug``, ``set_initial_weights`` (warm start of
        intercepts/linear shifts with R ``tram::Colr`` / ``MASS::polr``), ``initial_data``
        (training DataFrame or CSV path for the warm start), ``overwrite_initial_weights``.
        """
        self = cls()
        unknown = set(kwargs) - set(cls.DEFAULTS_CONFIG)
        if unknown:
            raise ValueError(f"[from_config] Unknown parameter(s): {', '.join(sorted(unknown))}")
        settings = {**cls.DEFAULTS_CONFIG, **kwargs}
        self.cfg = cfg
        if getattr(cfg, "CONF_DICT_PATH", None) and os.path.exists(cfg.CONF_DICT_PATH):
            cfg.update()
        cfg._verify_completeness()
        if getattr(cfg, "CONF_DICT_PATH", None):
            try:
                cfg.save()
            except Exception as e:  # pragma: no cover - best effort like the original
                print(f"[WARNING] Could not save configuration: {e}")

        self.debug, self.verbose = bool(settings["debug"]), bool(settings["verbose"])
        self.device = _resolve_device(settings["device"])
        self.nodes_dict = cfg.conf_dict["nodes"]
        self.order = topological_order(self.nodes_dict)
        self.structures = {n: build_node_structure(n, self.nodes_dict) for n in self.order}
        self.settings = settings

        key = jax.random.PRNGKey(int(settings["seed"] or 0))
        keys = dict(zip(self.order, jax.random.split(key, len(self.order))))
        with jax.default_device(self.device):
            self.params = {n: init_node_params(keys[n], self.structures[n]) for n in self.order}

        set_init = settings["set_initial_weights"]
        for n in self.order:
            if (set_init[n] if isinstance(set_init, dict) else set_init):
                self.params[n] = self._r_warm_start(n, settings["initial_data"])

        self.best_params = dict(self.params)
        exp_dir = self.experiment_dir
        if exp_dir:
            for n in self.order:
                path = os.path.join(exp_dir, n, _STATE_FILES["init"])
                if settings["overwrite_initial_weights"] or not os.path.exists(path):
                    save_params(path, self.params[n])
        if self.verbose:
            for n in self.order:
                print(f"[INFO] {self.structures[n].formula()}")
        return self

    @classmethod
    def from_directory(cls, EXPERIMENT_DIR, device="auto", debug=False, verbose=False, state="best"):
        """Load configuration, scaling and trained parameters from an experiment directory."""
        config_path = os.path.join(EXPERIMENT_DIR, "configuration.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[ERROR] Config file not found at {config_path}")
        cfg = TramDagConfig.load_json(config_path)
        self = cls.from_config(cfg, device=device, debug=debug, verbose=verbose, overwrite_initial_weights=False)
        minmax_path = os.path.join(EXPERIMENT_DIR, "min_max_scaling.json")
        if not os.path.exists(minmax_path):
            raise FileNotFoundError(f"[ERROR] MinMax file not found at {minmax_path}")
        with open(minmax_path) as f:
            self.minmax_dict = json.load(f)
        for n in self.order:
            p = self._load_state(n, "last")
            if p is not None:
                self.params[n] = p
            b = self._load_state(n, state)
            self.best_params[n] = b if b is not None else self.params[n]
        if verbose or debug:
            print(f"[INFO] Loaded TramDagModel from {EXPERIMENT_DIR}")
        return self

    # ================================================================== helpers

    @property
    def experiment_dir(self):
        try:
            return self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except (KeyError, TypeError, AttributeError):
            return None

    def _node_dir(self, node):
        if not self.experiment_dir:
            raise ValueError("No EXPERIMENT_DIR in cfg.conf_dict['PATHS'].")
        return os.path.join(self.experiment_dir, node)

    @property
    def models(self):
        """Node structures (kept under the old attribute name for compatibility)."""
        return self.structures

    def _minmax(self, node):
        if not hasattr(self, "minmax_dict"):
            raise RuntimeError(
                "[ERROR] minmax_dict not found. Call .fit() or .load_or_compute_minmax(train_df) first."
            )
        return tuple(float(v) for v in self.minmax_dict[node])

    def _load_state(self, node, state):
        """Parameters of ``state`` ('best' | 'last' | 'init') from disk, or None."""
        node_dir = self._node_dir(node)
        template = self.params[node]
        path = os.path.join(node_dir, _STATE_FILES[state])
        if os.path.exists(path):
            return load_params(path, template)
        legacy = os.path.join(node_dir, _LEGACY_FILES[state])
        if os.path.exists(legacy):
            try:
                import torch
            except ImportError:
                warnings.warn(f"{legacy} is a torch checkpoint; install torch to convert it.")
                return None
            sd = torch.load(legacy, map_location="cpu")
            return params_from_torch_state_dict(sd, template)
        return None

    def _params_for(self, state):
        if state == "best":
            return self.best_params
        if state == "last":
            return self.params
        if state == "init":
            out = {}
            for n in self.order:
                p = self._load_state(n, "init") if self.experiment_dir else None
                out[n] = p if p is not None else self.params[n]
            return out
        raise ValueError("state must be one of 'best', 'last', 'init'")

    def _encode(self, df, nodes, with_target=True):
        return {n: to_device(self.structures[n].encode_frame(df, with_target=with_target)) for n in nodes}

    @staticmethod
    def _as_frame(data):
        if isinstance(data, pd.DataFrame):
            return data
        if data is None:
            return None
        if hasattr(data, "df") and isinstance(data.df, pd.DataFrame):  # legacy TramDagDataset
            return data.df
        raise TypeError(f"[ERROR] data must be a pandas DataFrame, got {type(data)}")

    def _r_warm_start(self, node, initial_data):
        """Initialise intercepts (and pure linear shifts) from R Colr/POLR estimates."""
        from .utils.r_subprocess import fit_r_model_subprocess
        import tempfile

        if initial_data is None:
            raise ValueError("initial_data (DataFrame or CSV path) is required when set_initial_weights=True")
        st = self.structures[node]
        cleanup = None
        if hasattr(initial_data, "to_csv"):
            fd, cleanup = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            initial_data.to_csv(cleanup, index=False)
            path = cleanup
        else:
            path = initial_data
        only_ls = bool(st.shifts) and all(g.is_linear and len(g.parents) == 1 for g in st.shifts)
        covs = [g.parents[0] for g in st.shifts] if only_ls else None
        try:
            thetas_R, shift_coefs = fit_r_model_subprocess(
                node, "continous" if st.target == "continuous" else "ordinal", st.n_thetas, path,
                ls_shift_covariates=covs, debug=self.debug,
            )
        finally:
            if cleanup and os.path.exists(cleanup):
                os.remove(cleanup)
        return apply_warm_start(self.params[node], st, thetas_R, shift_coefs if only_ls else None)

    # ================================================================== scaling

    def load_or_compute_minmax(self, td_train_data=None, use_existing=False, write=True):
        """5% / 95% quantiles of the training data, used to scale continuous targets."""
        path = os.path.join(self.experiment_dir, "min_max_scaling.json") if self.experiment_dir else None
        if use_existing:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"MinMax file not found: {path}")
            with open(path) as f:
                self.minmax_dict = json.load(f)
            return self.minmax_dict
        df = self._as_frame(td_train_data)
        num = df.select_dtypes(include="number")  # excludes bool / categorical (only continuous targets are scaled)
        q = num.quantile([0.05, 0.95])
        self.minmax_dict = {c: [float(q.loc[0.05, c]), float(q.loc[0.95, c])] for c in q.columns}
        if write and path:
            os.makedirs(self.experiment_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.minmax_dict, f, indent=4)
        return self.minmax_dict

    # ================================================================== fit

    def fit(self, train_data, val_data=None, **kwargs):
        """Train all (or ``train_list``) node models jointly.

        Main options: ``epochs``, ``learning_rate``, ``batch_size`` (each a scalar, or for
        ``epochs``/``learning_rate`` a ``{node: value}`` dict), ``optimizer`` (an optax
        ``GradientTransformation`` replacing Adam), ``seed``, ``verbose``, ``resume`` (continue
        from ``last_params.npz`` and the stored histories, default True).
        Defaults come from ``cfg.conf_dict['training']`` when present (written by the
        configuration editor), otherwise from ``DEFAULTS_FIT``.
        Arguments specific to torch DataLoaders (``num_workers`` ...) are accepted and ignored.
        """
        ignored = set(kwargs) & _IGNORED_FIT_KWARGS
        for k in ignored:
            kwargs.pop(k)
        unknown = set(kwargs) - set(self.DEFAULTS_FIT)
        if unknown:
            raise ValueError(f"[fit] Unknown parameter(s): {', '.join(sorted(unknown))}")
        cfg_training = {k: v for k, v in (self.cfg.conf_dict.get("training") or {}).items() if k in self.DEFAULTS_FIT}
        s = {**self.DEFAULTS_FIT, **cfg_training, **kwargs}
        self.verbose, self.debug = bool(s["verbose"]), bool(s["debug"])
        if "device" in kwargs:
            self.device = _resolve_device(s["device"])

        train_df = self._as_frame(train_data)
        val_df = self._as_frame(val_data)
        nodes = [n for n in self.order if n in (s["train_list"] or self.order)]
        per_node = lambda v, n: v[n] if isinstance(v, dict) else v

        self.load_or_compute_minmax(train_df, use_existing=False, write=True)

        # ---- resume
        hist_old, start, remaining = {}, {}, {}
        for n in nodes:
            hist_old[n] = self._read_histories(n) if s["resume"] else None
            last = self._load_state(n, "last") if (s["resume"] and self.experiment_dir) else None
            if hist_old[n] and last is not None:
                self.params[n] = last
                best = self._load_state(n, "best")
                self.best_params[n] = best if best is not None else last
                start[n] = len(hist_old[n]["train"])
            else:
                hist_old[n] = None
                start[n] = 0
            remaining[n] = max(0, int(per_node(s["epochs"], n)) - start[n])
            if self.verbose:
                if remaining[n] == 0:
                    print(f"[INFO] Node {n} already trained for {start[n]} epochs. Skipping.")
                elif start[n]:
                    print(f"[INFO] Node {n}: resuming at epoch {start[n]}.")
        nodes = [n for n in nodes if remaining[n] > 0]
        if not nodes:
            return {} if s["return_history"] else None

        if self.verbose:
            print(f"[INFO] Training {len(nodes)} node(s) jointly on {self.device} "
                  f"({len(train_df)} rows, batch size {s['batch_size']})")
            for n in nodes:
                print(f"       {self.structures[n].formula()}")

        best_score = {}
        for n in nodes:
            h = hist_old[n]
            if h:
                vals = [v for v in (h["val"] if any(np.isfinite(h["val"])) else h["train"]) if np.isfinite(v)]
                best_score[n] = min(vals) if vals else np.inf
            else:
                best_score[n] = np.inf

        seed = s["seed"] if s["seed"] is not None else int(self.settings.get("seed") or 0) + sum(start.values())
        with jax.default_device(self.device):
            train_enc = self._encode(train_df, nodes)
            val_enc = self._encode(val_df, nodes) if val_df is not None and len(val_df) else None
            trainer = JointTrainer(
                {n: self.structures[n] for n in nodes}, nodes, {n: self._minmax(n) for n in nodes if self.structures[n].target == "continuous"},
                s["batch_size"], {n: per_node(s["learning_rate"], n) for n in nodes}, optimizer=s["optimizer"], seed=seed,
            )
            p_sub = {n: self.params[n] for n in nodes}
            b_sub = {n: self.best_params.get(n, self.params[n]) for n in nodes}
            p_sub, b_sub, best_score, hist = trainer.fit(
                p_sub, train_enc, val_enc, remaining, best_params=b_sub, best_score=best_score,
                verbose=self.verbose, epochs_per_call=s["epochs_per_call"],
            )
        for n in nodes:
            self.params[n] = p_sub[n]
            self.best_params[n] = b_sub[n]
        self._write_fit_outputs(nodes, hist, hist_old, start, remaining, s)
        if self.verbose:
            print(f"[INFO] Training finished in {hist['seconds']:.1f}s")
        if s["return_history"]:
            return {n: (self._read_histories(n) or {}) for n in nodes}
        return None

    def _read_histories(self, node):
        if not self.experiment_dir:
            return None
        d = self._node_dir(node)
        tp, vp = os.path.join(d, "train_loss_hist.json"), os.path.join(d, "val_loss_hist.json")
        if not (os.path.exists(tp) and os.path.exists(vp)):
            return None
        with open(tp) as f:
            tr = json.load(f)
        with open(vp) as f:
            va = json.load(f)
        return {"train": [float("nan") if v is None else v for v in tr], "val": [float("nan") if v is None else v for v in va]}

    def _write_fit_outputs(self, nodes, hist, hist_old, start, remaining, s):
        if not self.experiment_dir:
            return
        for i, n in enumerate(nodes):
            d = self._node_dir(n)
            os.makedirs(d, exist_ok=True)
            E = remaining[n]
            tr = [float(v) for v in hist["train"][:E, i]]
            va = [float(v) if np.isfinite(v) else None for v in hist["val"][:E, i]]
            if hist_old[n]:
                tr = list(hist_old[n]["train"]) + tr
                va = [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in hist_old[n]["val"]] + va
            with open(os.path.join(d, "train_loss_hist.json"), "w") as f:
                json.dump(tr, f)
            with open(os.path.join(d, "val_loss_hist.json"), "w") as f:
                json.dump(va, f)
            save_params(os.path.join(d, _STATE_FILES["last"]), self.params[n])
            save_params(os.path.join(d, _STATE_FILES["best"]), self.best_params[n])

            st = self.structures[n]
            lin = [g for g in st.shifts if g.is_linear]
            if s["save_linear_shifts"] and lin:
                path = os.path.join(d, "linear_shifts_all_epochs.json")
                data = self._json_or_empty(path) if start[n] else {}
                for e in range(E):
                    data[f"epoch_{start[n] + e + 1}"] = {
                        f"ls({', '.join(g.parents)})": _squeeze_list(hist["ls"][n][j][e]) for j, g in enumerate(lin)
                    }
                with open(path, "w") as f:
                    json.dump(data, f)
            if s["save_simple_intercepts"] and n in hist["si"]:
                path = os.path.join(d, "simple_intercepts_all_epochs.json")
                data = self._json_or_empty(path) if start[n] else {}
                for e in range(E):
                    data[f"epoch_{start[n] + e + 1}"] = [[float(v)] for v in hist["si"][n][e]]
                with open(path, "w") as f:
                    json.dump(data, f)

    @staticmethod
    def _json_or_empty(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    # ================================================================== evaluation

    def nll(self, data, variables=None, state="best"):
        """Mean negative log-likelihood per node on ``data``."""
        df = self._as_frame(data)
        nodes = [n for n in self.order if variables is None or n in variables]
        params = self._params_for(state)
        mm = {n: self._minmax(n) for n in nodes if self.structures[n].target == "continuous"}
        with jax.default_device(self.device):
            fn = make_per_sample_nll(self.structures, nodes, mm)
            out = fn({n: params[n] for n in nodes}, self._encode(df, nodes))
        return {n: float(jnp.mean(out[n])) for n in nodes}

    def get_latent(self, df, verbose=False, state="best"):
        """Abduction: ``{node}_U = h(x | pa)`` for continuous nodes, ``{node}_U_lower/_U_upper``
        (the latent interval of the observed class) for ordinal nodes."""
        params = self._params_for(state)
        cols = {}
        with jax.default_device(self.device):
            for n in self.order:
                st = self.structures[n]
                enc = self._encode(df, [n])[n]
                int_out, shift_sum, _ = node_forward(st, params[n], enc)
                cols[n] = df[n].to_numpy()
                if st.target == "continuous":
                    k_min, k_max = self._minmax(n)
                    _, h = continuous_nll_terms(int_out, shift_sum, enc["y"], k_min, k_max)
                    cols[f"{n}_U"] = np.asarray(h)
                else:
                    lo, up = ordinal_bounds(int_out, shift_sum, enc["y"])
                    cols[f"{n}_U_lower"] = np.asarray(lo)
                    cols[f"{n}_U_upper"] = np.asarray(up)
        out = pd.DataFrame(cols, index=df.index)
        if verbose:
            print("[INFO] Final latent DataFrame shape:", out.shape)
        return out

    def transformation(self, node, df, x=None, state="best"):
        """Evaluate ``h(x | pa)`` for the rows of ``df`` (continuous nodes).

        If ``x`` (1d grid) is given, returns an array ``(len(df), len(x))`` with ``h`` evaluated on
        the grid for each row's parents; otherwise ``h`` at the observed ``df[node]``.
        """
        st = self.structures[node]
        if st.target != "continuous":
            raise ValueError("transformation() is defined for continuous nodes; use cutpoints() for ordinal ones")
        params = self._params_for(state)[node]
        k_min, k_max = self._minmax(node)
        with jax.default_device(self.device):
            enc = to_device(st.encode_frame(df, with_target=False))
            int_out, shift_sum, _ = node_forward(st, params, enc)
            thetas = transform_intercepts_continuous(int_out)
            if x is None:
                y = jnp.asarray(df[node].to_numpy(), jnp.float32)
                return np.asarray(h_extrapolated(thetas, y, k_min, k_max) + shift_sum)
            grid = jnp.asarray(np.asarray(x, np.float32))
            f = jax.vmap(lambda th, s: h_extrapolated(jnp.broadcast_to(th, (grid.shape[0], th.shape[0])), grid, k_min, k_max) + s)
            return np.asarray(f(thetas, shift_sum))

    def cutpoints(self, node, df, state="best"):
        """Shifted cut-points ``theta_k - eta(pa)`` of an ordinal node, shape ``(len(df), K-1)``."""
        st = self.structures[node]
        params = self._params_for(state)[node]
        with jax.default_device(self.device):
            enc = to_device(st.encode_frame(df, with_target=False))
            int_out, shift_sum, _ = node_forward(st, params, enc)
            return np.asarray(ordinal_cutpoints(int_out) - shift_sum[:, None])

    def shift_term(self, node, term=0, x=None, state="best"):
        """Evaluate one shift network of ``node`` on input ``x`` (``(n, n_features)`` or 1d).

        ``term`` is the index of the shift group or the name of (one of) its parents.
        Example: inspect a learned complex shift ``cs(x2)`` of ``x3`` on a grid::

            grid = np.linspace(-1, 1, 200)
            plt.plot(grid, model.shift_term("x3", "x2", grid))
        """
        st = self.structures[node]
        if isinstance(term, str):
            idx = [i for i, g in enumerate(st.shifts) if term in g.parents]
            if not idx:
                raise KeyError(f"'{term}' is not a shift input of '{node}'")
            term = idx[0]
        g = st.shifts[term]
        x = np.asarray(x, np.float32)
        if x.ndim == 1:
            x = x[:, None]
        p = self._params_for(state)[node]["shifts"][term]
        return np.asarray(get_model_spec(g.model_name).apply(p, jnp.asarray(x)))[:, 0]

    # ================================================================== histories / parameters

    def loss_history(self):
        out = {}
        for n in self.order:
            h = self._read_histories(n)
            out[n] = {"train": h["train"] if h else None, "validation": h["val"] if h else None}
        return out

    def linear_shift_history(self):
        out = {}
        for n in self.order:
            path = os.path.join(self._node_dir(n), "linear_shifts_all_epochs.json")
            if os.path.exists(path):
                out[n] = pd.read_json(path)
            elif self.debug:
                print(f"[DEBUG] No linear shift history found for node '{n}' at {path}")
        return out

    def simple_intercept_history(self):
        out = {}
        for n in self.order:
            path = os.path.join(self._node_dir(n), "simple_intercepts_all_epochs.json")
            if os.path.exists(path):
                out[n] = pd.read_json(path)
            elif self.debug:
                print(f"[DEBUG] No simple intercept history found for node '{n}' at {path}")
        return out

    def get_thetas(self, node, state="best"):
        """Simple-intercept parameters (Bernstein coefficients or ordinal cut-points) as ``[[t1], [t2], ...]``."""
        st = self.structures[node]
        if not st.has_simple_intercept:
            return None
        w = self._params_for(state)[node]["int"][0]["w"][0]
        th = ordinal_cutpoints(w) if st.target == "ordinal" else transform_intercepts_continuous(w)
        return [[float(v)] for v in np.asarray(th)]

    def get_linear_shifts(self, node, state="best"):
        st = self.structures[node]
        p = self._params_for(state)[node]
        return {
            f"ls({', '.join(g.parents)})": _squeeze_list(np.asarray(sp[0]["w"][:, 0]))
            for g, sp in zip(st.shifts, p["shifts"]) if g.is_linear
        }

    def get_linear_shifts_dict(self):
        return {s: {n: self.get_linear_shifts(n, s) for n in self.order} for s in ("best", "last", "init")}

    def get_simple_intercepts_dict(self):
        return {s: {n: self.get_thetas(n, s) for n in self.order} for s in ("best", "last", "init")}

    def coefficients(self, state="best"):
        """Tidy table of all interpretable coefficients (linear shifts per parent / one-hot level)."""
        rows = []
        for n in self.order:
            st = self.structures[n]
            p = self._params_for(state)[n]
            for g, sp in zip(st.shifts, p["shifts"]):
                if not g.is_linear:
                    continue
                w = np.asarray(sp[0]["w"][:, 0])
                labels = []
                for par in g.parents:
                    k = st.parent_levels.get(par)
                    labels += [f"{par}={i}" for i in range(k)] if k else [par]
                for lab, v in zip(labels, w):
                    rows.append({"node": n, "term": "ls", "input": lab, "coefficient": float(v)})
        return pd.DataFrame(rows, columns=["node", "term", "input", "coefficient"])

    def get_train_val_nll(self, node, mode):
        h = self._read_histories(node)
        if not h:
            return None, None
        tr, va = np.array(h["train"], float), np.array(h["val"], float)
        if mode == "init":
            i = 0
        elif mode == "last":
            i = len(tr) - 1
        elif mode == "best":
            i = int(np.nanargmin(va)) if np.isfinite(va).any() else int(np.nanargmin(tr))
        else:
            raise ValueError("mode must be one of 'best', 'last', 'init'")
        return float(tr[i]), float(va[i])

    def formulas(self):
        return {n: self.structures[n].formula() for n in self.order}

    # ================================================================== sampling

    def sample(self, do_interventions=None, predefined_latent_samples_df=None, **kwargs):
        """Sample from the fitted SCM.

        * observational: ``model.sample(number_of_samples=10_000)``
        * interventional: ``model.sample(do_interventions={"x1": 0.5})``
        * counterfactual: ``u = model.get_latent(obs_df)`` then
          ``model.sample(do_interventions={...}, predefined_latent_samples_df=u)``

        Options: ``number_of_samples`` (10 000), ``number_of_counterfactual_samples`` (10 000, draws
        per observation for ordinal latent intervals), ``seed``, ``state`` ('best'),
        ``use_initial_weights_for_sampling``, ``save`` (write ``sampling/sampled.npy`` per node),
        ``verbose``. Returns ``(sampled_by_node, latents_by_node)`` with numpy arrays.
        """
        s = {
            "number_of_samples": 10_000,
            "number_of_counterfactual_samples": 10_000,
            "seed": None,
            "state": "best",
            "use_initial_weights_for_sampling": False,
            "save": True,
            "delete_all_previously_sampled": True,
            "verbose": self.verbose,
            "debug": False,
            "batch_size": None,
            "device": None,
        }
        unknown = set(kwargs) - set(s)
        if unknown:
            raise ValueError(f"[sample] Unknown parameter(s): {', '.join(sorted(unknown))}")
        s.update(kwargs)
        state = "init" if s["use_initial_weights_for_sampling"] else s["state"]
        params = self._params_for(state)
        mm = {n: self._minmax(n) for n in self.order if self.structures[n].target == "continuous"}
        if getattr(self, "_sampler", None) is None or self._sampler.minmax != mm:
            self._sampler = DagSampler(self.structures, self.order, params, mm)
        sampler = self._sampler
        sampler.params = params
        with jax.default_device(self.device if s["device"] is None else _resolve_device(s["device"])):
            sampled, latents, draws = sampler.sample(
                s["number_of_samples"], do=do_interventions, latents_df=predefined_latent_samples_df,
                n_cf=s["number_of_counterfactual_samples"], seed=s["seed"], verbose=s["verbose"] or s["debug"],
            )
        self.last_counterfactual_draws = draws
        if s["save"] and self.experiment_dir:
            for n in self.order:
                d = os.path.join(self._node_dir(n), "sampling")
                if s["delete_all_previously_sampled"] and os.path.isdir(d):
                    shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
                if isinstance(sampled[n], dict):
                    np.savez(os.path.join(d, "hist.npz"), **sampled[n])
                else:
                    np.save(os.path.join(d, "sampled.npy"), sampled[n])
                np.save(os.path.join(d, "latents.npy"), latents[n])
                if n in draws:
                    np.save(os.path.join(d, "draws.npy"), draws[n])
        return sampled, latents

    def load_sampled_and_latents(self, EXPERIMENT_DIR=None, nodes=None):
        exp = EXPERIMENT_DIR or self.experiment_dir
        sampled, latents = {}, {}
        for n in nodes or self.order:
            d = os.path.join(exp, n, "sampling")
            sp, lp, hp = (os.path.join(d, f) for f in ("sampled.npy", "latents.npy", "hist.npz"))
            if os.path.exists(sp):
                sampled[n] = np.load(sp)
            elif os.path.exists(hp):
                with np.load(hp) as f:
                    sampled[n] = dict(f)
            else:
                print(f"[WARNING] Missing sampling files for node '{n}' — skipping.")
                continue
            if os.path.exists(lp):
                latents[n] = np.load(lp)
        return sampled, latents

    # ================================================================== plots

    def plot_loss_history(self, variable=None):
        import matplotlib.pyplot as plt

        hist = self.loss_history()
        nodes = [variable] if variable else list(hist)
        nodes = [n for n in nodes if hist.get(n, {}).get("train")]
        if not nodes:
            print("[WARNING] No valid histories found to plot.")
            return
        fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        for ax, last in zip(axs, (False, True)):
            for n in nodes:
                tr = np.array(hist[n]["train"], float)
                va = np.array([np.nan if v is None else v for v in (hist[n]["validation"] or [])], float)
                E = len(tr)
                s0 = (E - 1 if E < 5 else int(E * 0.9)) if last else 0
                ep = np.arange(s0 + 1, E + 1)
                ax.plot(ep, tr[s0:], "--", label=f"{n} - train")
                if len(va):
                    ax.plot(ep, va[s0:], label=f"{n} - val")
            ax.set_title("Training and Validation NLL - " + ("Last 10% of Epochs" if last else "Full History"))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("NLL")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_linear_shift_history(self, data_dict=None, node=None, ref_lines=None):
        self._plot_history(data_dict or self.linear_shift_history(), node, ref_lines, "Shift Value", "Shift Term History", "")

    def plot_simple_intercepts_history(self, data_dict=None, node=None, ref_lines=None):
        self._plot_history(data_dict or self.simple_intercept_history(), node, ref_lines, "Intercept Weight", "Simple Intercept Evolution", "theta_")

    @staticmethod
    def _plot_history(data_dict, node, ref_lines, ylabel, title, prefix):
        import matplotlib.pyplot as plt

        for n in [node] if node else list(data_dict):
            df = data_dict[n].copy()
            rows = {}
            for idx in df.index:
                vals = [_first_scalar(v) for v in df.loc[idx]]
                if any(isinstance(v, list) for v in df.loc[idx]) and isinstance(df.loc[idx].iloc[0], list) and len(df.loc[idx].iloc[0]) > 1:
                    arr = np.array([np.ravel(v) for v in df.loc[idx]], float)
                    for j in range(arr.shape[1]):
                        rows[f"{idx}[{j}]"] = arr[:, j]
                else:
                    rows[f"{prefix}{idx}"] = np.array(vals, float)
            epochs = [int(str(c).replace("epoch_", "")) for c in df.columns]
            order = np.argsort(epochs)
            plt.figure(figsize=(10, 6))
            for lab, v in rows.items():
                plt.plot(np.array(epochs)[order], v[order], lw=1.4, label=lab)
            if ref_lines and n in ref_lines:
                for v in ref_lines[n]:
                    plt.axhline(y=v, color="k", linestyle="--", lw=1.0)
                    plt.text(max(epochs), v, f"{n}: {v}", va="bottom", ha="right", fontsize=8)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{title} — Node: {n}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
            plt.tight_layout()
            plt.show()

    def plot_latents(self, df, variable=None, confidence=0.95, simulations=1000):
        import matplotlib.pyplot as plt
        from scipy.stats import logistic, probplot

        lat = self.get_latent(df)
        for n in [variable] if variable else self.order:
            if f"{n}_U" not in lat.columns:
                if self.verbose:
                    print(f"[INFO] {n} is ordinal: latents are intervals, skipping QQ plot.")
                continue
            sample = lat[f"{n}_U"].to_numpy()
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].hist(sample, bins=50, color="steelblue", alpha=0.7)
            axs[0].set_title(f"Latent Histogram ({n})")
            axs[0].set_xlabel("U")
            axs[0].set_ylabel("Frequency")
            probplot(sample, dist="logistic", plot=axs[1])
            _confidence_bands(axs[1], sample, logistic, confidence, simulations)
            axs[1].set_title(f"Latent QQ Plot ({n})")
            plt.suptitle(f"Latent Diagnostics for Node: {n}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_hdag(self, df, variables=None, plot_n_rows=1, **kwargs):
        import matplotlib.pyplot as plt

        for n in variables or self.order:
            st = self.structures[n]
            rows = df.iloc[: (1 if not st.parents else min(len(df), plot_n_rows))]
            if st.target == "continuous":
                k_min, k_max = self._minmax(n)
                d = k_max - k_min
                lo = kwargs.get("xmin_plot", k_min - 0.1 * d)
                hi = kwargs.get("xmax_plot", k_max + 0.1 * d)
                grid = np.linspace(lo, hi, 1000)
                H = self.transformation(n, rows, x=grid)
                for i in range(len(rows)):
                    if st.parents:
                        print(f"\n=== Sample {i+1}/{len(rows)} ===")
                        print(rows.iloc[i])
                    plt.figure(figsize=(8, 6))
                    for mask, col, lab in (
                        (grid < k_min, "red", "x < min_val"),
                        ((grid >= k_min) & (grid <= k_max), "blue", "min_val <= x <= max_val"),
                        (grid > k_max, "red", "x > max_val"),
                    ):
                        plt.plot(grid[mask], H[i][mask], color=col, label=lab)
                    plt.xlabel(f" ({n})")
                    plt.ylabel(f"h({n}|{' '.join(st.parents)})")
                    plt.title(f"transformation function {n} — sample {i+1}")
                    plt.grid(True)
                    plt.legend()
                    plt.show()
            else:
                C = self.cutpoints(n, rows)
                for i in range(len(rows)):
                    plt.figure(figsize=(6, 4))
                    k = np.arange(1, C.shape[1] + 1)
                    plt.plot(k, C[i], "o", color="C1", markersize=8)
                    plt.xticks(k)
                    plt.xlabel("Ordinal class index (k)")
                    plt.ylabel("Shifted cutpoint (θₖ - ηᵢ)")
                    plt.title(f"{n} — Sample {i+1}")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    plt.show()

    def plot_samples_vs_true(self, df, sampled=None, variable=None, bins=100, hist_true_color="blue",
                             hist_est_color="orange", figsize=(14, 5)):
        import matplotlib.pyplot as plt
        from statsmodels.graphics.gofplots import qqplot_2samples

        if sampled is None:
            sampled, _ = self.load_sampled_and_latents()
        for n in variable or self.order:
            if n not in sampled or n not in df.columns:
                continue
            sv = sampled[n]
            if isinstance(sv, dict):
                sv = np.repeat(sv["vals"].ravel(), sv["counts"].ravel())
            sv = np.asarray(sv)
            if sv.ndim == 2:
                print(f"[INFO] CAUTION! {n}: samples are class probabilities per observation; "
                      "the plot shows the most probable class per observation.")
                sv = np.argmax(sv, axis=1)
            sv = sv[np.isfinite(sv)]
            tv = df[n].dropna().to_numpy(dtype=float)
            tv = tv[np.isfinite(tv)]
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            if self.structures[n].target == "continuous":
                axs[0].hist(tv, bins=bins, density=True, alpha=0.6, color=hist_true_color, label=f"True {n}")
                axs[0].hist(sv, bins=bins, density=True, alpha=0.6, color=hist_est_color, label="Sampled")
                axs[0].set_xlabel("Value")
                axs[0].set_ylabel("Density")
                axs[0].set_title(f"Histogram overlay for {n}")
                axs[0].legend()
                axs[0].grid(True, ls="--", alpha=0.4)
                qqplot_2samples(tv, sv, line="45", ax=axs[1])
                axs[1].set_xlabel("True quantiles")
                axs[1].set_ylabel("Sampled quantiles")
                axs[1].set_title(f"QQ plot for {n}")
                axs[1].grid(True, ls="--", alpha=0.4)
            else:
                u = np.union1d(np.unique(tv), np.unique(sv))
                tc = np.array([(tv == v).sum() for v in u], float)
                sc = np.array([(sv == v).sum() for v in u], float)
                axs[0].bar(u - 0.2, tc / tc.sum(), width=0.4, color=hist_true_color, alpha=0.7, label="True")
                axs[0].bar(u + 0.2, sc / sc.sum(), width=0.4, color=hist_est_color, alpha=0.7, label="Sampled")
                axs[0].set_xticks(u)
                axs[0].set_xlabel("Ordinal Level")
                axs[0].set_ylabel("Relative Frequency")
                axs[0].set_title(f"Ordinal bar plot for {n}")
                axs[0].legend()
                axs[0].grid(True, ls="--", alpha=0.4)
                axs[1].axis("off")
            plt.tight_layout()
            plt.show()

    # ================================================================== summary

    def summary(self, verbose=False):
        print("\n" + "=" * 120)
        print(f"{'TRAM DAG MODEL SUMMARY':^120}")
        print("=" * 120)
        rows = []
        for n in self.order:
            h = self._read_histories(n)
            bt, bv = self.get_train_val_nll(n, "best") if h else (None, None)
            lt, lv = self.get_train_val_nll(n, "last") if h else (None, None)
            rows.append({"Node": n, "Best Train NLL": bt, "Best Val NLL": bv, "Last Train NLL": lt,
                         "Last Val NLL": lv, "Epochs": len(h["train"]) if h else 0})
        print("\n[1] TRAINING METRICS OVERVIEW")
        print("-" * 120)
        print(pd.DataFrame(rows).round(4).to_string(index=False, justify="center", col_space=14))
        print("-" * 120)
        print("\n[2] NODE-SPECIFIC DETAILS")
        print("-" * 120)
        for n in self.order:
            st = self.structures[n]
            print(f"\n{f'NODE: {n}':^120}")
            print(f"  {st.formula()}")
            for state in ("init", "last", "best"):
                print(f"\n  [{state.upper()} STATE]")
                th = self.get_thetas(n, state)
                if th is not None:
                    arr = np.round(np.array(th).ravel(), 4)
                    print(f"    Θ ({len(arr)}): {np.array2string(arr, max_line_width=110, separator=', ')}")
                else:
                    print("    Θ: complex intercept (input dependent)")
                for k, v in self.get_linear_shifts(n, state).items():
                    print(f"     {k}: {np.round(v, 4)}")
            if verbose:
                print("\n  [DETAILS]")
                nparams = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(self.params[n]))
                print(f"    intercept model: {st.intercept_model} ({st.n_int_features} -> {st.n_thetas})")
                for g in st.shifts:
                    print(f"    shift model: {g.model_name} {g.parents} ({g.n_features} -> 1)")
                print(f"    Parameter count: {nparams:,}")
            print("-" * 120)
        print("=" * 120 + "\n")


# ====================================================================== module helpers


def apply_warm_start(params, st, thetas_R, shift_coefs=None):
    """Place R estimates into the last intercept layer (first input channel) and linear shifts."""
    thetas_R = np.asarray(thetas_R, np.float64).ravel()
    if st.target == "continuous":
        tilde = inverse_transform_intercepts_continuous(thetas_R)
    else:
        tilde = inverse_transform_intercepts_ordinal(thetas_R)
    if tilde.size != st.n_thetas:
        raise ValueError(f"R returned {tilde.size} intercepts, expected {st.n_thetas}")
    params = jax.tree_util.tree_map(lambda x: x, params)
    last = dict(params["int"][-1])
    w = np.zeros(last["w"].shape, np.float32)
    w[0, :] = tilde
    last["w"] = jnp.asarray(w)
    if "b" in last:
        last["b"] = jnp.zeros_like(last["b"])
    params["int"] = list(params["int"][:-1]) + [last]
    if shift_coefs is not None and len(shift_coefs) == len(params["shifts"]):
        new = []
        for sp, c in zip(params["shifts"], shift_coefs):
            layer = dict(sp[-1])
            w = np.zeros(layer["w"].shape, np.float32)
            w[0, 0] = float(c)
            layer["w"] = jnp.asarray(w)
            new.append(list(sp[:-1]) + [layer])
        params["shifts"] = new
    return params


def _squeeze_list(a):
    a = np.asarray(a).ravel()
    return float(a[0]) if a.size == 1 else [float(v) for v in a]


def _first_scalar(x):
    while isinstance(x, list) and x:
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _confidence_bands(ax, sample, dist, confidence=0.95, simulations=1000):
    n = len(sample)
    if n == 0:
        return
    q = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    theo = dist.ppf(q)
    sims = np.sort(dist.rvs(size=(simulations, n)), axis=1)
    lo = np.percentile(sims, 100 * (1 - confidence) / 2, axis=0)
    hi = np.percentile(sims, 100 * (1 + confidence) / 2, axis=0)
    ax.clear()
    ax.plot(theo, np.sort(sample), "o", markersize=3, alpha=0.6, label="Empirical Q-Q")
    ax.plot(theo, theo, "b--", label="y = x")
    ax.fill_between(theo, lo, hi, color="gray", alpha=0.3, label=f"{int(confidence*100)}% CI")
    ax.legend()
