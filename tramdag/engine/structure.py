"""
Translate the per-node entries of ``configuration.json`` into the structure of a
node model (which inputs feed the intercept network and which feed each shift
network) and encode pandas data into the corresponding design matrices.

The ordering and grouping rules are identical to the original implementation
(``ordered_parents`` / ``group_by_base`` / ``GenericDataset``) so parameters are
interchangeable with the PyTorch version:

* terms are sorted ``ci < ciXX < cs < csXX < ls``,
* all ``ci*`` terms feed one intercept network (otherwise a simple intercept with a
  constant input of 1 is used),
* every plain ``cs`` / ``ls`` term gets its own shift network, numbered terms such as
  ``cs11, cs12`` share one network (interaction ``cs(x1, x2)``),
* parents whose data type contains ``Xn`` are one-hot encoded, everything else is fed
  as a single float column.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .nets import get_model_spec

DEFAULT_N_THETAS_CONTINUOUS = 20
_ORDER = ["ci", "ciXX", "cs", "csXX", "ls"]


def is_continuous_target(data_type: str) -> bool:
    dt = data_type.lower()
    return "yc" in dt or "continous" in dt or "continuous" in dt


def is_ordinal_target(data_type: str) -> bool:
    dt = data_type.lower()
    return "yo" in dt and "ordinal" in dt


def is_onehot_predictor(data_type: str) -> bool:
    return "xn" in data_type.lower()


def _sort_key(term):
    base = re.match(r"[a-zA-Z]+", term)
    digits = re.findall(r"\d+", term)
    base = base.group(0) if base else ""
    d = int(digits[0]) if digits else -1
    if base in ("ci", "cs") and d != -1:
        return (_ORDER.index(base + "XX"), d)
    if base in _ORDER:
        return (_ORDER.index(base), -1)
    return (len(_ORDER), d)


def ordered_terms(node_conf):
    """Parents sorted like the original ``ordered_parents``: returns OrderedDict parent -> (term, model)."""
    terms = node_conf.get("transformation_terms_in_h()", {}) or {}
    models = node_conf.get("transformation_term_nn_models_in_h()", {}) or {}
    keys = sorted(terms.keys(), key=lambda k: _sort_key(terms[k]))
    return OrderedDict((k, (terms[k], models.get(k))) for k in keys)


def _group_key(parent, term, prefixes):
    for p in prefixes:
        if term.startswith(p):
            if term == p:
                return parent
            if len(term) > len(p) and term[len(p)].isdigit():
                return term[: len(p) + 1]
            return term
    return None


@dataclass
class TermGroup:
    """One network inside h(): its input parents and model class."""

    parents: List[str]
    terms: List[str]
    model_name: str
    n_features: int
    kind: str  # "intercept" | "shift"

    @property
    def is_linear(self):
        from .nets import base_model_name

        return base_model_name(self.model_name) == "LinearShift"

    @property
    def label(self):
        base = "ls" if self.is_linear else ("ci" if self.kind == "intercept" else "cs")
        return f"{base}({', '.join(self.parents)})"


@dataclass
class NodeStructure:
    name: str
    data_type: str
    target: str  # "continuous" | "ordinal"
    n_thetas: int
    levels: Optional[int]
    parents: List[str]
    intercept: Optional[TermGroup]  # None -> simple intercept
    shifts: List[TermGroup] = field(default_factory=list)
    parent_types: dict = field(default_factory=dict)
    parent_levels: dict = field(default_factory=dict)

    @property
    def intercept_model(self):
        return self.intercept.model_name if self.intercept else "SimpleIntercept"

    @property
    def n_int_features(self):
        return self.intercept.n_features if self.intercept else 1

    @property
    def has_simple_intercept(self):
        return self.intercept is None

    def formula(self):
        """Human readable form of the transformation function, e.g. ``h(x3 | x1, x2) = SI + ls(x1) + cs(x2)``."""
        lhs = f"h({self.name} | {', '.join(self.parents)})" if self.parents else f"h({self.name})"
        rhs = [f"ci({', '.join(self.intercept.parents)})" if self.intercept else "SI"]
        rhs += [g.label for g in self.shifts]
        return f"{lhs} = {' + '.join(rhs)}"

    # ------------------------------------------------------------------ encoding

    def _encode_parent(self, parent, values):
        values = np.asarray(values)
        if is_onehot_predictor(self.parent_types[parent]):
            k = self.parent_levels[parent]
            idx = values.astype(np.int64)
            if idx.size and (idx.min() < 0 or idx.max() >= k):
                raise ValueError(f"parent '{parent}' of '{self.name}' has values outside 0..{k-1}")
            out = np.zeros((len(idx), k), np.float32)
            out[np.arange(len(idx)), idx] = 1.0
            return out
        return values.astype(np.float32).reshape(-1, 1)

    def encode_inputs(self, columns: dict, n: int):
        """``columns``: mapping parent -> 1d array of length ``n``. Returns ``{"int": ..., "shifts": [...]}``."""
        if self.intercept is None:
            int_x = np.ones((n, 1), np.float32)
        else:
            int_x = np.concatenate([self._encode_parent(p, columns[p]) for p in self.intercept.parents], axis=1)
        shifts = [
            np.concatenate([self._encode_parent(p, columns[p]) for p in g.parents], axis=1) for g in self.shifts
        ]
        return {"int": int_x, "shifts": shifts}

    def encode_target(self, values):
        values = np.asarray(values)
        if self.target == "continuous":
            return values.astype(np.float32)
        y = values.astype(np.int64)
        if y.size and (y.min() < 0 or y.max() >= self.levels):
            raise ValueError(f"ordinal target '{self.name}' must take values in 0..{self.levels-1}")
        return y.astype(np.int32)

    def encode_frame(self, df: pd.DataFrame, with_target=True):
        missing = [p for p in self.parents if p not in df.columns]
        if missing:
            raise KeyError(f"columns {missing} (parents of '{self.name}') missing from data")
        cols = {p: column_values(df[p]) for p in self.parents}
        out = self.encode_inputs(cols, len(df))
        if with_target:
            if self.name not in df.columns:
                raise KeyError(f"target column '{self.name}' missing from data")
            out["y"] = self.encode_target(column_values(df[self.name]))
        return out


def column_values(series):
    """Numeric values of a column; ordered categoricals become their codes 0..K-1."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    if series.dtype == bool:
        return series.to_numpy().astype(np.int64)
    return series.to_numpy()


def build_node_structure(node: str, nodes_conf: dict) -> NodeStructure:
    conf = nodes_conf[node]
    dt = conf["data_type"]
    if is_continuous_target(dt):
        target, n_thetas, levels = "continuous", DEFAULT_N_THETAS_CONTINUOUS, conf.get("levels")
    elif is_ordinal_target(dt):
        levels = conf.get("levels")
        if levels is None:
            raise ValueError(
                f"Ordinal node '{node}' has no 'levels' in the configuration. "
                "Use cfg.compute_levels(train_df) or TramDagConfig.from_dataframe(...)."
            )
        target, n_thetas = "ordinal", int(levels) - 1
    else:
        raise NotImplementedError(f"data type '{dt}' of node '{node}' is not supported by the JAX engine")

    parent_types = dict(conf.get("parents_datatype", {}) or {})
    parent_levels = {}
    for p, pdt in parent_types.items():
        if is_onehot_predictor(pdt):
            lv = nodes_conf.get(p, {}).get("levels")
            if lv is None:
                raise ValueError(f"Parent '{p}' of '{node}' is one-hot encoded (Xn) but has no 'levels'.")
            parent_levels[p] = int(lv)

    def n_feat(parents):
        return sum(parent_levels.get(p, 1) for p in parents)

    ordered = ordered_terms(conf)
    int_groups, shift_groups = OrderedDict(), OrderedDict()
    for parent, (term, model) in ordered.items():
        if term.startswith(("ci", "si")):
            key = _group_key(parent, term, ("ci", "si"))
            int_groups.setdefault(key, []).append((parent, term, model))
        else:
            key = _group_key(parent, term, ("cs", "ls"))
            shift_groups.setdefault(key, []).append((parent, term, model))

    if len(int_groups) > 1:
        raise ValueError(f"Node '{node}': multiple complex intercept groups detected; only one is supported.")
    intercept = None
    for feats in int_groups.values():
        parents = [f[0] for f in feats]
        model = feats[0][2] or "ComplexInterceptDefaultTabular"
        intercept = TermGroup(parents, [f[1] for f in feats], model, n_feat(parents), "intercept")

    shifts = []
    for feats in shift_groups.values():
        parents = [f[0] for f in feats]
        term = feats[0][1]
        default = "LinearShift" if term.startswith("ls") else "ComplexShiftDefaultTabular"
        model = feats[0][2] or default
        shifts.append(TermGroup(parents, [f[1] for f in feats], model, n_feat(parents), "shift"))

    # fail early on unknown model names
    get_model_spec(intercept.model_name if intercept else "SimpleIntercept")
    for g in shifts:
        get_model_spec(g.model_name)

    return NodeStructure(
        name=node,
        data_type=dt,
        target=target,
        n_thetas=n_thetas,
        levels=int(levels) if levels is not None else None,
        parents=list(conf.get("parents", []) or []),
        intercept=intercept,
        shifts=shifts,
        parent_types=parent_types,
        parent_levels=parent_levels,
    )


def topological_order(nodes_conf: dict):
    """Nodes in an order where every parent precedes its children (stable w.r.t. the config order)."""
    order, done = [], set()
    pending = list(nodes_conf.keys())
    while pending:
        progressed = False
        for n in list(pending):
            if all(p in done for p in nodes_conf[n].get("parents", []) or []):
                order.append(n)
                done.add(n)
                pending.remove(n)
                progressed = True
        if not progressed:
            raise ValueError(f"The graph has a cycle among {pending}")
    return order
