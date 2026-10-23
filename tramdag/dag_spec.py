"""
:class:`DagSpec` - a readable description of a TRAM-DAG that converts to and from the
``configuration.json`` format.

Instead of filling an upper-triangular matrix with codes such as ``ls``, ``cs11`` or
``ci``, a graph is described by nodes (name, data type, number of levels) and edges
(source, target, term, optional interaction group, optional network)::

    spec = DagSpec.from_edges(
        "x1 -> x2 ; x1 -> x3 ; x2 -> x3 [cs]",
        data_types={"x1": "continuous", "x2": "continuous", "x3": "ordinal"},
        levels={"x3": 4},
    )
    spec.validate()            # -> (errors, warnings)
    conf = spec.to_conf_dict("my_experiment", "runs/my_experiment")

Edge mini-language (statements separated by ``;`` or new lines, ``#`` starts a comment)::

    x1 -> x2                   linear shift (default)
    x1 -> x3 [cs]              complex shift  (also: ls, ci)
    x1, x2 -> x4               two separate edges
    x1 -> x2 -> x3             chain
    (x1, x2) -> x4 [cs]        interaction: one network cs(x1, x2)
    (x1, x2) -> x4 [ci]        complex intercept depending on x1 and x2
    x2 -> x3 [cs, model=ComplexShiftCustomTabular]
    x2 -> x3 [cs, model=MLP(32,32,tanh)]
    x3 : ordinal(4)            node declaration (type and levels)
    x5 : continuous
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------- data types

DATA_TYPES = {
    "continous": "continuous",
    "ordinal_Xn_Yo": "ordinal (one-hot as input, ordinal as target)",
    "ordinal_Xc_Yo": "ordinal (numeric as input, ordinal as target)",
    "ordinal_Xn_Yc": "ordinal (one-hot as input, continuous as target)",
    "ordinal_Xc_Yc": "ordinal treated as continuous",
}

_ALIASES = {
    "continuous": "continous",
    "continous": "continous",
    "cont": "continous",
    "c": "continous",
    "float": "continous",
    "ordinal": "ordinal_Xn_Yo",
    "ord": "ordinal_Xn_Yo",
    "binary": "ordinal_Xn_Yo",
    "categorical": "ordinal_Xn_Yo",
    "o": "ordinal_Xn_Yo",
}

TERMS = {
    "ls": "linear shift  beta * x  (interpretable)",
    "cs": "complex shift  f(x)  (neural network)",
    "ci": "complex intercept  h depends on x in a non-additive way",
}

DEFAULT_MODELS = {"ls": "LinearShift", "cs": "ComplexShiftDefaultTabular", "ci": "ComplexInterceptDefaultTabular"}


def normalize_data_type(dt: str) -> str:
    if dt is None:
        raise ValueError("data type missing")
    key = str(dt).strip()
    if key in DATA_TYPES:
        return key
    low = key.lower()
    for k in DATA_TYPES:
        if k.lower() == low:
            return k
    if low in _ALIASES:
        return _ALIASES[low]
    raise ValueError(f"unknown data type '{dt}'. Use one of {list(DATA_TYPES)} or 'continuous' / 'ordinal'.")


def is_ordinal(dt: str) -> bool:
    return "ordinal" in dt.lower()


def needs_levels(dt: str) -> bool:
    """Levels are needed for ordinal targets (Yo) and one-hot inputs (Xn)."""
    return "yo" in dt.lower() or "xn" in dt.lower()


# ---------------------------------------------------------------------------- spec


@dataclass
class Node:
    name: str
    data_type: str = "continous"
    levels: Optional[int] = None
    x: Optional[float] = None  # position in the visual editor
    y: Optional[float] = None


@dataclass
class Edge:
    source: str
    target: str
    term: str = "ls"
    group: Optional[int] = None  # interaction group (cs / ci only)
    model: Optional[str] = None  # network name, None -> default for the term


@dataclass
class DagSpec:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    training: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ editing

    def node(self, name) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"unknown node '{name}'")

    @property
    def names(self):
        return [n.name for n in self.nodes]

    def add_node(self, name, data_type="continuous", levels=None, x=None, y=None):
        if name in self.names:
            raise ValueError(f"node '{name}' already exists")
        self.nodes.append(Node(name, normalize_data_type(data_type), levels, x, y))
        return self

    def remove_node(self, name):
        self.nodes = [n for n in self.nodes if n.name != name]
        self.edges = [e for e in self.edges if name not in (e.source, e.target)]
        return self

    def set_node(self, name, data_type=None, levels=None):
        n = self.node(name)
        if data_type is not None:
            n.data_type = normalize_data_type(data_type)
        if levels is not None:
            n.levels = int(levels)
        return self

    def find_edge(self, source, target):
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def add_edge(self, source, target, term="ls", group=None, model=None):
        for n in (source, target):
            if n not in self.names:
                raise KeyError(f"unknown node '{n}'")
        e = self.find_edge(source, target)
        if e is None:
            e = Edge(source, target)
            self.edges.append(e)
        e.term, e.group, e.model = term, group, model
        return self

    def remove_edge(self, source, target):
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        return self

    def parents(self, node):
        return [e.source for e in self.edges if e.target == node]

    def children(self, node):
        return [e.target for e in self.edges if e.source == node]

    def incoming(self, node):
        return [e for e in self.edges if e.target == node]

    # ------------------------------------------------------------------ graph

    def topological_order(self):
        """Causal order; ties keep the order of ``nodes`` (stable)."""
        order, done = [], set()
        pending = self.names
        while pending:
            ready = [n for n in pending if all(p in done for p in self.parents(n))]
            if not ready:
                raise ValueError(f"the graph contains a cycle through {self.find_cycle()}")
            order.append(ready[0])
            done.add(ready[0])
            pending.remove(ready[0])
        return order

    def find_cycle(self):
        color = {n: 0 for n in self.names}
        stack = []

        def dfs(u):
            color[u] = 1
            stack.append(u)
            for v in self.children(u):
                if v not in color:
                    continue
                if color[v] == 1:
                    return stack[stack.index(v):] + [v]
                if color[v] == 0:
                    c = dfs(v)
                    if c:
                        return c
            stack.pop()
            color[u] = 2
            return None

        for n in self.names:
            if color[n] == 0:
                c = dfs(n)
                if c:
                    return c
        return None

    # ------------------------------------------------------------------ codes

    def term_codes(self, target):
        """``{parent: code}`` for the edges into ``target`` using the original coding (ls, cs, cs11, ci12 ...)."""
        order = {n: i for i, n in enumerate(self._safe_order())}
        edges = sorted(self.incoming(target), key=lambda e: order.get(e.source, 0))
        groups = {}
        for e in edges:
            if e.group is not None and e.term in ("cs", "ci"):
                groups.setdefault((e.term, e.group), []).append(e)
        # renumber groups 1.. per term in order of first appearance
        renum, counters = {}, {"cs": 0, "ci": 0}
        for (term, g) in groups:
            counters[term] += 1
            renum[(term, g)] = counters[term]
        codes = {}
        for e in edges:
            if e.group is not None and e.term in ("cs", "ci"):
                members = groups[(e.term, e.group)]
                codes[e.source] = f"{e.term}{renum[(e.term, e.group)]}{members.index(e) + 1}"
            else:
                codes[e.source] = e.term
        return codes

    def model_names(self, target):
        """``{parent: model class name}`` incl. the interaction suffix, as in the original config."""
        codes = self.term_codes(target)
        group_model = {}
        for e in self.incoming(target):
            if e.group is not None and e.term in ("cs", "ci"):
                group_model.setdefault((e.term, e.group), e.model or DEFAULT_MODELS[e.term])
        out = {}
        for e in self.incoming(target):
            code = codes[e.source]
            if e.group is not None and e.term in ("cs", "ci"):
                out[e.source] = group_model[(e.term, e.group)] + code[2:]
            else:
                out[e.source] = e.model or DEFAULT_MODELS[e.term]
        return out

    def _safe_order(self):
        try:
            return self.topological_order()
        except ValueError:
            return self.names

    # ------------------------------------------------------------------ validation

    def validate(self, check_models=True):
        """Returns ``(errors, warnings)`` as lists of strings (errors block training)."""
        errors, warnings = [], []
        seen = set()
        for n in self.nodes:
            if not n.name or not str(n.name).strip():
                errors.append("a node has an empty name")
            if n.name in seen:
                errors.append(f"duplicate node name '{n.name}'")
            seen.add(n.name)
            if re.search(r"_[Uu]$", n.name):
                errors.append(f"'{n.name}': names ending in '_U' are reserved for latent variables")
            if not re.match(r"^[A-Za-z_.][A-Za-z0-9_.]*$", n.name):
                warnings.append(f"'{n.name}': use letters, digits and '_' only (needed for the R warm start)")
            try:
                normalize_data_type(n.data_type)
            except ValueError as ex:
                errors.append(f"'{n.name}': {ex}")
                continue
            if needs_levels(n.data_type):
                if n.levels is None:
                    errors.append(f"'{n.name}' is ordinal but the number of levels is not set")
                elif int(n.levels) < 2:
                    errors.append(f"'{n.name}' needs at least 2 levels")

        names = set(self.names)
        pairs = set()
        for e in self.edges:
            tag = f"{e.source} -> {e.target}"
            if e.source not in names or e.target not in names:
                errors.append(f"{tag}: unknown node")
                continue
            if e.source == e.target:
                errors.append(f"{tag}: self loops are not allowed")
            if (e.source, e.target) in pairs:
                errors.append(f"{tag}: duplicate edge")
            pairs.add((e.source, e.target))
            if e.term not in TERMS:
                errors.append(f"{tag}: unknown term '{e.term}' (use ls, cs or ci)")
            if e.term == "ls" and e.group is not None:
                errors.append(f"{tag}: linear shifts cannot be part of an interaction group")
            if e.group is not None and not (1 <= int(e.group) <= 9):
                errors.append(f"{tag}: interaction group must be between 1 and 9")

        cycle = self.find_cycle()
        if cycle:
            errors.append("the graph has a cycle: " + " -> ".join(cycle))

        for n in self.names:
            inc = self.incoming(n)
            ci_plain = [e for e in inc if e.term == "ci" and e.group is None]
            ci_groups = {e.group for e in inc if e.term == "ci" and e.group is not None}
            if len(ci_plain) > 1:
                errors.append(f"'{n}': only one plain complex intercept is allowed; "
                              f"put {', '.join(e.source for e in ci_plain)} into one interaction group instead")
            if (1 if ci_plain else 0) + len(ci_groups) > 1:
                errors.append(f"'{n}': only one complex intercept (ci or one ci group) is supported per node")
            for term in ("cs", "ci"):
                groups = {}
                for e in inc:
                    if e.term == term and e.group is not None:
                        groups.setdefault(e.group, []).append(e)
                for g, members in groups.items():
                    if len(members) < 2:
                        errors.append(f"'{n}': {term} group {g} has only one input ({members[0].source}); "
                                      "remove the group or add a second input")
                    if len(members) > 9:
                        errors.append(f"'{n}': {term} group {g} has more than 9 inputs")
                    models = {m.model for m in members if m.model}
                    if len(models) > 1:
                        warnings.append(f"'{n}': {term} group {g} uses different networks {sorted(models)}; "
                                        f"the first one is used for the whole group")
            if len([g for g in {(e.term, e.group) for e in inc if e.group is not None}]) > 9:
                errors.append(f"'{n}': more than 9 interaction groups")

        if check_models:
            from .engine.nets import get_model_spec

            for e in self.edges:
                if e.model:
                    try:
                        spec = get_model_spec(e.model)
                    except Exception as ex:  # noqa: BLE001
                        errors.append(f"{e.source} -> {e.target}: {ex}")
                        continue
                    role = getattr(spec, "role", "any")
                    want = "intercept" if e.term == "ci" else "shift"
                    if e.term in ("cs", "ci") and role not in (want, "any"):
                        warnings.append(f"{e.source} -> {e.target}: '{e.model}' is meant as a {role} network")
                    if e.term == "ls" and e.model != "LinearShift":
                        warnings.append(f"{e.source} -> {e.target}: a linear shift should use 'LinearShift'")

        for n in self.names:
            if not self.parents(n) and not self.children(n) and len(self.nodes) > 1:
                warnings.append(f"'{n}' is not connected to any other node")
        return errors, warnings

    def check(self):
        errors, warnings = self.validate()
        if errors:
            raise ValueError("Invalid DAG specification:\n  - " + "\n  - ".join(errors))
        return warnings

    # ------------------------------------------------------------------ formulas

    def formula(self, node):
        inc = self.incoming(node)
        if not inc:
            return f"h({node}) = SI"
        codes = self.term_codes(node)
        parents = [e.source for e in sorted(inc, key=lambda e: self._safe_order().index(e.source) if e.source in self.names else 0)]
        ci = [p for p in parents if codes[p].startswith("ci")]
        rhs = [f"ci({', '.join(ci)})" if ci else "SI"]
        done = set()
        for p in parents:
            c = codes[p]
            if c.startswith("ci") or p in done:
                continue
            if len(c) > 2:
                members = [q for q in parents if codes[q][:3] == c[:3]]
                done.update(members)
                rhs.append(f"cs({', '.join(members)})")
            else:
                rhs.append(f"{c}({p})")
        return f"h({node} | {', '.join(parents)}) = {' + '.join(rhs)}"

    def formulas(self):
        return {n: self.formula(n) for n in self.names}

    # ------------------------------------------------------------------ conversion

    def to_conf_dict(self, experiment_name=None, experiment_dir=None, data_path=None, seed=None):
        """Configuration dict in the format of the original package (``configuration.json``)."""
        from .utils.configuration import create_node_dict

        self.check()
        order = self.topological_order()
        idx = {n: i for i, n in enumerate(order)}
        data_type = {n: normalize_data_type(self.node(n).data_type) for n in order}
        k = len(order)
        adj = np.full((k, k), "0", dtype=object)
        nn = np.full((k, k), "0", dtype=object)
        for t in order:
            codes, models = self.term_codes(t), self.model_names(t)
            for p, code in codes.items():
                adj[idx[p], idx[t]] = code
                nn[idx[p], idx[t]] = models[p]
        levels = {n: int(self.node(n).levels) for n in order if self.node(n).levels is not None and is_ordinal(data_type[n])}
        nodes = create_node_dict(adj, nn, data_type, None, None, levels_dict=levels)
        for n in order:  # levels for every ordinal node (create_node_dict only handles the dict it is given)
            if n in levels:
                nodes[n]["levels"] = levels[n]
        conf = {
            "date_of_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "PATHS": {"DATA_PATH": data_path or experiment_dir, "LOG_DIR": None, "EXPERIMENT_DIR": experiment_dir},
            "data_type": data_type,
            "adj_matrix": adj.tolist(),
            "model_names": nn.tolist(),
            "seed": seed,
            "nodes": nodes,
        }
        if self.training:
            conf["training"] = dict(self.training)
        layout = {n.name: [n.x, n.y] for n in self.nodes if n.x is not None and n.y is not None}
        if layout:
            conf["ui_layout"] = layout
        return conf

    @classmethod
    def from_conf_dict(cls, conf):
        """Read the graph back from a ``configuration.json`` dict (also files made by the old widgets)."""
        dt = conf.get("data_type") or {}
        nodes_conf = conf.get("nodes") or {}
        layout = conf.get("ui_layout") or {}
        spec = cls(training=dict(conf.get("training") or {}))
        names = list(dt) or list(nodes_conf)
        for n in names:
            pos = layout.get(n) or [None, None]
            spec.nodes.append(Node(n, dt.get(n) or nodes_conf.get(n, {}).get("data_type", "continous"),
                                   nodes_conf.get(n, {}).get("levels"), pos[0], pos[1]))
        adj = conf.get("adj_matrix")
        models = conf.get("model_names")
        if adj is not None:
            for i, src in enumerate(names):
                for j, tgt in enumerate(names):
                    code = str(adj[i][j])
                    if code == "0":
                        continue
                    term, group = _parse_code(code)
                    model = None
                    if models is not None and str(models[i][j]) != "0":
                        from .engine.nets import base_model_name

                        model = base_model_name(str(models[i][j]))
                        if model == DEFAULT_MODELS.get(term):
                            model = None
                    spec.edges.append(Edge(src, tgt, term, group, model))
        else:
            for t, nc in nodes_conf.items():
                for p, code in (nc.get("transformation_terms_in_h()") or {}).items():
                    term, group = _parse_code(code)
                    spec.edges.append(Edge(p, t, term, group, None))
        return spec

    def to_dict(self):
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [asdict(e) for e in self.edges],
                "training": dict(self.training)}

    @classmethod
    def from_dict(cls, d):
        spec = cls(training=dict(d.get("training") or {}))
        for n in d.get("nodes", []):
            lv = n.get("levels")
            spec.nodes.append(Node(n["name"], n.get("data_type", "continous"),
                                   int(lv) if lv not in (None, "") else None, n.get("x"), n.get("y")))
        for e in d.get("edges", []):
            g = e.get("group")
            spec.edges.append(Edge(e["source"], e["target"], e.get("term", "ls"),
                                   int(g) if g not in (None, "", 0, "0") else None, e.get("model") or None))
        return spec

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_edges(cls, text, data_types=None, levels=None, nodes=None):
        """Build a spec from the edge mini-language (see module docstring)."""
        spec = cls()
        for n in nodes or []:
            spec.add_node(n)
        apply_edge_text(spec, text, create_nodes=True)
        for n, dt in (data_types or {}).items():
            if n not in spec.names:
                spec.add_node(n)
            spec.set_node(n, data_type=dt)
        for n, lv in (levels or {}).items():
            spec.set_node(n, levels=lv)
        return spec

    def auto_layout(self, width=640, height=420):
        """Layered layout: x by causal depth, y spread within a layer."""
        depth = {}
        for n in self._safe_order():
            ps = self.parents(n)
            depth[n] = 1 + max((depth.get(p, 0) for p in ps), default=-1)
        layers = {}
        for n, d in depth.items():
            layers.setdefault(d, []).append(n)
        nlay = max(layers) + 1 if layers else 1
        widest = max((len(m) for m in layers.values()), default=1)
        # layers sit on an arc (middle layers lower) so that an edge skipping layers does not run
        # through the nodes in between, which happens when all layers are on one straight line
        amp = (height - 120) * (0.45 if widest == 1 else 0.25) if nlay > 2 else 0.0
        band = (height - 120) - amp
        for d, members in layers.items():
            r = d / max(1, nlay - 1)
            arc = amp * (1 - (2 * r - 1) ** 2)
            for i, n in enumerate(members):
                node = self.node(n)
                node.x = 80 + (width - 160) * r if nlay > 1 else width / 2
                node.y = 60 + arc + band * ((i + 0.5) / len(members))
        return self


def _parse_code(code):
    m = re.match(r"^(ls|cs|ci|si)(\d)?(\d)?$", str(code))
    if not m:
        raise ValueError(f"unknown term code '{code}'")
    term = m.group(1)
    group = int(m.group(2)) if m.group(2) else None
    return term, group


# ---------------------------------------------------------------------------- edge language


def _split_top(s, sep=","):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def _parse_side(s):
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return [x.strip() for x in s[1:-1].split(",") if x.strip()], True
    return [x.strip() for x in s.split(",") if x.strip()], False


def _term_for_model(model):
    """Default term when only a network is given: LinearShift -> ls, intercept nets -> ci, else cs."""
    if not model:
        return "ls"
    from .engine.nets import base_model_name, get_model_spec

    if base_model_name(model) == "LinearShift":
        return "ls"
    try:
        role = getattr(get_model_spec(model), "role", "any")
    except Exception:  # noqa: BLE001 - reported by validate()
        role = "any"
    return "ci" if role == "intercept" else "cs"


def apply_edge_text(spec, text, create_nodes=True):
    """Apply edge / node statements to ``spec`` (in place)."""
    if not text:
        return spec
    stmts = []
    for line in str(text).splitlines():
        line = line.split("#", 1)[0]
        stmts += [s.strip() for s in _split_top(line, ";")]
    for st in stmts:
        if not st:
            continue
        if "->" not in st:
            m = re.match(r"^([^:]+):\s*([A-Za-z_]+)\s*(?:\(\s*(\d+)\s*\))?$", st)
            if not m:
                raise ValueError(f"cannot parse '{st}' (expected 'a -> b [term]' or 'name : type(levels)')")
            name, dt, lv = m.group(1).strip(), m.group(2), m.group(3)
            if name not in spec.names:
                spec.add_node(name)
            spec.set_node(name, data_type=dt, levels=int(lv) if lv else None)
            continue
        opts = {}
        m = re.match(r"^(.*?)\s*\[(.*)\]\s*$", st)
        body = st
        if m:
            body = m.group(1)
            for o in _split_top(m.group(2)):
                if "=" in o:
                    k, v = o.split("=", 1)
                    opts[k.strip().lower()] = v.strip()
                elif o.lower() in TERMS:
                    opts["term"] = o.lower()
                elif o.upper().startswith("MLP(") or o[:1].isupper():
                    opts["model"] = o
                else:
                    raise ValueError(f"unknown option '{o}' in '{st}'")
        model = opts.get("model")
        term = opts.get("term") or _term_for_model(model)
        parts = [p for p in re.split(r"\s*->\s*", body)]
        if len(parts) < 2 or any(not p.strip() for p in parts):
            raise ValueError(f"cannot parse edge statement '{st}'")
        for a, b in zip(parts[:-1], parts[1:]):
            srcs, grouped = _parse_side(a)
            tgts, tgrp = _parse_side(b)
            if tgrp and len(tgts) > 1:
                raise ValueError(f"'{st}': parentheses mark interacting inputs and are only allowed on the left side")
            for n in srcs + tgts:
                if n not in spec.names:
                    if not create_nodes:
                        raise KeyError(f"unknown node '{n}'")
                    spec.add_node(n)
            for t in tgts:
                group = None
                if grouped and len(srcs) > 1:
                    if term == "ls":
                        raise ValueError(f"'{st}': interactions need a complex term, write [cs] or [ci]")
                    used = {e.group for e in spec.incoming(t) if e.term == term and e.group is not None}
                    group = next(g for g in range(1, 10) if g not in used)
                for s in srcs:
                    spec.add_edge(s, t, term=term, group=group, model=model)
    return spec


# ---------------------------------------------------------------------------- data type inference


def infer_node(series, max_levels=10):
    """Guess ``(data_type, levels)`` for a pandas Series."""
    import pandas as pd

    s = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "ordinal_Xn_Yo", len(series.cat.categories)
    if s.dtype == bool:
        return "ordinal_Xn_Yo", 2
    if not np.issubdtype(s.dtype, np.number):
        raise ValueError(f"column '{series.name}' is not numeric; encode it as integers 0..K-1")
    vals = s.to_numpy()
    if len(vals) and np.all(np.equal(np.mod(vals, 1), 0)):
        u = np.unique(vals)
        if u.min() >= 0 and len(u) <= max_levels and u.max() < max_levels:
            return "ordinal_Xn_Yo", int(u.max()) + 1
    return "continous", None
