"""
State handling shared by the notebook widget (:mod:`tramdag.ui.widget`) and the
standalone browser app (:mod:`tramdag.ui.server`). The JavaScript editor only edits
a plain JSON description of the graph; validation, formulas and saving live here.
"""

import os
import time

from ..dag_spec import DATA_TYPES, TERMS, DagSpec, infer_node


def _options(cfg):
    from ..engine.nets import available_models

    paths = (cfg.conf_dict.get("PATHS") or {}) if cfg is not None else {}
    return {
        "data_types": DATA_TYPES,
        "terms": TERMS,
        "models": {"shift": available_models("shift"), "intercept": available_models("intercept")},
        "experiment_name": (cfg.conf_dict.get("experiment_name") if cfg is not None else None) or "",
        "experiment_dir": paths.get("EXPERIMENT_DIR") or "",
        "config_path": getattr(cfg, "CONF_DICT_PATH", None) or "",
    }


class EditorBackend:
    def __init__(self, cfg, df=None):
        self.cfg = cfg
        self.df = df
        spec = cfg.to_spec() if cfg.conf_dict.get("data_type") else DagSpec()
        if df is not None:
            for col in df.columns:
                if str(col) not in spec.names:
                    dt, lv = infer_node(df[col])
                    spec.add_node(str(col), dt, lv)
        if any(n.x is None or n.y is None for n in spec.nodes):
            spec.auto_layout()
        for k, v in {"epochs": 100, "learning_rate": 0.01, "batch_size": 1000}.items():
            spec.training.setdefault(k, v)
        self.spec = spec
        self.saved_at = None

    def state(self):
        return {"spec": self.spec.to_dict(), "report": self.report(), "options": _options(self.cfg)}

    def report(self, dirty=False):
        errors, warnings = self.spec.validate()
        formulas = {}
        for n in self.spec.names:
            try:
                formulas[n] = self.spec.formula(n)
            except Exception as ex:  # noqa: BLE001 - shown in the UI
                formulas[n] = f"h({n}) = ?  ({ex})"
        return {"errors": errors, "warnings": warnings, "formulas": formulas, "saved_at": self.saved_at, "dirty": dirty}

    def update_spec(self, spec_dict):
        try:
            self.spec = DagSpec.from_dict(spec_dict)
        except Exception as ex:  # noqa: BLE001
            rep = self.report(dirty=True)
            rep["errors"] = [f"could not read the graph: {ex}"] + rep["errors"]
            return rep
        return self.report(dirty=True)

    def message(self, msg):
        """Handle a command from the editor; returns the traits/keys that changed."""
        kind = msg.get("type")
        if kind == "save":
            errors, _ = self.spec.validate()
            if errors:
                rep = self.report(dirty=True)
                rep["errors"] = ["not saved - fix the errors first"] + rep["errors"]
                return {"report": rep}
            self.cfg.apply_spec(self.spec, write=bool(self.cfg.CONF_DICT_PATH))
            self.saved_at = time.strftime("%H:%M:%S")
            rep = self.report(dirty=False)
            rep["saved_path"] = self.cfg.CONF_DICT_PATH or ""
            return {"report": rep}
        if kind == "auto_layout":
            self.spec.auto_layout(width=float(msg.get("width") or 640), height=float(msg.get("height") or 420))
            return {"spec": self.spec.to_dict(), "report": self.report(dirty=True)}
        if kind == "edges_text":
            from ..dag_spec import apply_edge_text

            try:
                apply_edge_text(self.spec, msg.get("text", ""), create_nodes=True)
            except Exception as ex:  # noqa: BLE001
                rep = self.report(dirty=True)
                rep["errors"] = [f"edge text: {ex}"] + rep["errors"]
                return {"report": rep}
            for n in self.spec.nodes:
                if n.x is None:
                    n.x, n.y = 60.0, 60.0
            return {"spec": self.spec.to_dict(), "report": self.report(dirty=True)}
        if kind == "reload":
            if self.cfg.CONF_DICT_PATH and os.path.exists(self.cfg.CONF_DICT_PATH):
                self.cfg.update()
            self.__init__(self.cfg, self.df)
            return {"spec": self.spec.to_dict(), "report": self.report(dirty=False)}
        return {}
