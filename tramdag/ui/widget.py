"""Notebook widget: ``cfg.edit()`` or ``DagEditor(cfg, df)``.

Requires ``anywidget`` (``pip install anywidget``); works in JupyterLab, Notebook 7,
VS Code and Google Colab.
"""

import pathlib

try:
    import anywidget
    import traitlets
except ImportError as e:  # pragma: no cover
    raise ImportError("The visual editor needs anywidget: pip install anywidget") from e

from .backend import EditorBackend

_STATIC = pathlib.Path(__file__).parent / "static"


class DagEditor(anywidget.AnyWidget):
    _esm = _STATIC / "editor.js"
    _css = _STATIC / "editor.css"

    spec = traitlets.Dict().tag(sync=True)
    report = traitlets.Dict().tag(sync=True)
    options = traitlets.Dict().tag(sync=True)
    height = traitlets.Int(520).tag(sync=True)

    def __init__(self, cfg, df=None, height=520, **kwargs):
        self._backend = EditorBackend(cfg, df=df)
        state = self._backend.state()
        super().__init__(spec=state["spec"], report=state["report"], options=state["options"], height=height, **kwargs)
        self.on_msg(self._on_msg)

    @traitlets.observe("spec")
    def _on_spec(self, change):
        if getattr(self, "_backend", None) is None:
            return
        self.report = self._backend.update_spec(change["new"])

    def _on_msg(self, _widget, msg, _buffers):
        for k, v in self._backend.message(msg).items():
            setattr(self, k, v)

    # convenience for scripted use
    @property
    def dag(self):
        """The current (possibly unsaved) graph as :class:`tramdag.DagSpec`."""
        return self._backend.spec

    def save(self):
        out = self._backend.message({"type": "save"})
        self.report = out["report"]
        if self.report.get("errors"):
            raise ValueError("\n".join(self.report["errors"]))
        return self._backend.cfg
