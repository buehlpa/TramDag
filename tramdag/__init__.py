"""
TRAM-DAG: interpretable neural causal models with transformation models on a DAG.

The default engine runs on JAX (CPU / GPU / TPU). The original PyTorch implementation
is still available as :mod:`tramdag.legacy_torch` (needs ``torch``).

Quick start::

    from tramdag import TramDagConfig, TramDagModel

    cfg = TramDagConfig.from_dataframe(
        train_df,
        edges="x1 -> x2; x1 -> x3; x2 -> x3 [cs]",
        experiment_dir="runs/example",
    )
    model = TramDagModel.from_config(cfg)
    model.fit(train_df, val_df, epochs=200)
    samples, latents = model.sample(do_interventions={"x1": 0.0})
"""

import os as _os
from importlib.metadata import PackageNotFoundError, version

# TRAM-DAG models are small; grabbing 75% of the GPU up front (JAX default) only gets in
# the way of notebooks and other processes. Respect an explicit user setting.
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from .TramDagConfig import TramDagConfig

_LAZY = {
    "TramDagModel": ("tramdag.model", "TramDagModel"),
    "DagSpec": ("tramdag.dag_spec", "DagSpec"),
    "register_model": ("tramdag.engine.nets", "register_model"),
    "available_models": ("tramdag.engine.nets", "available_models"),
    "DagEditor": ("tramdag.ui.widget", "DagEditor"),
    # legacy torch API (requires torch)
    "TramDagDataset": ("tramdag.legacy_torch.TramDagDataset", "TramDagDataset"),
    "TorchTramDagModel": ("tramdag.legacy_torch.TramDagModel", "TramDagModel"),
}

__all__ = ["TramDagConfig", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tramdag' has no attribute '{name}'")


try:
    __version__ = version("tramdag")
except PackageNotFoundError:
    __version__ = "0.0.0"
