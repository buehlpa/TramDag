"""Original PyTorch implementation of TRAM-DAG.

This subpackage is kept as the numerical reference for the JAX engine
(see ``tests/test_parity_torch.py``). It requires ``torch`` and ``torchvision``.
New code should use :class:`tramdag.TramDagModel`, which runs on JAX.
"""
from .TramDagModel import TramDagModel
from .TramDagDataset import TramDagDataset

__all__ = ["TramDagModel", "TramDagDataset"]
