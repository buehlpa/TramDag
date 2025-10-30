# tramdag/__init__.py
from .TramDagConfig import TramDagConfig
from .TramDagDataset import TramDagDataset
from .TramDagModel import TramDagModel
from importlib.metadata import version, PackageNotFoundError


__all__ = ["TramDagConfig", "TramDagDataset", "TramDagModel"]

try:
    __version__ = version("tramdag")
except PackageNotFoundError:
    __version__ = "0.0.0"