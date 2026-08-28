from .model import CellMNN
from .data.marginals import TimeSeriesMarginals
from .data.data_preprocessing import load_marginals
from .data.data_loading import build_datasets

__all__ = [
    "CellMNN",
    "TimeSeriesMarginals",
    "load_marginals",
    "build_datasets",
]

# Version information
__version__ = '0.1.0'
