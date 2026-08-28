from .model import CellMNN
from .data.marginals import TimeSeriesMarginals
from .data.sources import load_marginals, marginals_from_anndata
from .data.data_loading import build_datasets

__all__ = [
    "CellMNN",
    "TimeSeriesMarginals",
    "load_marginals",
    "marginals_from_anndata",
    "build_datasets",
]

# Version information
__version__ = '0.1.0'
