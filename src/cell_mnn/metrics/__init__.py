from .emd import compute_wasserstein
from .mmd import MMDLoss

# Samples scored per marginal at val/test time unless a caller says otherwise.
DEFAULT_EVAL_N_SAMPLES = 4000

__all__ = [
    "compute_wasserstein",
    "MMDLoss",
    "DEFAULT_EVAL_N_SAMPLES",
]
