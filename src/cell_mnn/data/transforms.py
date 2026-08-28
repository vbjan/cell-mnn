import numpy as np

from .marginals import TimeSeriesMarginals


def zscore(marginals: TimeSeriesMarginals) -> TimeSeriesMarginals:
    """
    Standardize each feature to zero mean and unit variance.
    Statistics are pooled over *all* timepoints.
    """
    pooled = np.concatenate(marginals.X, axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)

    # check for constant feature with zero variance
    constant = np.flatnonzero(std == 0).tolist()
    if constant:
        raise ValueError(f"{marginals.name}: features {constant} are constant")

    return TimeSeriesMarginals(
        X=[(x - mean) / std for x in marginals.X],
        t_grid=list(marginals.t_grid),
        name=marginals.name,
    )
