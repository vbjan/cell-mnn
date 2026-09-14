import numpy as np

from ..checks import require
from .marginals import MinMaxParams, TimeSeriesMarginals, ZScoreParams


def zscore(marginals: TimeSeriesMarginals) -> TimeSeriesMarginals:
    """
    Standardize each feature to zero mean and unit variance.
    Statistics are pooled over *all* timepoints, and recorded on the result as
    `feature_scaling`.
    """
    require(marginals.feature_scaling is None,
            f"{marginals.name}: already standardized. Re-fitting would record the "
            f"statistics of standardized data and lose the original units")

    pooled = np.concatenate(marginals.X, axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)

    constant = np.flatnonzero(std == 0).tolist()
    require(not constant, f"{marginals.name}: features {constant} are constant")

    params = ZScoreParams(mean=mean, std=std)

    return TimeSeriesMarginals(
        X=[params.apply(x) for x in marginals.X],
        t_grid=list(marginals.t_grid),
        name=marginals.name,
        feature_scaling=params,
        time_scaling=marginals.time_scaling,
    )


def minmax_time(marginals: TimeSeriesMarginals) -> TimeSeriesMarginals:
    """
    Scale `t_grid` onto [0, 1], recording the original interval as `time_scaling`.

    Relative spacing is preserved, so non-uniform gaps stay meaningful.
    """
    require(marginals.time_scaling is None,
            f"{marginals.name}: time already scaled. Re-fitting would record the "
            f"interval of scaled time and lose the original units")
    require(marginals.n_times > 1,
            f"{marginals.name}: cannot min-max scale time with a single timepoint; "
            f"there is no interval to scale onto [0, 1]")

    # t_grid is validated strictly ascending, so the ends are the min and the max.
    params = MinMaxParams(lo=marginals.t_grid[0], hi=marginals.t_grid[-1])

    return TimeSeriesMarginals(
        X=list(marginals.X),
        t_grid=[params.apply(t) for t in marginals.t_grid],
        name=marginals.name,
        feature_scaling=marginals.feature_scaling,
        time_scaling=params,
    )
