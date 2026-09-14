from dataclasses import dataclass

import numpy as np

from ..checks import require


@dataclass(frozen=True)
class ZScoreParams:
    """
    Per-feature statistics of the data as it was *before* standardization.

    Recorded on the standardized `TimeSeriesMarginals` so the transform can be
    undone (to report in the original units) or replayed on a second timecourse
    that has to land in the same coordinates.
    """

    mean: np.ndarray  # (n_features,)
    std: np.ndarray   # (n_features,)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean


@dataclass(frozen=True)
class MinMaxParams:
    """
    The interval a coordinate spanned *before* it was scaled onto [0, 1].

    `span` is also the factor the dynamics absorb: scaling time by it leaves the
    trajectory unchanged only if `A` grows by the same factor, so `A_original =
    A_scaled / span` recovers the operator in the original time units.
    """

    lo: float
    hi: float

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def apply(self, t: float) -> float:
        return (t - self.lo) / self.span

    def inverse(self, t: float) -> float:
        return t * self.span + self.lo


@dataclass(repr=False)
class TimeSeriesMarginals:
    """
    A time series observed as *unpaired marginals*: one population of samples per timepoint.

    This is the format between data sources and datasets. 

    Attributes:
        X: one array of shape (n_cells_i, n_features) per timepoint. Cells are unpaired
           across timepoints, and `n_cells_i` may differ between them.
        t_grid: acquisition time of each entry of `X`, strictly ascending. These are real
           times, not indices: they enter the dynamics directly as `expm(A * delta_t)`, so
           non-uniform spacing is meaningful.
        name: dataset label, used in error messages and logs.
        feature_scaling: statistics of `X` before standardization, or None if `X` is
           still in its original units. Set by `transforms.zscore`.
        time_scaling: interval `t_grid` spanned before it was scaled onto [0, 1], or
           None if `t_grid` is still in its original units. Set by `transforms.minmax_time`.
    """

    X: list[np.ndarray]
    t_grid: list[float]
    name: str = "unnamed"
    feature_scaling: ZScoreParams | None = None
    time_scaling: MinMaxParams | None = None

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def n_times(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X[0].shape[1]

    @property
    def cells_per_t(self) -> list[int]:
        return [x.shape[0] for x in self.X]

    @property
    def n_cells(self) -> int:
        return sum(self.cells_per_t)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """The `idx`-th marginal and the time it was acquired at."""
        return self.X[idx], self.t_grid[idx]

    def drop(self, idx: int) -> "TimeSeriesMarginals":
        """A copy without the `idx`-th marginal (the held-out timepoint)."""
        if not 0 <= idx < self.n_times:
            raise IndexError(
                f"{self.name}: cannot drop index {idx} from {self.n_times} marginals")

        keep = [i for i in range(self.n_times) if i != idx]

        return TimeSeriesMarginals(
            X=[self.X[i] for i in keep],
            t_grid=[self.t_grid[i] for i in keep],
            name=self.name,
            feature_scaling=self.feature_scaling,
            time_scaling=self.time_scaling,
        )

    def __repr__(self) -> str:
        # The default dataclass repr would dump every cell of every marginal.
        return (f"{type(self).__name__}(name={self.name!r}, n_times={self.n_times}, "
                f"n_features={self.n_features}, cells_per_t={self.cells_per_t}, "
                f"t_grid={self.t_grid}, "
                f"standardized={self.feature_scaling is not None}, "
                f"time_scaled={self.time_scaling is not None})")


def _validate(m: TimeSeriesMarginals) -> None:
    """
    The constraints the downstream datasets and model training rely on.
    Folded out of `__post_init__` so the class body reads as the data contract
    rather than as a wall of checks. 
    """
    require(len(m.X) > 0, f"{m.name}: no marginals given")
    require(len(m.X) == len(m.t_grid),
            f"{m.name}: got {len(m.X)} marginals but {len(m.t_grid)} times")

    for i, x in enumerate(m.X):
        at = f"{m.name}: marginal at index {i} (t={m.t_grid[i]})"
        require(x.ndim == 2,
                f"{at} must be 2D (n_cells, n_features), got {x.ndim}D")
        require(x.shape[0] > 0, f"{at} is empty")
        require(x.shape[1] == m.n_features,
                f"{at} has {x.shape[1]} features, but index 0 has {m.n_features}")
        require(np.isfinite(x).all(), f"{at} contains NaN or inf")

    # Strict: equal times would make delta_t zero
    for i, (t_prev, t_next) in enumerate(zip(m.t_grid, m.t_grid[1:])):
        require(t_prev < t_next,
                f"{m.name}: t_grid must be strictly ascending, "
                f"got t[{i}]={t_prev} >= t[{i + 1}]={t_next}")

    if (fs := m.feature_scaling) is not None:
        require(fs.mean.shape == (m.n_features,),
                f"{m.name}: feature_scaling.mean has shape {fs.mean.shape}, "
                f"expected ({m.n_features},)")
        require(fs.std.shape == (m.n_features,),
                f"{m.name}: feature_scaling.std has shape {fs.std.shape}, "
                f"expected ({m.n_features},)")
        require(bool(np.all(fs.std > 0)),
                f"{m.name}: feature_scaling.std must be positive to be invertible")

    if (ts := m.time_scaling) is not None:
        require(ts.hi > ts.lo,
                f"{m.name}: time_scaling must span a positive interval, "
                f"got lo={ts.lo} >= hi={ts.hi}")
