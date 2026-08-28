from dataclasses import dataclass

import numpy as np

from ..checks import require


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
    """

    X: list[np.ndarray]
    t_grid: list[float]
    name: str = "unnamed"

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
        )

    def __repr__(self) -> str:
        # The default dataclass repr would dump every cell of every marginal.
        return (f"{type(self).__name__}(name={self.name!r}, n_times={self.n_times}, "
                f"n_features={self.n_features}, cells_per_t={self.cells_per_t}, "
                f"t_grid={self.t_grid})")


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
