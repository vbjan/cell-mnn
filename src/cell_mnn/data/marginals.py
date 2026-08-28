from dataclasses import dataclass

import numpy as np


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
        # This method just checks whether X and t_grid fulfill the constraints that are needed 
        # for the downstream datasets and model training
        if len(self.X) == 0:
            raise ValueError(f"{self.name}: no marginals given")
        
        if len(self.X) != len(self.t_grid):
            raise ValueError(
                f"{self.name}: got {len(self.X)} marginals but {len(self.t_grid)} times"
            )

        for i, x in enumerate(self.X):
            if x.ndim != 2:
                raise ValueError(
                    f"{self.name}: marginals must be 2D (n_cells, n_features), "
                    f"got {x.ndim}D at index {i}"
                )
            
            if x.shape[0] == 0:
                raise ValueError(f"{self.name}: marginal at index {i} (t={self.t_grid[i]}) is empty")
            
            if x.shape[1] != self.n_features:
                raise ValueError(
                    f"{self.name}: inconsistent feature dimension -- {self.n_features} at index 0 "
                    f"but {x.shape[1]} at index {i}"
                )
            
            if not np.isfinite(x).all():
                raise ValueError(
                    f"{self.name}: marginal at index {i} (t={self.t_grid[i]}) "
                    f"contains NaN or inf"
                )

        # Strict: equal times would make delta_t zero
        for i, (t_prev, t_next) in enumerate(zip(self.t_grid, self.t_grid[1:])):
            if t_next <= t_prev:
                raise ValueError(
                    f"{self.name}: t_grid must be strictly ascending, "
                    f"got t[{i}]={t_prev} >= t[{i + 1}]={t_next}"
                )

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



