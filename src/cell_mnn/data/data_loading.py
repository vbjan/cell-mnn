from collections.abc import Iterator
from typing import Any

import torch
import numpy as np

from torch.utils.data import IterableDataset
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher
from einops import repeat
from .data_preprocessing import get_data


def split_evenly(total: int, n: int) -> list[int]:
    """Split `total` into `n` counts that sum to `total`; the last absorbs the remainder."""
    per_part = total // n
    return [per_part] * (n - 1) + [total - per_part * (n - 1)]


def to_tensor(X: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """Cells `X` of shape (n_cells, D) as a float32 tensor on `device`."""
    return torch.from_numpy(X).float().to(device)


def sample_cells(X: np.ndarray, n: int, device: torch.device | str) -> torch.Tensor:
    """Draw `n` cells with replacement from `X` (n_cells, D) onto `device`."""
    return to_tensor(X[np.random.randint(0, X.shape[0], size=n)], device)


def assert_valid_skip_idx(skip_idx: int, n_times: int) -> None:
    assert 1 <= skip_idx <= n_times - 1, "skip_idx must be in [1, n_times - 1]"


class TimeFilteredDataset(IterableDataset):
    """
    User gives skip_idx, which is the index of the timepoint to skip.

    Datastructure:
    - t_indcs for navigating through timepoints and cells as range [0, n_times)
    - X_filtered: list of numpy arrays, one per timepoint, excluding the skipped one
    - t_grid_filtered: list of times corresponding to the acquisition times of each component of X_filtered (no t_skip)
    """

    def __init__(
            self,
            X: list[np.ndarray],
            t_grid: list[float],
            batch_size: int,
            device: torch.device | str,
            skip_idx: int,
            train_on_skip: bool = False
        ) -> None:
        super().__init__()

        self.n_times = len(X)
        self.t_grid = t_grid
        self.skip_idx = skip_idx
        self.t_skip = t_grid[skip_idx]
        self.train_on_skip = train_on_skip
        self.latent_dim = X[0].shape[-1]
        assert_valid_skip_idx(self.skip_idx, self.n_times)

        # Filter out the timepoint we want to skip
        if self.train_on_skip:
            self.t_indcs = range(self.n_times)
        else:
            self.t_indcs = [i for i in range(self.n_times) if i != self.skip_idx]

        self.t_grid_filtered = [t_grid[i] for i in self.t_indcs]
        self.X_filtered = [X[i] for i in self.t_indcs]

        self.cells_per_t = [x.shape[0] for x in self.X_filtered]

        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        # heuristic decision: set number of batches per epoch equal to same number required to seeing each cell measurement once
        return int(sum(self.cells_per_t) / self.batch_size)


class FlowMatchingDataset(TimeFilteredDataset):
    """
    Base for the CFM baselines. Subclasses only pick `flow_matcher_cls`, so every
    train dataset in this module takes the same constructor keywords.
    """

    flow_matcher_cls: type[ConditionalFlowMatcher] | None = None

    def __init__(self, *args: Any, sigma: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.flow_matcher_cls is not None, \
            f"{type(self).__name__} must set flow_matcher_cls"
        self.flow_matcher = self.flow_matcher_cls(sigma=sigma)

    def _pair(self, i: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        # Consecutive pair in the *filtered* grid: if skip_idx=3, t_indcs might be
        # [0,1,2,4,5], so the pairs are (0->1), (1->2), (2->4), (4->5).
        t_i = self.t_grid_filtered[i]      # e.g. 2
        t_j = self.t_grid_filtered[i + 1]  # e.g. 4
        return self.X_filtered[i], self.X_filtered[i + 1], t_i, t_j - t_i

    def _flow_sample(
            self,
            x0: torch.Tensor,
            x1: torch.Tensor,
            t_i: float,
            delta_t: float
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flow-matching step returns:
        #   t   \in [0,1]
        #   x_t = (1-t)*x0 + t*x1 + noise
        #   u_t = x1 - x0  (by default, total displacement)
        # [:3] drops the 4th element torchcfm only returns for return_noise=True
        t, x_t, u_t = self.flow_matcher.sample_location_and_conditional_flow(
            x0, x1
        )[:3]

        # 1) Scale t to the correct time interval [t_i, t_j].
        # 2) Convert total displacement into per-unit-time velocity:
        #    if delta_t=2, then velocity = (x1 - x0) / 2
        return x_t, t_i + delta_t * t, u_t / delta_t

    def _sample_pair(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One (x_t, T, u_t) batch for the i-th consecutive pair of marginals."""
        x0, x1, t_i, delta_t = self._pair(i)
        return self._flow_sample(
            sample_cells(x0, self.batch_size, self.device),
            sample_cells(x1, self.batch_size, self.device),
            t_i,
            delta_t,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n_pairs = len(self.X_filtered) - 1
        while True:
            per_pair = [self._sample_pair(i) for i in range(n_pairs)]
            # Concatenate across pairs
            x_t, T, u_t = (torch.cat(parts, dim=0) for parts in zip(*per_pair))
            yield x_t, T, u_t


class IndependentFlowMatchingDataset(FlowMatchingDataset):
    flow_matcher_cls = ConditionalFlowMatcher


class BatchOTFlowMatchingDataset(FlowMatchingDataset):
    flow_matcher_cls = ExactOptimalTransportConditionalFlowMatcher


class OTFlowMatchingDataset(FlowMatchingDataset):
    """
    Extension of the FlowMatchingDataset that precomputes the entire optimal transport map over the dataset,
    once at initialization and then reuses it for each iteration.

    Unlike FlowMatchingDataset which computes the optimal transport problem for each batch on-the-fly,
    this class precomputes the matching between points at consecutive timepoints and samples from these
    precomputed pairs during iteration.
    """

    flow_matcher_cls = ExactOptimalTransportConditionalFlowMatcher

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        print("Precomputing optimal transport pairs...")
        self.precomputed_pairs = []
        for i in range(len(self.X_filtered) - 1):
            x0, x1, t_i, delta_t = self._pair(i)
            print(f"  Processing time pair {t_i}->{t_i + delta_t} with "
                  f"{x0.shape[0]} source points and {x1.shape[0]} target points")

            # Couple the whole marginals at once, not just a batch
            pair = self._flow_sample(
                to_tensor(x0, self.device),
                to_tensor(x1, self.device),
                t_i,
                delta_t,
            )
            self.precomputed_pairs.append(pair)

            print(f"  Precomputed {pair[0].shape[0]} pairs for times {t_i}->{t_i + delta_t}")

        print(f"Precomputation completed for {len(self.precomputed_pairs)} consecutive timepoint pairs")

    def _sample_pair(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t, T, u_t = self.precomputed_pairs[i]
        indices = np.random.randint(0, x_t.shape[0], size=self.batch_size)
        return x_t[indices], T[indices], u_t[indices]


class SkipMarginalEvalDataset(IterableDataset):
    def __init__(
            self,
            X: list[np.ndarray],
            t_grid: list[float],
            device: torch.device | str,
            skip_idx: int,
            batch_size: int | None = None
        ) -> None:
        super().__init__()
        self.n_times = len(X)

        self.skip_idx = skip_idx
        assert_valid_skip_idx(self.skip_idx, self.n_times)
        self.t_skip = t_grid[self.skip_idx]
        self.batch_size = batch_size

        # Predict distribution at t_skip from previous timepoint
        self.prev_idx = skip_idx - 1
        self.t_prev = t_grid[self.prev_idx]

        assert self.t_prev <= self.t_skip, f"t_prev {self.t_prev} must be less than or equal to t_skip {self.t_skip}"
        self.X_t_prev = X[self.prev_idx]
        self.X_t_skip = X[self.skip_idx]

        self.device = device

    def _load_marginal(self, X: np.ndarray) -> torch.Tensor:
        # batch_size None loads the whole marginal, in order, in a single batch
        indices = (np.arange(X.shape[0]) if self.batch_size is None
                   else np.random.randint(0, X.shape[0], size=self.batch_size))
        return to_tensor(X[indices], self.device)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        while True:
            x_t_prev = self._load_marginal(self.X_t_prev)
            x_t_skip = self._load_marginal(self.X_t_skip)

            t = torch.full((x_t_prev.shape[0],), float(
                self.t_prev), device=self.device)

            yield (x_t_prev, t, x_t_skip, self.t_skip)

    def __len__(self) -> int:
        if self.batch_size is None:
            # Original behavior: everything is loaded in one batch
            return 1
        else:
            # Number of batches needed to see all data once
            return int(self.X_t_prev.shape[0] / self.batch_size)


class MnnDataset(TimeFilteredDataset):
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:
            # Rows are grouped by the initial condition's timepoint. The loss sums over
            # per-row masks, so it does not depend on the order of the rows.
            x_t, t = self._sample_batch_from_all_times()
            x_population, t_population = self._sample_population()
            yield x_t, t, x_population, t_population

    def _sample_batch_from_all_times(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The final timepoint cannot serve as an initial condition: no later marginal
        # is left to supervise its trajectory.
        num_source_times = len(self.t_indcs) - 1
        num_samples = split_evenly(self.batch_size, num_source_times)

        x_ts = []
        t = []
        for i, bs in enumerate(num_samples):
            cells = sample_cells(self.X_filtered[i], bs, self.device)
            x_ts.append(cells.unsqueeze(1))  # Add time dimension
            t.append(torch.full((bs, 1, 1), float(self.t_grid_filtered[i]),
                                device=self.device))

        # Concatenate x_ts along the batch dimension
        x_t = torch.cat(x_ts, dim=0)
        t = torch.cat(t, dim=0)
        assert x_t.shape[0] == self.batch_size, f"Expected {self.batch_size} cells, got {x_t.shape[0]}"
        return x_t, t

    def _sample_population(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample batch_size points from each time point for population
        x_population = []
        for i in range(len(self.t_indcs)):
            cells = sample_cells(
                self.X_filtered[i], self.batch_size, self.device)
            # Add time dimension
            x_population.append(cells.unsqueeze(1))

        # Concatenate population along the time dimension
        x_population = torch.cat(x_population, dim=1)
        t_population = torch.tensor(
            self.t_grid_filtered, dtype=torch.float
        ).to(self.device)
        t_population = repeat(
            t_population, 't -> b t 1', b=self.batch_size)
        return x_population, t_population


TRAIN_DATASET_BY_METHOD = {
    "mnn": MnnDataset,
    "i-cfm": IndependentFlowMatchingDataset,
    "batch-ot-cfm": BatchOTFlowMatchingDataset,
    "ot-cfm": OTFlowMatchingDataset,
}


def get_datasets(
        ds_name: str,
        skip_idx: int,
        batch_size: int,
        device: torch.device,
        method: str,
        val_prop: float = 0.0,
        train_on_all_times: bool = False
    ) -> tuple[TimeFilteredDataset, SkipMarginalEvalDataset]:
    """(train_dataset, val_dataset) for `ds_name`, holding out marginal `skip_idx`."""
    data_dir = get_data(ds_name=ds_name, val_prop=val_prop)
    X_train = data_dir["X_train"]
    t_grid = data_dir["t_train"]

    if method not in TRAIN_DATASET_BY_METHOD:
        raise ValueError(
            f"Unrecognized method: {method}. "
            f"Must be one of {sorted(TRAIN_DATASET_BY_METHOD)}"
        )
    ds_constructor = TRAIN_DATASET_BY_METHOD[method]

    train_dataset = ds_constructor(
        X=X_train,
        t_grid=t_grid,
        skip_idx=skip_idx,
        batch_size=batch_size,
        device=device,
        train_on_skip=train_on_all_times
    )
    # Compute validation score batch wise only if dataset is too big to compute OT for all points
    too_big = X_train[0].shape[0] > 10_000
    val_dataset = SkipMarginalEvalDataset(
        X=X_train,
        t_grid=t_grid,
        device=device,
        skip_idx=skip_idx,
        batch_size=batch_size if too_big else None,
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    data_dir = get_data(val_prop=0.1)
    X_train = data_dir["X_train"]
    t_grid = data_dir["t_train"]
    train_dataset = MnnDataset(
        X=X_train,
        t_grid=t_grid,
        batch_size=10,
        device="cpu",
        skip_idx=2,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=None)
    dataiter = iter(dataloader)
    x_t, t, x_population, t_population = next(dataiter)
    print(f"x_t: {x_t.shape}, t: {t.shape}, x_population: {x_population.shape}, t_population: {t_population.shape}")
    print(f"t:{t}, t_population: {t_population}")
