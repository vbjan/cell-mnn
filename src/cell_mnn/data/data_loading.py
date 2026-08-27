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
            X,
            t_grid,
            batch_size,
            device,
            skip_idx,
            train_on_skip=False
        ):
        super().__init__()

        self.n_times = len(X)
        self.t_grid = t_grid
        self.skip_idx = skip_idx
        self.t_skip = t_grid[skip_idx]
        self.train_on_skip = train_on_skip
        self.latent_dim = X[0].shape[-1]
        assert 1 <= self.skip_idx <= self.n_times - \
            1, "skip_idx must be in [1, n_times - 1]"

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

    def __len__(self):
        # heuristic decision: set number of batches per epoch equal to same number required to seeing each cell measurement once
        return int(sum(self.cells_per_t) / self.batch_size)


class FlowMatchingDataset(TimeFilteredDataset):
    def __init__(
            self,
            X,
            flow_matcher,
            batch_size,
            device,
            skip_idx,
            t_grid,
            train_on_skip=False
        ):
        super().__init__(
            X=X,
            t_grid=t_grid,
            batch_size=batch_size,
            device=device,
            skip_idx=skip_idx,
            train_on_skip=train_on_skip,
        )
        self.flow_matcher = flow_matcher

    def __iter__(self):
        while True:
            ts_list = []
            xts_list = []
            uts_list = []

            # Go over consecutive pairs in the filtered list
            # e.g. if skip_idx=3, t_indcs might be [0,1,2,4,5],
            # so pairs are (0->1), (1->2), (2->4), (4->5).
            for i in range(len(self.X_filtered) - 1):
                x0 = self.X_filtered[i]
                x1 = self.X_filtered[i + 1]

                # Original integer time labels
                t_i = self.t_grid_filtered[i]      # e.g. 2
                t_j = self.t_grid_filtered[i + 1]  # e.g. 4
                delta_t = t_j - t_i              # e.g. 2

                # Sample random points from x0 and x1
                idx0 = np.random.randint(0, x0.shape[0], size=self.batch_size)
                idx1 = np.random.randint(0, x1.shape[0], size=self.batch_size)

                x0_sample = torch.from_numpy(x0[idx0]).float().to(self.device)
                x1_sample = torch.from_numpy(x1[idx1]).float().to(self.device)

                # Flow-matching step returns:
                #   t   \in [0,1]
                #   x_t = (1-t)*x0 + t*x1 + noise
                #   u_t = x1 - x0  (by default, total displacement)
                t, x_t, u_t = self.flow_matcher.sample_location_and_conditional_flow(
                    x0_sample, x1_sample
                )

                # 1) Scale t to the correct time interval [t_i, t_j].
                T = t_i + delta_t * t

                # 2) Convert total displacement into per-unit-time velocity:
                #    if delta_t=2, then velocity = (x1 - x0) / 2
                u_t = u_t / delta_t

                ts_list.append(T)
                xts_list.append(x_t)
                uts_list.append(u_t)

            # Concatenate across pairs
            T_cat = torch.cat(ts_list, dim=0)
            xT_cat = torch.cat(xts_list, dim=0)
            uT_cat = torch.cat(uts_list, dim=0)

            yield (xT_cat, T_cat, uT_cat)


class IndependentFlowMatchingDataset(FlowMatchingDataset):
    def __init__(self, *args, **kwargs):
        flow_matcher = ConditionalFlowMatcher(sigma=0.1)
        super().__init__(flow_matcher=flow_matcher, *args, **kwargs)


class BatchOTFlowMatchingDataset(FlowMatchingDataset):
    def __init__(self, *args, **kwargs):
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
        super().__init__(flow_matcher=flow_matcher, *args, **kwargs)


class OTFlowMatchingDataset(FlowMatchingDataset):
    """
    Extension of the FlowMatchingDataset that precomputes the entire optimal transport map over the dataset,
    once at initialization and then reuses it for each iteration.

    Unlike FlowMatchingDataset which computes the optimal transport problem for each batch on-the-fly,
    this class precomputes the matching between points at consecutive timepoints and samples from these
    precomputed pairs during iteration.
    """

    def __init__(self, *args, sigma=0.1, **kwargs):
        # Initialize with a temporary flow_matcher that will be used for precomputation
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        super().__init__(flow_matcher=flow_matcher, *args, **kwargs)

        # Precompute all optimal transport pairs between consecutive timepoints
        self.precomputed_data = []

        print("Precomputing optimal transport pairs...")
        for i in range(len(self.X_filtered) - 1):
            x0 = self.X_filtered[i]
            x1 = self.X_filtered[i + 1]

            # Convert to torch tensors
            x0_tensor = torch.from_numpy(x0).float().to(self.device)
            x1_tensor = torch.from_numpy(x1).float().to(self.device)

            # Original integer time labels
            t_i = self.t_grid_filtered[i]
            t_j = self.t_grid_filtered[i + 1]
            delta_t = t_j - t_i

            print(
                f"  Processing time pair {t_i}->{t_j} with {x0.shape[0]} source points and {x1.shape[0]} target points")

            # Use the sample_location_and_conditional_flow method on the entire dataset at once
            # This computes the OT problem for all points, not just a batch
            all_t, all_x_t, all_u_t = self.flow_matcher.sample_location_and_conditional_flow(
                x0_tensor, x1_tensor
            )

            # Scale t and u_t as in the original code
            all_T = t_i + delta_t * all_t
            all_u_t = all_u_t / delta_t

            # Store the precomputed data
            self.precomputed_data.append({
                't': all_t,
                'T': all_T,
                'x_t': all_x_t,
                'u_t': all_u_t,
                't_i': t_i,
                't_j': t_j,
                'delta_t': delta_t
            })

            print(
                f"  Precomputed {all_t.shape[0]} pairs for times {t_i}->{t_j}")

        print(
            f"Precomputation completed for {len(self.precomputed_data)} consecutive timepoint pairs")

    def __iter__(self):
        while True:
            ts_list = []
            xts_list = []
            uts_list = []

            # Sample from each precomputed pair
            for pair_data in self.precomputed_data:
                # Random sampling from the precomputed data
                n_points = pair_data['x_t'].shape[0]
                indices = self.rng.integers(0, n_points, size=self.batch_size)

                # Extract the sampled data
                ts_list.append(pair_data['T'][indices])
                xts_list.append(pair_data['x_t'][indices])
                uts_list.append(pair_data['u_t'][indices])

            # Concatenate across pairs
            T_cat = torch.cat(ts_list, dim=0)
            xT_cat = torch.cat(xts_list, dim=0)
            uT_cat = torch.cat(uts_list, dim=0)

            yield (xT_cat, T_cat, uT_cat)


class SkipMarginalEvalDataset(IterableDataset):
    def __init__(self, X, t_grid, device, skip_idx, batch_size=None):
        super().__init__()
        self.n_times = len(X)

        self.skip_idx = skip_idx
        assert 1 <= self.skip_idx <= self.n_times - \
            1, "skip_idx must be in [1, n_times - 1]"
        self.t_skip = t_grid[self.skip_idx]
        self.batch_size = batch_size

        # Predict distribution at t_skip from previous timepoint
        self.prev_idx = skip_idx - 1
        self.t_prev = t_grid[self.prev_idx]

        assert self.t_prev <= self.t_skip, f"t_skip {self.t_skip} must be less than or equal to t_prev {self.t_prev}"
        self.X_t_prev = X[self.prev_idx]
        self.X_t_skip = X[self.skip_idx]

        self.device = device

    def __iter__(self):
        while True:
            if self.batch_size is None:
                indices_prev = np.arange(self.X_t_prev.shape[0])
                indices_t_skip = np.arange(self.X_t_skip.shape[0])
            else:
                # Batched behavior: sample random indices
                indices_prev = np.random.randint(0, self.X_t_prev.shape[0], size=self.batch_size)
                indices_t_skip = np.random.randint(0, self.X_t_skip.shape[0], size=self.batch_size)

            x_t_prev = torch.from_numpy(
                self.X_t_prev[indices_prev]).float().to(self.device)
            x_t_skip = torch.from_numpy(
                self.X_t_skip[indices_t_skip]).float().to(self.device)

            t = torch.full((x_t_prev.shape[0],), float(
                self.t_prev), device=self.device)

            yield (x_t_prev, t, x_t_skip, self.t_skip)

    def __len__(self):
        if self.batch_size is None:
            # Original behavior: everything is loaded in one batch
            return 1
        else:
            # Number of batches needed to see all data once
            return int(self.X_t_prev.shape[0] / self.batch_size)


class MnnDataset(TimeFilteredDataset):
    def __init__(self, X, t_grid, batch_size, skip_idx, device, train_on_skip=False):
        super().__init__(
            X=X,
            t_grid=t_grid,
            batch_size=batch_size,
            device=device,
            skip_idx=skip_idx,
            train_on_skip=train_on_skip,
        )
        self.last_sample_idx = len(self.t_indcs) - 2

    def __iter__(self):
        while True:
            # Rows are grouped by the initial condition's timepoint. The loss sums over
            # per-row masks, so it does not depend on the order of the rows.
            x_t, t = self._sample_batch_from_all_times()
            x_population, t_population = self._sample_population()
            yield x_t, t, x_population, t_population

    def _sample_batch_from_all_times(self):
        # The final timepoint cannot serve as an initial condition: no later marginal
        # is left to supervise its trajectory.
        num_source_times = self.last_sample_idx + 1
        num_samples = split_evenly(self.batch_size, num_source_times)

        x_ts = []
        t = []
        for i, bs in enumerate(num_samples):
            cell_indices = np.random.randint(
                0, self.cells_per_t[i], size=bs)
            cells = torch.from_numpy(
                self.X_filtered[i][cell_indices]
            ).float().to(self.device)
            x_ts.append(cells.unsqueeze(1))  # Add time dimension
            t.append(torch.full((bs, 1, 1), float(self.t_grid_filtered[i]),
                                device=self.device))

        # Concatenate x_ts along the batch dimension
        x_t = torch.cat(x_ts, dim=0)
        t = torch.cat(t, dim=0)
        assert x_t.shape[0] == self.batch_size, f"Expected {self.batch_size} cells, got {x_t.shape[0]}"
        return x_t, t

    def _sample_population(self):
        # Sample batch_size points from each time point for population
        x_population = []
        for i in range(len(self.t_indcs)):
            cell_indices = np.random.randint(
                0, self.cells_per_t[i], size=self.batch_size)
            cells = torch.from_numpy(
                self.X_filtered[i][cell_indices]).float().to(self.device)
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
    

class MixedDataset(MnnDataset):
    def __init__(
            self, 
            datasets: list[MnnDataset]
        ):
        self.datasets = datasets
        self.dataset_iterators = [iter(dataset) for dataset in datasets]
        self.latent_dim = datasets[0].latent_dim
        self.t_skip = datasets[0].t_skip
        self.t_grid = [t for dataset in datasets for t in dataset.t_grid]
        self.t_grid = sorted(list(set(self.t_grid)))  # Remove duplicates and sort
        self.device = datasets[0].device

    def __iter__(self):
        while True:
            for i, dataset_iter in enumerate(self.dataset_iterators):
                try:
                    batch = next(dataset_iter)
                    yield batch
                except StopIteration:
                    # Reset the iterator for this dataset if it's exhausted
                    self.dataset_iterators[i] = iter(self.datasets[i])
                    batch = next(self.dataset_iterators[i])
                    yield batch

    def __len__(self):
        # Sum of all dataset lengths
        return sum(len(dataset) for dataset in self.datasets)


def construct_train_val_datasets(
        ds_name: str,
        skip_idx: int,
        batch_size: int,
        device,
        method: str,
        val_prop=0.0,
        train_on_all_times: bool = False
    ):
    data_dir = get_data(ds_name=ds_name, val_prop=val_prop)
    X_train = data_dir["X_train"]
    t_grid = data_dir["t_train"]

    assert method in ["ot-cfm", "mnn", "i-cfm", "batch-ot-cfm"], \
        f"Unrecognized method: {method}. Must be one of ['ot-cfm', 'mnn', 'i-cfm', 'batch-ot-cfm']"

    if method == "ot-cfm":
        ds_constructor = OTFlowMatchingDataset
    elif method == "mnn":
        ds_constructor = MnnDataset
    elif method == "i-cfm":
        ds_constructor = IndependentFlowMatchingDataset
    elif method == "batch-ot-cfm":
        ds_constructor = BatchOTFlowMatchingDataset
    else:
        raise ValueError(f"Unknown method: {method}")
    
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


def get_datasets(
        ds_name: str,
        val_ds_name: str | None,
        skip_idx: int,
        train_on_all_times: bool = False,
        *args,
        **kwargs
    ):
    if ds_name != "mix":
        assert val_ds_name is None, "val_ds_name should be None when ds_name is not 'mix'"
        return construct_train_val_datasets(
                ds_name,
                skip_idx=skip_idx,
                train_on_all_times=train_on_all_times,
                *args,
                **kwargs
            )
    else:
        all_ds_names = set(["cite", "multi"])
        assert val_ds_name in all_ds_names, f"val_ds_name must be one of {all_ds_names}, got {val_ds_name}"
        
        ds_collection = []
        train, val_dataset = construct_train_val_datasets(
            ds_name=val_ds_name,
            train_on_all_times=train_on_all_times,
            skip_idx=skip_idx,
            *args,
            **kwargs
        )
        ds_collection.append(train)

        all_ds_names.remove(val_ds_name)
        for _ds_name in all_ds_names:

            _train, _ = construct_train_val_datasets(
                ds_name=_ds_name,
                train_on_all_times=train_on_all_times,
                skip_idx=skip_idx,
                *args,
                **kwargs
            )
            ds_collection.append(_train)
        
        train_dataset = MixedDataset(
            datasets=ds_collection
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
