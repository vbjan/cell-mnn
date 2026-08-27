import torch
import numpy as np

from torch.utils.data import IterableDataset
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher
from einops import repeat
from .data_preprocessing import get_data

class FlowMatchingDataset(IterableDataset):
    """
    An IterableDataset that, on each iteration, samples from all
    consecutive *filtered* timepoints (skipping one day) and yields:
      - xT the sampled points at fractional time T.
      - T  in the real interval [day_i, day_j] (not just day_i + t).
      - uT the *per-unit-time* velocity, i.e. (x1 - x0) / (day_j - day_i).
    """

    def __init__(
            self, 
            X, 
            flow_matcher, 
            batch_size, 
            device, 
            skip_day_idx,
            days,
            train_on_skip_day=False
        ):
        """
        Args:
            sigma (float): Noise scale for ExactOptimalTransportConditionalFlowMatcher
            batch_size (int): How many points to sample per pair of days
            device (torch.device): Where to put data (CPU or CUDA)
            skip_day (int): Which day index to skip entirely
        """
        super().__init__()

        self.n_days = len(X)
        self.days = days
        self.skip_day_idx = skip_day_idx
        self.t_skip = days[skip_day_idx]
        self.train_on_skip_day = train_on_skip_day
        self.latent_dim = X[0].shape[-1]
        assert 1 <= self.skip_day_idx <= self.n_days - \
            1, "skip_day_idx must be in [1, n_days - 1]"

        # Filter out the day we want to skip
        if self.train_on_skip_day:
            self.day_indices = range(self.n_days)
        else:
            self.day_indices = [i for i in range(self.n_days) if i != self.skip_day_idx]
            
        self.days_filtered = [days[i] for i in self.day_indices]
        self.X_filtered = [X[i] for i in self.day_indices]

        self.cells_per_day = [x.shape[0] for x in self.X_filtered]

        self.flow_matcher = flow_matcher
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        # heuristic decision: set number of batches per epoch equal to same number required to seeing each cell measurement once
        return int(sum(self.cells_per_day) / self.batch_size)

    def __iter__(self):
        """
        Yields:
            xT  (Tensor): shape [(num_pairs)*batch_size, dim].
            T   (Tensor): shape [(num_pairs)*batch_size].
                          Real time in [day_i, day_j], properly scaled.
            uT  (Tensor): shape [(num_pairs)*batch_size, dim].
                          Per-day velocity = (x1 - x0) / (day_j - day_i).
        """
        while True:
            ts_list = []
            xts_list = []
            uts_list = []

            # Go over consecutive pairs in the filtered list
            # e.g. if skip_day=3, day_indices might be [0,1,2,4,5],
            # so pairs are (0->1), (1->2), (2->4), (4->5).
            for i in range(len(self.X_filtered) - 1):
                x0 = self.X_filtered[i]
                x1 = self.X_filtered[i + 1]

                # Original integer day labels
                day_i = self.days_filtered[i]      # e.g. 2
                day_j = self.days_filtered[i + 1]  # e.g. 4
                day_diff = day_j - day_i         # e.g. 2

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

                # 1) Scale t to the correct day interval [day_i, day_j].
                T = day_i + day_diff * t

                # 2) Convert total displacement into per-unit-time velocity:
                #    if day_diff=2, then velocity = (x1 - x0) / 2
                u_t = u_t / day_diff  

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
        """
        Args:
            X (list): List of numpy arrays, one per timepoint
            batch_size (int): How many points to sample per pair of days
            device (torch.device): Where to put data (CPU or CUDA)
            skip_day (int): Which day index to skip entirely
            sigma (float): Noise scale for ExactOptimalTransportConditionalFlowMatcher
            seed (int): Optional seed for reproducibility
        """
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

            # Original integer day labels
            day_i = self.days_filtered[i]
            day_j = self.days_filtered[i + 1]
            day_diff = day_j - day_i

            print(
                f"  Processing day pair {day_i}->{day_j} with {x0.shape[0]} source points and {x1.shape[0]} target points")

            # Use the sample_location_and_conditional_flow method on the entire dataset at once
            # This computes the OT problem for all points, not just a batch
            all_t, all_x_t, all_u_t = self.flow_matcher.sample_location_and_conditional_flow(
                x0_tensor, x1_tensor
            )

            # Scale t and u_t as in the original code
            all_T = day_i + day_diff * all_t
            all_u_t = all_u_t / day_diff

            # Store the precomputed data
            self.precomputed_data.append({
                't': all_t,
                'T': all_T,
                'x_t': all_x_t,
                'u_t': all_u_t,
                'day_i': day_i,
                'day_j': day_j,
                'day_diff': day_diff
            })

            print(
                f"  Precomputed {all_t.shape[0]} pairs for days {day_i}->{day_j}")

        print(
            f"Precomputation completed for {len(self.precomputed_data)} consecutive timepoint pairs")

    def __iter__(self):
        """
        Yields:
            xT  (Tensor): shape [(num_pairs)*batch_size, dim].
            T   (Tensor): shape [(num_pairs)*batch_size].
                          Real time in [day_i, day_j], properly scaled.
            uT  (Tensor): shape [(num_pairs)*batch_size, dim].
                          Per-day velocity = (x1 - x0) / (day_j - day_i).
        """
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


class EmbryoidFlowMatchingTestDataset(IterableDataset):
    def __init__(self, X, days, device, skip_day_idx, batch_size=None):
        super().__init__()
        self.n_times = len(X)

        assert 1 <= self.skip_day_idx <= self.n_times - \
            1, "skip_day_idx must be in [1, n_days - 1]"
        self.skip_day_idx = skip_day_idx
        self.t_skip = days[self.skip_day_idx]
        self.batch_size = batch_size

        # Predict distribution at t_skip from previous day
        self.prev_day_idx = skip_day_idx - 1
        self.t_prev = days[self.prev_day_idx]

        assert self.t_prev <= self.t_skip, f"t_skip {self.t_skip} must be less than or equal to t_prev {self.t_prev}"
        self.X_prev_day = X[self.prev_day_idx]
        self.X_t_skip = X[self.skip_day_idx]

        self.device = device

    def __iter__(self):
        while True:
            if self.batch_size is None:
                indices_prev = np.arange(self.X_prev_day.shape[0])
                indices_t_skip = np.arange(self.X_t_skip.shape[0])
            else:
                # Batched behavior: sample random indices
                indices_prev = np.random.randint(0, self.X_prev_day.shape[0], size=self.batch_size)
                indices_t_skip = np.random.randint(0, self.X_t_skip.shape[0], size=self.batch_size)

            x_prev_day = torch.from_numpy(
                self.X_prev_day[indices_prev]).float().to(self.device)
            x_t_skip = torch.from_numpy(
                self.X_t_skip[indices_t_skip]).float().to(self.device)

            t = torch.full((x_prev_day.shape[0],), float(
                self.t_prev), device=self.device)

            yield (x_prev_day, t, x_t_skip, self.t_skip)

    def __len__(self):
        if self.batch_size is None:
            # Original behavior: everything is loaded in one batch
            return 1
        else:
            # Number of batches needed to see all data once
            return int(self.X_prev_day.shape[0] / self.batch_size)


class MnnDataset(IterableDataset):
    def __init__(self, X, days, batch_size, skip_day_idx, device, train_on_skip_day=False):
        """
        User gives skip_day_idx, which is the index of the day to skip.

        Datastructure:
        - day_indices for navigating through days and cells as range [0, n_days)
        - X_filtered: list of numpy arrays, one per day, excluding the skip_day
        - days_filtered: list of days corresponding to the acquisition days of each component of X_filtered (no skip_day)
        """
        super().__init__()
        self.train_on_skip_day = train_on_skip_day
        n_days = len(X)
        self.latent_dim = X[0].shape[-1]
        self.days = days

        self.skip_day_idx = skip_day_idx
        assert 1 <= self.skip_day_idx <= n_days - \
            1, "skip_day_idx must be in [1, n_days - 1]"
        self.t_skip = days[skip_day_idx]

        # Filter out the day we want to skip
        if self.train_on_skip_day:
            self.day_indices = range(n_days)
        else:
            self.day_indices = [i for i in range(
                n_days) if i != self.skip_day_idx]

        self.X_filtered = [X[i] for i in self.day_indices]
        self.days_filtered = [days[i] for i in self.day_indices]
        self.last_sample_day_idx = len(self.day_indices) - 2

        self.cells_per_day = [x.shape[0] for x in self.X_filtered]

        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return int(sum(self.cells_per_day) / self.batch_size)

    def __iter__(self):
        while True:
            x_t, t = self._sample_batch_from_all_days()

            x_population, t_population = self._sample_population()

            # shuffle the batch
            perm = torch.randperm(self.batch_size, device=self.device)
            x_t = x_t[perm]
            t = t[perm]
            x_population = x_population[perm]
            t_population = t_population[perm]
            yield x_t, t, x_population, t_population

    def _sample_batch_from_all_days(self):
        per_day_samples = self.batch_size // (self.last_sample_day_idx + 1)
        last_day_num_samples = self.batch_size - \
            per_day_samples * self.last_sample_day_idx

        # Create x_population with samples from all time points
        x_ts = []
        t = []
        for i in range(len(self.day_indices)):
            # Sample points from each time point for initial condition
            is_valid_train_input = (i <= self.last_sample_day_idx)
            bs = per_day_samples if i != self.last_sample_day_idx else last_day_num_samples
            if is_valid_train_input:
                cell_indices = np.random.randint(
                    0, self.cells_per_day[i], size=bs)
                cells = torch.from_numpy(
                    self.X_filtered[i][cell_indices]
                ).float().to(self.device)
                x_ts.append(cells.unsqueeze(1))  # Add time dimension
                t.append(torch.full((bs, 1, 1), float(self.days_filtered[i]),
                                    device=self.device))

        # Concatenate x_ts along the batch dimension
        x_t = torch.cat(x_ts, dim=0)
        t = torch.cat(t, dim=0)
        assert x_t.shape[0] == self.batch_size, f"Expected {self.batch_size} cells, got {x_t.shape[0]}"
        return x_t, t

    def _sample_population(self):
        # Sample batch_size points from each time point for population
        x_population = []
        for i in range(len(self.day_indices)):
            cell_indices = np.random.randint(
                0, self.cells_per_day[i], size=self.batch_size)
            cells = torch.from_numpy(
                self.X_filtered[i][cell_indices]).float().to(self.device)
            # Add time dimension
            x_population.append(cells.unsqueeze(1))

        # Concatenate population along the time dimension
        x_population = torch.cat(x_population, dim=1)
        t_population = torch.tensor(
            self.days_filtered, dtype=torch.float
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
        self.days = [day for dataset in datasets for day in dataset.days]
        self.days = sorted(list(set(self.days)))  # Remove duplicates and sort
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
        skip_day_idx: int,
        batch_size: int,
        device,   
        method: str, 
        val_prop=0.0,
        train_on_all_days: bool = False
    ):
    data_dir = get_data(ds_name=ds_name, val_prop=val_prop)
    X_train = data_dir["X_train"]
    days = data_dir["t_train"]

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
        days=days,
        skip_day_idx=skip_day_idx,
        batch_size=batch_size,
        device=device,
        train_on_skip_day=train_on_all_days
    )
    # Compute validation score batch wise only if dataset is too big to compute OT for all points
    too_big = X_train[0].shape[0] > 10_000 
    val_dataset = EmbryoidFlowMatchingTestDataset(
        X=X_train,
        days=days,
        device=device,
        skip_day_idx=skip_day_idx,
        batch_size=batch_size if too_big else None,
    )
    return train_dataset, val_dataset


def get_datasets(
        ds_name: str,
        val_ds_name: str | None,
        skip_day_idx: int,
        train_on_all_days: bool = False,
        *args,
        **kwargs
    ):
    if ds_name != "mix":
        assert val_ds_name is None, "val_ds_name should be None when ds_name is not 'mix'"
        return construct_train_val_datasets(
                ds_name, 
                skip_day_idx=skip_day_idx,
                train_on_all_days=train_on_all_days,
                *args, 
                **kwargs
            )
    else:
        all_ds_names = set(["cite", "multi"])
        assert val_ds_name in all_ds_names, f"val_ds_name must be one of {all_ds_names}, got {val_ds_name}"
        
        ds_collection = []
        train, val_dataset = construct_train_val_datasets(
            ds_name=val_ds_name,
            train_on_all_days=train_on_all_days,
            skip_day_idx=skip_day_idx,
            *args,
            **kwargs
        )
        ds_collection.append(train)

        all_ds_names.remove(val_ds_name)
        for _ds_name in all_ds_names:

            _train, _ = construct_train_val_datasets(
                ds_name=_ds_name,
                train_on_all_days=train_on_all_days,
                skip_day_idx=skip_day_idx,  
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
    days = data_dir["t_train"]
    train_dataset = MnnDataset(
        X=X_train,
        days=days,
        batch_size=10,
        device="cpu",
        skip_day_idx=2,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=None)
    dataiter = iter(dataloader)
    x_t, t, x_population, t_population = next(dataiter)
    print(f"x_t: {x_t.shape}, t: {t.shape}, x_population: {x_population.shape}, t_population: {t_population.shape}")
    print(f"t:{t}, t_population: {t_population}")
