"""
Dataset sources: Pipeline from stored data to `TimeSeriesMarginals`.

A *source* knows where its data lives and how to find the acquisition time and the
feature embedding inside it. 

Every reader funnels into `marginals_from_flat`, so the grouping-by-time rule is
written down exactly once.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .marginals import TimeSeriesMarginals
from .transforms import zscore

if TYPE_CHECKING:
    from anndata import AnnData


class MarginalSource(Protocol):
    """Anything that can produce a `TimeSeriesMarginals`."""

    def load(self, n_features: int, name: str) -> TimeSeriesMarginals: ...


def marginals_from_flat(
        coords: np.ndarray,    # (n_cells, k)
        t_values: np.ndarray,  # (n_cells,)
        n_features: int,
        name: str,
    ) -> TimeSeriesMarginals:
    """
    Group flat per-cell observations into one marginal per unique time.
    """
    if coords.ndim != 2:
        raise ValueError(
            f"{name}: embedding must be 2D (n_cells, n_components), got {coords.ndim}D")

    if coords.shape[1] < n_features:
        raise ValueError(
            f"{name}: requested {n_features} features but the embedding only has "
            f"{coords.shape[1]} components")

    if t_values.ndim != 1:
        raise ValueError(
            f"{name}: time labels must be 1D (n_cells,), got {t_values.ndim}D")

    if t_values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{name}: {coords.shape[0]} cells in the embedding but "
            f"{t_values.shape[0]} time labels")

    coords = coords[:, :n_features]

    X, t_grid = [], []
    for t in np.unique(t_values):  # np.unique sorts; pandas' does not
        X.append(coords[t_values == t])
        t_grid.append(float(t))

    return TimeSeriesMarginals(X=X, t_grid=t_grid, name=name)


def marginals_from_anndata(
        adata: "AnnData",
        time_key: str,
        n_features: int,
        embedding_key: str = "X_pca",
        use_codes: bool = False,
        name: str = "anndata",
    ) -> TimeSeriesMarginals:
    """
    `TimeSeriesMarginals` from an in-memory AnnData..

    Args:
        time_key: `obs` column holding the acquisition time of each cell.
        embedding_key: `obsm` key holding the precomputed embedding.
        use_codes: take the categorical *codes* of `time_key` instead of its
            values. For datasets whose labels are strings ("Day 00-03"), where the
            category order carries the time order and the values do not parse as
            numbers.
    """
    if time_key not in adata.obs:
        raise KeyError(
            f"{name}: obs has no column {time_key!r}; available: {list(adata.obs)}")

    if embedding_key not in adata.obsm:
        raise KeyError(
            f"{name}: obsm has no key {embedding_key!r}; available: {list(adata.obsm)}")

    labels = adata.obs[time_key]

    if use_codes:
        if not hasattr(labels, "cat"):
            raise TypeError(
                f"{name}: use_codes requires obs[{time_key!r}] to be categorical, "
                f"got dtype {labels.dtype}")
        labels = labels.cat.codes

    return marginals_from_flat(
        coords=np.asarray(adata.obsm[embedding_key]),
        t_values=np.asarray(labels),
        n_features=n_features,
        name=name,
    )


@dataclass(frozen=True)
class AnnDataSource:
    """An h5ad file with a precomputed embedding in `obsm` and times in `obs`."""

    path: str
    time_key: str
    embedding_key: str = "X_pca"
    use_codes: bool = False

    def load(self, n_features: int, name: str) -> TimeSeriesMarginals:
        # Local import: scanpy is a heavy dependency that only this reader needs.
        import scanpy as sc

        return marginals_from_anndata(
            sc.read_h5ad(self.path),
            time_key=self.time_key,
            n_features=n_features,
            embedding_key=self.embedding_key,
            use_codes=self.use_codes,
            name=name,
        )


@dataclass(frozen=True)
class NpzSource:
    """An .npz archive holding the embedding and the times as flat arrays."""

    path: str
    time_key: str = "sample_labels"
    embedding_key: str = "pcs"

    def load(self, n_features: int, name: str) -> TimeSeriesMarginals:
        with np.load(self.path) as npz:
            for key in (self.time_key, self.embedding_key):
                if key not in npz:
                    raise KeyError(
                        f"{name}: {self.path} has no array {key!r}; "
                        f"available: {list(npz)}")

            coords = np.asarray(npz[self.embedding_key])
            t_values = np.asarray(npz[self.time_key])

        return marginals_from_flat(
            coords=coords,
            t_values=t_values,
            n_features=n_features,
            name=name,
        )


# Paths are relative to the repo root -- every script must be launched from there.
DATASETS: dict[str, MarginalSource] = {
    "embryoid": NpzSource(
        "data/ebdata/eb_velocity_v5.npz",
    ),
    "embryoid_less_preprocessed": AnnDataSource(
        "data/ebdata/ebdata_v3_recomputed_pca.h5ad",
        time_key="sample_labels",
        use_codes=True,
    ),
    "embryoid_inflated": AnnDataSource(
        "data/ebdata/eb_velocity_v5_inflated.h5ad",
        time_key="sample_labels",
    ),
    "cite": AnnDataSource(
        "data/citedata/op_cite_inputs_0.h5ad",
        time_key="day",
    ),
    "cite_inflated": AnnDataSource(
        "data/citedata/op_cite_inputs_0_inflated.h5ad",
        time_key="day",
    ),
    "multi": AnnDataSource(
        "data/multidata/op_train_multi_targets_0.h5ad",
        time_key="day",
    ),
    "multi_inflated": AnnDataSource(
        "data/multidata/op_train_multi_targets_0_inflated.h5ad",
        time_key="day",
    ),
}


def load_marginals(
        ds_name: str = "embryoid",
        n_features: int = 5,
        standardize: bool = True,
    ) -> TimeSeriesMarginals:
    """
    Args:
        ds_name: key into `DATASETS`.
        n_features: how many leading components of the *precomputed* embedding to
            keep.
        standardize: pooled z-score over all timepoints.
    """
    if ds_name not in DATASETS:
        raise ValueError(
            f"Dataset {ds_name!r} not recognized. Must be one of {sorted(DATASETS)}")

    marginals = DATASETS[ds_name].load(n_features=n_features, name=ds_name)

    return zscore(marginals) if standardize else marginals


if __name__ == "__main__":
    # Smoke-checks every registry entry; reports the ones that are not downloaded
    # instead of failing, so it is runnable on a fresh clone.
    for ds_name, source in sorted(DATASETS.items()):
        try:
            print(load_marginals(ds_name=ds_name))
        except OSError:
            print(f"{ds_name}: could not read {source}")
