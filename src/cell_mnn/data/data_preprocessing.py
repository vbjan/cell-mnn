import numpy as np
import scanpy as sc

from .marginals import TimeSeriesMarginals


def load_marginals(ds_name: str = "embryoid", pca_dims: int = 5) -> TimeSeriesMarginals:
    if ds_name == "embryoid_less_preprocessed":
        adata = sc.read_h5ad('data/ebdata/ebdata_v3_recomputed_pca.h5ad')
        t_labels = range(len(adata.obs["sample_labels"].unique()))
    elif ds_name == "embryoid":
        adata = np.load('data/ebdata/eb_velocity_v5.npz')
        t_labels = np.unique(adata["sample_labels"]) 
    elif ds_name == "embryoid_inflated":
        adata = sc.read_h5ad('data/ebdata/eb_velocity_v5_inflated.h5ad')
        t_labels = adata.obs['sample_labels'].unique()
    elif ds_name == "cite":
        adata = sc.read_h5ad('data/citedata/op_cite_inputs_0.h5ad')
        t_labels = adata.obs['day'].unique()
    elif ds_name == "cite_inflated":
        adata = sc.read_h5ad('data/citedata/op_cite_inputs_0_inflated.h5ad')
        t_labels = adata.obs['day'].unique()
    elif ds_name == "multi":
        adata = sc.read_h5ad('data/multidata/op_train_multi_targets_0.h5ad')
        t_labels = adata.obs['day'].unique()
    elif ds_name == "multi_inflated":
        adata = sc.read_h5ad('data/multidata/op_train_multi_targets_0_inflated.h5ad')
        t_labels = adata.obs['day'].unique()
    else:
        raise ValueError(f"Dataset {ds_name} not recognized.")

    if isinstance(adata, np.lib.npyio.NpzFile):
        X_pca = adata["pcs"]
    elif isinstance(adata, sc.AnnData):
        X_pca = np.array(adata.obsm["X_pca"])
    else:
        raise ValueError(f"Unsupported data format {type(adata)}. Expected AnnData or NpzFile.")

    if X_pca.shape[1] < pca_dims:
        raise ValueError(
            f"Requested PCA dimensions {pca_dims} exceed available dimensions {X_pca.shape[1]}.")

    coords = X_pca[:, :pca_dims]
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    X = []
    for t in t_labels:
        if ds_name == "embryoid_less_preprocessed":  # selection logic needs to be dataset-specific due to different naming
            mask_t = adata.obs["sample_labels"].cat.codes == t
        elif ds_name in ["cite", "multi", "cite_inflated", "multi_inflated"]:
            mask_t = adata.obs["day"] == t
        elif ds_name == "embryoid":
            mask_t = adata["sample_labels"] == t
        elif ds_name == "embryoid_inflated":
            mask_t = adata.obs["sample_labels"] == t
        else:
            raise ValueError(f"Dataset {ds_name} not recognized for time selection.")

        X.append(coords[mask_t])

    t_labels = sorted(t_labels)
    t_grid: list[float] = [float(t) for t in t_labels]  

    return TimeSeriesMarginals(X=X, t_grid=t_grid, name=ds_name)


if __name__ == "__main__":
    marginals = load_marginals(ds_name="multi")
    print("Data preprocessing complete.")
    print(marginals)
