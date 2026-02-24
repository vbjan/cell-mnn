import numpy as np
import scanpy as sc


def split_train_val_indices(n_cells, val_prop):
    """
    Split data indices into training and validation sets.

    Args:
        n_cells (int): Number of cells to split.
        val_prop (float): Proportion of cells to use for validation.

    Returns:
        tuple: (training indices, validation indices)
    """
    val_size = int(n_cells * val_prop)
    perm = np.random.permutation(n_cells)

    val_idx = perm[:val_size]
    trn_idx = perm[val_size:]

    return trn_idx, val_idx


def get_data(val_prop=0., ds_name="embryoid", pca_dims=5):
    if ds_name == "embryoid_less_preprocessed":
        adata = sc.read_h5ad('data/ebdata/ebdata_v3_recomputed_pca.h5ad')
        days = range(len(adata.obs["sample_labels"].unique()))
    elif ds_name == "embryoid":
        adata = np.load('data/ebdata/eb_velocity_v5.npz')
        days = np.unique(adata["sample_labels"]) 
    elif ds_name == "embryoid_inflated":
        adata = sc.read_h5ad('data/ebdata/eb_velocity_v5_inflated.h5ad')
        days = adata.obs['sample_labels'].unique()
    elif ds_name == "cite":
        adata = sc.read_h5ad('data/citedata/op_cite_inputs_0.h5ad')
        days = adata.obs['day'].unique()
    elif ds_name == "cite_inflated":
        adata = sc.read_h5ad('data/citedata/op_cite_inputs_0_inflated.h5ad')
        days = adata.obs['day'].unique()
    elif ds_name == "multi":
        adata = sc.read_h5ad('data/multidata/op_train_multi_targets_0.h5ad')
        days = adata.obs['day'].unique()
    elif ds_name == "multi_inflated":
        adata = sc.read_h5ad('data/multidata/op_train_multi_targets_0_inflated.h5ad')
        days = adata.obs['day'].unique()
    else:
        raise ValueError(f"Dataset {ds_name} not recognized.")
    
    days = sorted(days)

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

    X_train, X_val = [], []
    t_train, t_val = [], []
    idx_train, idx_val = [], []

    for t in days:
        if ds_name == "embryoid_less_preprocessed":  # selection logic needs to be dataset-specific due to different naming
            mask_t = adata.obs["sample_labels"].cat.codes == t
        elif ds_name in ["cite", "multi", "synthetic", "cite_inflated", "multi_inflated"]:
            mask_t = adata.obs["day"] == t
        elif ds_name == "embryoid":
            mask_t = adata["sample_labels"] == t
        elif ds_name == "embryoid_inflated":
            mask_t = adata.obs["sample_labels"] == t
        else:
            raise ValueError(f"Dataset {ds_name} not recognized for time selection.")

        # original obs indices (ordered)
        time_idx = np.where(mask_t)[0]
        time_data = coords[mask_t]

        n_cells = time_data.shape[0]
        trn_idx, val_idx = split_train_val_indices(n_cells, val_prop)

        X_train.append(time_data[trn_idx])
        X_val.append(time_data[val_idx])

        t_train.append(t)
        t_val.append(t)

        idx_train.append(time_idx[trn_idx])
        idx_val.append(time_idx[val_idx])

    return {
        'X_train': X_train,
        'X_val': X_val,
        't_train': t_train,
        't_val': t_val,
        'idx_train': idx_train,
        'idx_val': idx_val
    }


if __name__ == "__main__":
    data = get_data(ds_name="multi", val_prop=0.2)
    print("Data preprocessing complete.")
    print(f"Train data shape: {[x.shape for x in data['X_train']]}")
    print(f"Validation data shape: {[x.shape for x in data['X_val']]}")
    print(f"Train time labels: {data['t_train']}")
    print(f"Validation time labels: {data['t_val']}")
