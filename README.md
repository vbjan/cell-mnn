# Architecture

Everything above `TimeSeriesMarginals` knows about files and formats but nothing about
training. Everything below it knows about training but nothing about files. That one
type is the entire interface between the two halves.

```mermaid
flowchart TB
    toml["datasets.toml<br>ds_name -> reader + kwargs"] --> src
    files[".h5ad / .npz"] --> src["AnnDataSource / NpzSource<br>data/sources.py"]
    src --> zs["zscore<br>data/transforms.py"]
    adata["your own AnnData"] --> mfa["marginals_from_anndata"]

    zs --> M
    mfa --> M

    M["TimeSeriesMarginals<br>X: one (n_cells_i, D) array per timepoint<br>t_grid: ascending real times"]

    M --> B["build_datasets(skip_idx, method)<br>data/data_loading.py"]

    B --> TR["MnnDataset<br>(or a CFM baseline)"]
    B --> VA["SkipMarginalEvalDataset"]

    TR -->|"(x_t, t, x_population, t_population)"| enc
    VA -->|"(x_t_prev, t, x_t_skip, t_skip)"| enc

    enc["CellMNN.encode<br>MLP: (x, t) -> A"]
    enc -->|"A: (B, 1, D, D)"| dec["CellMNN.decode_trajectory<br>x(t') = expm(A(t'-t)) x"]

    dec -->|"x_traj: (B, T, D)"| loss["train: MMD + kinetic"]
    dec -->|"x_traj at t_skip"| emd["val/test: val_emd(t_skip=...)"]
```

`skip_idx` names the held-out timepoint: training never sees it, and validation scores
the model by evolving the previous marginal forward to it, with exact Wasserstein-1.

Three extension points follow from the shape above:

- A dataset an existing reader handles is **one table in `datasets.toml`** — no code.
- A new kind of data is **one class in `sources.py`** plus one `SOURCE_TYPES` entry.
