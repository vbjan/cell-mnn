# Architecture

Everything above `TimeSeriesMarginals` knows about files and formats but nothing about
training. Everything below it knows about training but nothing about files. That one
type is the entire interface between the two halves.

```mermaid
flowchart TB
    toml["datasets.toml<br>ds_name -> reader + kwargs"] --> src
    files[".h5ad / .npz"] --> src["AnnDataSource / NpzSource<br>data/sources.py"]
    src --> zs["zscore + minmax_time<br>data/transforms.py"]
    adata["your own AnnData"] --> mfa["marginals_from_anndata"]

    zs --> M
    mfa --> M

    M["TimeSeriesMarginals<br>X: one (n_cells_i, D) array per timepoint<br>t_grid: ascending times, scaled to [0,1]<br>feature_scaling / time_scaling: how to undo it"]

    M --> B["build_datasets(skip_idx, method)<br>data/data_loading.py"]

    B --> TR["MnnDataset<br>(or a CFM baseline)"]
    B --> VA["SkipMarginalEvalDataset"]

    TR -->|"(x_t, t, x_population, t_population)"| enc
    VA -->|"(x_t_prev, t, x_t_skip, t_skip, skip_idx)"| enc

    enc["CellMNN.encode<br>MLP: (x, t) -> A"]
    enc -->|"A: (B, 1, D, D)"| dec["CellMNN.decode_trajectory<br>x(t') = expm(A(t'-t)) x"]

    dec -->|"x_traj: (B, T, D)"| loss["train: MMD + kinetic"]
    dec -->|"x_traj at t_skip"| emd["val/test: val_emd(skip_idx=...)"]
```

`skip_idx` names the held-out timepoint: training never sees it, and validation scores
the model by evolving the previous marginal forward to it, with exact Wasserstein-1.

Three extension points follow from the shape above:

- A dataset an existing reader handles is **one table in `datasets.toml`** — no code.
- A new kind of data is **one class in `sources.py`** plus one `SOURCE_TYPES` entry.

## CLI

Both training scripts live under `src/cell_mnn/cli/` and need `PYTHONPATH=src` until
packaging lands:

```bash
PYTHONPATH=src python -m cell_mnn.cli.train_mnn --skip_idx 1 --ds_name embryoid
PYTHONPATH=src python -m cell_mnn.cli.train_cfm --skip_idx 1 --ds_name embryoid --method i-cfm
```

Every flag has a hardcoded default; pass only the ones you want to change. To reuse a
set of overrides, save them as a flat TOML (`flag_name = value`) and point `--config`
at it — CLI flags still win over the file, and the file wins over the hardcoded
defaults:

```bash
PYTHONPATH=src python -m cell_mnn.cli.train_mnn --config configs/mnn/embryoid_baseline.toml
```

See `CLAUDE.md` for the full flag reference.
