# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Reference implementation of **Cell-MNN** (Cell Mechanistic Neural Networks), ICLR 2026 — an encoder/decoder whose latent representation is a *locally linearized ODE* (`ẋ = A x`) governing single-cell trajectories in PCA'd gene-expression space. Unlike Neural ODEs / flow matching (which output a velocity), the MLP predicts the linear operator itself, which makes `A` inspectable as gene–gene interactions.

## Time notation

The data layer and both CLIs speak in **timepoints, not days**: `skip_idx`, `t_grid`, `t_skip`, `t_prev`, `n_times`, `t_indcs`, `cells_per_t`, `delta_t`. Nothing in `src/` says "day" any more — if you reintroduce it, you are fighting the convention.

The one exception is *data* column names, which must stay as they are on disk: `obs["day"]` and `obs["sample_labels"]` in the h5ad files, and the `'days'` / `'sample_labels'` keys in the npz (`data/inflate_data.py`).

## Setup & commands

```bash
conda env create -f environment.yml && conda activate cell_mnn_env

# Datasets (Embryoid / Cite / Multi). Must be run from the repo root.
chmod +x data/download_data.sh && ./data/download_data.sh
```

There is **no `pyproject.toml` yet** (Phase 1 of `LIBRARY_MIGRATION.md` is unfinished), so `import cell_mnn` fails from the repo root. Until packaging lands, prefix commands with `PYTHONPATH=src`.

```bash
# Train + auto-evaluate Cell-MNN, holding out marginal `skip_idx`
PYTHONPATH=src python -m cell_mnn.cli.train_mnn --skip_idx 1 --ds_name embryoid

# Extra random timepoints for the kinetic regularizer
PYTHONPATH=src python -m cell_mnn.cli.train_mnn --skip_idx 1 --ds_name embryoid --kinetic_grid_multiplier 4

# Baselines (I-CFM / OT-CFM), same CLI shape
PYTHONPATH=src python -m cell_mnn.cli.train_cfm --skip_idx 1 --ds_name embryoid --method i-cfm

# Data utilities (plain scripts, no PYTHONPATH needed)
python data/inflate_data.py data/ebdata/eb_velocity_v5.npz 50000 --noise_std 0.1
python data/recompute_pca.py data/ebdata/ebdata_v3.h5ad -n 50
```

`train_mnn.py` flags: `--epochs --skip_idx --debug --patience --time_limit --check_val_every_n_epoch --seed --lr --weight_decay --batch_size --train_on_all_times --lambda_kinetic --gamma --kinetic_grid_multiplier --width --depth --init_scale --mmd_sigma --ds_name --resume_from_checkpoint`.
`train_cfm.py` flags: `--epochs --skip_idx --debug --patience --time_limit --check_val_every_n_epoch --method --seed --ds_name --batch_size`.

There is **no test suite** (`pytest` is in the env but no test files exist). The closest thing to unit checks are the `__main__` blocks in `src/cell_mnn/data/marginals.py`, `data_preprocessing.py`, and `data_loading.py`, runnable as `PYTHONPATH=src python -m cell_mnn.data.marginals`. Only the `marginals` one runs without the downloaded datasets.

Operational notes:
- **All dataset paths are relative** (`data/ebdata/...`) — every script must be launched from the repo root.
- **W&B is not optional.** Both training scripts unconditionally build a `WandbLogger` (`save_dir="logs"`) and log final test metrics to the run summary. Use `wandb login` or `WANDB_MODE=offline`. `WANDB_MODE=disabled` is **not** a safe substitute for `train_cfm.py`: its `experiment.config.update(...)` (`train_cfm.py:183`) raises `AttributeError` against the disabled-run stub.
- Artifacts land in `weights/mnn/<model_name>/` (`best-model.ckpt`, `hyperparameters.json`, `test_results.json`); `train_cfm.py` uses `weights/checkpoints/<model_name>/`. `.gitignore` covers only `.vscode` and `__pycache__`, so neither `weights/` nor `logs/` is ignored — don't commit them.
- `train_mnn.py` hardcodes `use_cuda = True`, so it **requires a GPU**; `train_cfm.py` falls back to CPU.
- `train_cfm.py`'s post-fit `test_trainer` omits `devices=1` (which `train_mnn.py` sets), so on a multi-GPU host Lightning spawns DDP for the test pass and re-executes `main()` in the child process. Pin `CUDA_VISIBLE_DEVICES=0` for single-run reproductions.
- Defaults in `parse_args` are the paper's hyperparameters — changing one silently changes the reproduction.
- To run against synthetic data without the downloads, build a `TimeSeriesMarginals` and hand it to `build_datasets` — no monkeypatching. That covers every code path except preprocessing.

## Architecture

The whole pipeline is built around one evaluation protocol: **leave-one-marginal-out interpolation**. Timepoint `skip_idx` is dropped from training; validation/test predicts that marginal by evolving the *previous* timepoint's cells forward, scored with exact Wasserstein-1. The metric key is constructed dynamically as `val_emd(t_skip={t_skip})` and both `EarlyStopping` and `ModelCheckpoint` monitor that exact string — renaming the log key breaks training silently. Read the value off the object that *logs* it (`train_mnn.py` uses `train_dataset.t_skip`, `train_cfm.py` uses `cfm_model.t_skip`) rather than re-deriving it by indexing a grid.

**0. The data contract — `src/cell_mnn/data/marginals.py::TimeSeriesMarginals`**
The single interface between data sources and datasets: `X` (one array of shape `(n_cells_i, n_features)` per timepoint) + `t_grid` (strictly ascending real times) + `name`. `__post_init__` validates arity, 2-D-ness, consistent `n_features`, non-empty marginals, finiteness, and strict ascent, and coerces `t_grid` to python `float` — so numpy scalars never reach the `val_emd(t_skip=...)` metric key. Exposes `n_times` / `n_features` / `cells_per_t` / `n_cells`, `m[i] -> (X_i, t_i)`, and `drop(i)`. Anything that can produce populations-per-timepoint (the registry, a synthetic generator, a bare pair of lists) produces one of these, and every dataset consumes one.

**1. Preprocessing — `src/cell_mnn/data/data_preprocessing.py::load_marginals`**
Dispatches on `ds_name` to one of the h5ad/npz files, takes the first `pca_dims=5` *precomputed* PCs (`obsm["X_pca"]`, or `"pcs"` for the npz), z-scores them globally, and returns a `TimeSeriesMarginals`. Timepoint selection is per-dataset (`obs["day"]` vs `obs["sample_labels"]` vs `.cat.codes`), so adding a dataset means touching two branches. `pca_dims` is not exposed on any CLI, so the feature dimension is effectively 5 repo-wide — but both CLIs now read it from `marginals.n_features` rather than hardcoding it.

Valid `ds_name`: `embryoid`, `embryoid_less_preprocessed`, `embryoid_inflated`, `cite`, `cite_inflated`, `multi`, `multi_inflated`.

**2. Datasets — `src/cell_mnn/data/data_loading.py`**
Every dataset is an `IterableDataset` that yields *already-batched* tensors on the target device, so all `DataLoader`s are constructed with `batch_size=None`. `__len__` is a heuristic (total cells ÷ batch size) that defines what an "epoch" means. `TimeFilteredDataset` is the shared base: it takes a `TimeSeriesMarginals` and exposes `marginals` (as handed in) alongside `train_marginals` (= `marginals.drop(skip_idx)`, or all of them when `train_on_skip=True`).

- `MnnDataset` → `(x_t, t, x_population, t_population)`: per-cell initial conditions spread across timepoints, plus a resampled full marginal for *every* timepoint, which is what the MMD loss compares against.
- `SkipMarginalEvalDataset` → `(x_t_prev, t, x_t_skip, t_skip)`. Used for val/test by **both** methods. Loads whole marginals in a single batch unless the dataset exceeds 10k cells at the first timepoint (`too_big` in `build_datasets`), because exact OT over everything gets expensive.
- `IndependentFlowMatchingDataset` / `BatchOTFlowMatchingDataset` / `OTFlowMatchingDataset` → `(xT, T, uT)` for the CFM baselines, with `t∈[0,1]` rescaled onto the real interval `[t_i, t_j]` and displacement converted to per-unit-time velocity (`u_t / delta_t`). This is what lets the baselines handle non-uniform time gaps and the skipped timepoint.
- `build_datasets(marginals, ...)` is the single entry point used by both training scripts, building the train/val pair from a `TimeSeriesMarginals`; `method` (`"mnn"`, `"i-cfm"`, `"batch-ot-cfm"`, `"ot-cfm"`) selects the train-dataset class via the `TRAIN_DATASET_BY_METHOD` table. All four train classes take the same constructor keywords; the three CFM baselines differ only in the `flow_matcher_cls` class attribute, so adding a baseline means one subclass plus one table entry. Cell sampling goes through the module-level `to_tensor` / `sample_cells` helpers.

**3. Model — `src/cell_mnn/model.py::CellMNN` (`pl.LightningModule`)**
`encode(x_t, t)`: one MLP (LeakyReLU, `depth` hidden layers of `width`, kaiming-init, last layer scaled by `init_scale`) maps the `D+1` input to `latent_dim²` outputs, reshaped **directly** into the local operator `A` of shape `(B, 1, D, D)`. There is no eigen-parameterization — no `P`, no eigenvalue vector, no `num_const_dims` ablation.

`decode_trajectory(A, x_t, t, decode_ts)` solves the linear system in closed form: `x(t') = expm(A·(t'−t)) x`, via `torch.linalg.matrix_exp`, and returns `x` stacked with `ẋ = A x` along a trailing dimension. `forward` splits that into `(x_traj, x_dot_traj, A)`.

Decoding happens **at whatever timepoints are handed in** (`t_population` from the batch), not on a fixed `dt` grid, so there is no index arithmetic and no integer-timepoint assumption — non-uniform and non-integer `t_grid` values work.

Training loss (`training_step`) is two terms:
1. MMD between predicted and observed marginals on every supervised timepoint, weighted by `gamma ** i` where `i` counts how many timepoints ahead the prediction reaches (nearer marginals matter more). The `i`-th mask is built by cumsum over `t_population > t`, so each row only supervises strictly-future marginals.
2. `lambda_kinetic · mean(ẋ²)` — kinetic-energy regularizer. With `kinetic_grid_multiplier > 1`, `_decode_dense_kinetic_grid` adds `T·(multiplier−1)` uniformly random timepoints in `[t_min, t_max]` to this term only.

`validation_step` decodes the previous marginal straight to `t_skip`, logs `val_mmd(...)` / `val_emd(...)`, and pushes histograms of `eigvals(A)` (real and imaginary parts) to W&B via `log_A_eigenvalues`. `test_step` is `validation_step` with `num_iter_max=1_000_000`.

**4. Metrics — `src/cell_mnn/metrics/`**
`MMDLoss` is the *training* objective: biased MMD² with a Laplace kernel over L1 distances normalized by `D`. `compute_wasserstein` is the *reporting* metric: exact `ot.emd2` from POT, with `num_iter_max` raised from 200k to 1M in `test_step` for the final number. The two are deliberately different — don't swap one for the other. `emd.py` also holds `compute_ot_coupling` (a barycentric-sampling helper), which is not re-exported from `metrics/__init__.py` and currently unused.

## Known gaps & gotchas

There is no README in the repo any more; several things it used to document have been deleted.

- **`--debug` does not run clean to completion in either script.** `fast_dev_run` disables `ModelCheckpoint`, so the `weights/<...>/` directory is never created and the post-fit `save_hyperparams_to_json` dies with `FileNotFoundError`. Training and validation do execute first, so it still smoke-tests the model path — just expect the traceback at the end. In `train_mnn.py` the write is the *first* thing to fail; the `torch.load(checkpoint_callback.best_model_path)` on the next line would fail too, on an empty path.
- `interpretability.py` was removed in `076b0fa` ("removed unused code"); `validate_on_trrust.py`, `pure_OT_interpolation.py`, and `pre_trained_models/` went in `c0a43ef`. The surviving half of that experiment is the ground-truth TSVs in `data/tf_targets_trrust/` (FOS, HMGA1, JUN, POU5F1, SOX2, YBX1). Recover the drivers with `git show 076b0fa^:src/cell_mnn/interpretability.py` and `git show c0a43ef^:validate_on_trrust.py`. `predict_gene_interaction` only calls `model.encode(x_sample, t)` and treats the result as `A`, which is exactly what the current model returns — so it is still API-compatible and can be un-deleted as-is; only its `day` argument name is off-convention (`t`).
- `src/cell_mnn/__init__.py` exports `CellMNN`, `TimeSeriesMarginals`, `load_marginals`, and `build_datasets`. The dataset classes themselves are import-by-path only.
- The `skip_idx` asserts are `1 <= skip_idx <= n_times - 1` in both `TimeFilteredDataset` and `SkipMarginalEvalDataset`, so the **final** timepoint *can* be held out.
- `train_mnn.py` sets `CUBLAS_WORKSPACE_CONFIG` "to force determinism", but calls `fix_seed(seed, use_det_algos=False)`; deterministic algorithms are off by default.
- `.vscode/settings.json` pins an interpreter path that is not `cell_mnn_env`.
