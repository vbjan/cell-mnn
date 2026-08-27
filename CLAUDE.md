# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Reference implementation of **Cell-MNN** (Cell Mechanistic Neural Networks), ICLR 2026 — an encoder/decoder whose latent representation is a *locally linearized ODE* (`ẋ = A x`) governing single-cell trajectories in PCA'd gene-expression space. Unlike Neural ODEs / flow matching (which output a velocity), the MLP predicts the linear operator itself, which makes `A` inspectable as gene–gene interactions.

## Setup & commands

```bash
conda env create -f environment.yml && conda activate cell_mnn_env

# Datasets (Embryoid / Cite / Multi). Must be run from the repo root.
chmod +x data/download_data.sh && ./data/download_data.sh
```

```bash
# Train + auto-evaluate Cell-MNN, holding out marginal `skip_day_idx`
python train_mnn.py --skip_day_idx 1 --ds_name embryoid

# One eigenvalue forced to zero
python train_mnn.py --skip_day_idx 1 --ds_name embryoid --num_const_dims 1

# Amortized: train on cite+multi jointly, validate on one of them
python train_mnn.py --skip_day_idx 1 --ds_name mix --val_ds_name cite --width 128

# Baselines (I-CFM / OT-CFM), same CLI shape
python train_cfm.py --skip_day_idx 1 --ds_name embryoid --method i-cfm

# Fast smoke test of either script (Lightning fast_dev_run: 1 train + 1 val batch)
python train_mnn.py --debug

# Data utilities
python data/inflate_data.py data/ebdata/eb_velocity_v5.npz 50000 --noise_std 0.1
python data/recompute_pca.py data/ebdata/ebdata_v3.h5ad -n 50
```

There is **no test suite** (`pytest` is in the env but no test files exist). The closest thing to unit checks are the `__main__` blocks in `lib/data/data_preprocessing.py` and `lib/data/data_loading.py`, runnable as `python -m lib.data.data_preprocessing`.

Operational notes:
- **All dataset paths are relative** (`data/ebdata/...`) — every script must be launched from the repo root.
- **W&B is not optional.** Both training scripts unconditionally build a `WandbLogger` (`save_dir="logs"`) and log final test metrics to the run summary. Use `wandb login`, or `WANDB_MODE=offline` / `WANDB_MODE=disabled` when running without an account.
- Artifacts land in `weights/mnn/<model_name>/` (`best-model.ckpt`, `hyperparameters.json`, `test_results.json`); `train_cfm.py` uses `weights/checkpoints/<model_name>/`. Neither directory is gitignored (there is no `.gitignore`) — don't commit them.
- `train_mnn.py` hardcodes `use_cuda = True`, so it **requires a GPU**; `train_cfm.py` falls back to CPU.
- Defaults in `parse_args` are the paper's hyperparameters — changing one silently changes the reproduction.

## Architecture

The whole pipeline is built around one evaluation protocol: **leave-one-marginal-out interpolation**. Day `skip_day_idx` is dropped from training; validation/test predicts that day's distribution by evolving the *previous* day's cells forward, scored with exact Wasserstein-1. The metric name is constructed dynamically (`val_emd(skip_day={skip_day})`) and both `EarlyStopping` and `ModelCheckpoint` monitor that exact string — renaming the log key breaks training silently.

**1. Preprocessing — `lib/data/data_preprocessing.py::get_data`**
Dispatches on `ds_name` to one of the h5ad/npz files, takes the first `pca_dims=5` *precomputed* PCs (`obsm["X_pca"]`, or `"pcs"` for the npz), z-scores them globally, then returns `X_train` as a **list of arrays, one per day** plus `t_train` day labels. Day-selection logic is per-dataset (`obs["day"]` vs `obs["sample_labels"]` vs `.cat.codes`), so adding a dataset means touching two branches. `pca_dims` is not exposed on any CLI, and `train_cfm.py` hardcodes `latent_dim = 5` — the latent dimension is effectively fixed at 5 repo-wide.

Valid `ds_name`: `embryoid`, `embryoid_less_preprocessed`, `embryoid_inflated`, `cite`, `cite_inflated`, `multi`, `multi_inflated`, and `mix` (handled one level up in `get_datasets`).

**2. Datasets — `lib/data/data_loading.py`**
Every dataset is an `IterableDataset` that yields *already-batched* tensors on the target device, so all `DataLoader`s are constructed with `batch_size=None`. `__len__` is a heuristic (total cells ÷ batch size) that defines what an "epoch" means.

- `MnnDataset` → `(x_t, t, x_population, t_population)`: per-cell initial conditions spread across days, plus a resampled full marginal for *every* day, which is what the MMD loss compares against.
- `SkipMarginalEvalDataset` → `(t, x_prev_day, x_skip_day, x_next_avail_day)`. Used for val/test by **both** methods. Loads whole marginals in a single batch unless the dataset exceeds 10k cells per day (`too_big` in `construct_train_val_datasets`), because exact OT over everything gets expensive.
- `IndependentFlowMatchingDataset` / `BatchOTFlowMatchingDataset` / `OTFlowMatchingDataset` → `(T, xT, uT)` for the CFM baselines, with `t∈[0,1]` rescaled to real day intervals and displacement converted to per-day velocity (`u_t / day_diff`). This is what lets the baselines handle non-uniform day gaps and the skipped day.
- `MixedDataset` (`ds_name="mix"`) round-robins between the cite and multi iterators for amortized training; validation still comes from `val_ds_name` only.
- `get_datasets` is the single entry point used by both training scripts; `method` (`"mnn"`, `"i-cfm"`, `"batch-ot-cfm"`, `"ot-cfm"`) selects the train-dataset class.

**3. Model — `lib/model.py::CellMNN` (`pl.LightningModule`)**
`encode(x_t, t)`: one MLP maps the `D+1` input to `latent_dim² + dynamic_dims` outputs, split into

- `eigenvals` — `dynamic_dims` values, right-padded with `num_const_dims` **zeros** (the "eigenvalue set to zero" ablation), and
- `P` — reshaped to `D×D` **plus the identity**, so the mixing matrix starts near-identity (reinforced by `init_scale` shrinking the MLP's last layer).

`forward` then solves the diagonal system in closed form: `z = P x`, `z(t') = exp(Λ·Δt) z`, map back with `torch.linalg.inv(P)`, and return both the trajectory and its derivative. Trajectories are always decoded on a **fixed grid** `arange(min_day, max_day + dt, dt)` with `dt = 0.1`; specific days are indexed by `int((day - min_t)/dt)`. Integer-valued day labels and `dt` are therefore coupled assumptions — non-integer day labels would break the index arithmetic and the `torch.isclose` train masking.

Training loss (`training_step`) is three terms:
1. MMD between predicted and observed marginals on every supervised day, weighted by `gamma ** i` where `i` counts how many days ahead the prediction reaches (nearer marginals matter more);
2. `lambda_kinetic · mean(ẋ²)` — kinetic-energy regularizer;
3. `mean(1 / (|det P| + 1e-4))` — keeps `P` invertible, which the closed-form solution depends on.

`ode_order = 1` is fixed with the comment that higher order overfits. `construct_A(P_inv, eigenvals, P) = P_inv Λ P` is the interpretable local operator.

**4. Metrics — `lib/metrics/`**
`MMDLoss` is the *training* objective: biased MMD² with a Laplace kernel over L1 distances normalized by `D`. `compute_wasserstein` is the *reporting* metric: exact `ot.emd2` from POT, with `num_iter_max` raised from 200k to 1M in `test_step` for the final number. The two are deliberately different — don't swap one for the other.

**5. Interpretability — `lib/interpretability.py`**
`predict_gene_interaction` averages `A` across an **ensemble** of checkpoints, lifts it from PCA space to gene space with the loadings `W = adata.varm['PCs']` (`einsum('vp,bpq,wq->bvw')`), then scales rows by the cell's gene-space expression so the result is a per-cell, signed interaction matrix. Sign is what the TRRUST validation checks (Activation ⇒ positive, Repression ⇒ negative).

## Known discrepancies with the README

The README documents scripts that were deleted in `c0a43ef` ("deleted unnecessary files for library"):

- `validate_on_trrust.py`, `pure_OT_interpolation.py`, and the `pre_trained_models/` checkpoints no longer exist. The surviving halves of that experiment are `lib/interpretability.py` and the ground-truth TSVs in `data/tf_targets_trrust/` (FOS, HMGA1, JUN, POU5F1, SOX2, YBX1). Recover the driver with `git show c0a43ef^:validate_on_trrust.py` before re-implementing it.
- `--method ot-cfm` is currently broken: `OTFlowMatchingDataset.__iter__` uses `self.rng`, which is never assigned (`lib/data/data_loading.py:233`). `i-cfm` and `batch-ot-cfm` work.
- `MnnDataset` and `SkipMarginalEvalDataset` assert `0 < skip_day_idx < n_days - 1`, so the **final** day cannot be held out — narrower than the README's `1, 2, ..., t_max - 1`.
- `train_mnn.py` sets `CUBLAS_WORKSPACE_CONFIG` "to force determinism", but calls `fix_seed(seed, use_det_algos=False)`; deterministic algorithms are off by default.
- `.vscode/settings.json` pins an interpreter path that is not `cell_mnn_env`.
