# Library migration checklist

Turning this research repo into production code that downstream projects `pip install` and
`import`. Items are ordered so each phase leaves the repo working.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[-]` skipped (add a reason)

---

## Decision
The repo should allow the end-user to train models on their *own* data

---

## Phase 1 — Make it installable

Mechanical, no behavior changes. Goal: `pip install -e .` then `import cell_mnn` from any
working directory.

### 1.1 Package layout

- [x] Rename `lib/` → `src/cell_mnn/` (src-layout). `lib` as a top-level distribution name is a
      collision hazard for every importer.
- [x] Add the missing `src/cell_mnn/data/__init__.py`. `lib/data/` currently has none — it works
      as an implicit namespace package, but `setuptools.find_packages()` silently drops it from a
      wheel.
- [x] Convert internal absolute self-imports to relative:
  - [x] `lib/model.py:9` — `from lib.metrics import MMDLoss, compute_wasserstein`
  - [x] `lib/data/data_loading.py:11` — `from lib.data.data_preprocessing import get_data`
  - [x] `train_mnn.py:1-3`, `train_cfm.py` — top-level `from lib...` imports
- [x] Drop the unused `import scanpy as sc` at `lib/data/data_loading.py:3` (and the duplicate
      `import numpy` / `import torch` at lines 4-7).
- [x] Add `src/cell_mnn/py.typed` (the codebase is already annotated; make it count downstream).

# Questions to resolve
- too heavy on imports? -> environment.yaml contains way too much?
- hard coded time grid in CellMNN?
- What to do about train scripts?
- How/what to test?

### 1.2 `pyproject.toml`

- [ ] Create it. Build backend: hatchling or setuptools with explicit `packages = ["src/cell_mnn"]`.
- [ ] Runtime dependencies with **loose** bounds: `torch`, `numpy`, `scipy`, `anndata`, `einops`, `pot`.
- [ ] Optional extras:
  - [ ] `train` → `lightning`/`pytorch-lightning`, `wandb`, `matplotlib`
  - [ ] `baselines` → `torchcfm`, `torchdyn` (only `train_cfm.py` needs these)
  - [ ] `data` → `scanpy` (only the dataset registry needs it)
  - [ ] `dev` → `pytest`, `ruff`
- [ ] `[project.scripts]`: `cell-mnn-train = "cell_mnn.cli.train_mnn:main"`,
      `cell-mnn-train-cfm = "cell_mnn.cli.train_cfm:main"`.
- [ ] Keep `environment.yml` **only** as the paper-reproduction environment, and say so in a header
      comment. It is a frozen 200-line full-env dump (jupyter, kaggle, autopep8, mygene, scvelo,
      `numpy==1.26.4`) and is unusable as a dependency spec — it will fight every downstream env.

### 1.3 Entry points

- [x] Move `train_mnn.py` and `train_cfm.py` bodies into `src/cell_mnn/cli/`, each with a
      `main(argv=None)` and the `if __name__ == "__main__"` block reduced to `main()`.
- [x] Leave `train_mnn.py` / `train_cfm.py` at the repo root as thin shims so documented commands
      keep working.
- [ ] Decide where `data/inflate_data.py` and `data/recompute_pca.py` live — importable utilities
      under `cell_mnn.data.tools`, or scripts that stay out of the package.

### 1.4 Housekeeping

- [ ] Add `.gitignore` — there is none today, so `weights/`, `logs/`, downloaded `*.h5ad`/`*.npz`,
      `wandb/`, `__pycache__`, `*.egg-info` are one `git add -A` from being committed.
- [ ] Single source of truth for `__version__` (currently only `lib/__init__.py:13` = `0.1.0`);
      have `pyproject.toml` read it or vice versa.
- [ ] Add `CHANGELOG.md` and start it at the pre-migration state.
- [ ] Fix or remove `.vscode/settings.json` — it pins an interpreter path that is not `cell_mnn_env`.

**Phase 1 acceptance:** in a fresh env, `pip install -e .` succeeds; `cd /tmp && python -c "import
cell_mnn; print(cell_mnn.__version__)"` succeeds; `python train_mnn.py --debug` still runs.

---

## Phase 2 — Sever the couplings that make it unusable

This is where the real API decisions happen. Four things currently prevent the package from being
dropped into someone else's project.

### 2.1 Logging coupling (sharpest — breaks any non-W&B user)

`lib/model.py` imports `wandb` and `matplotlib` at module scope (lines 6-7), and
`validation_step` unconditionally calls `log_A_eigenvalues` → `self.logger.experiment.log(...)`
(`model.py:425`). With `logger=False` or any non-W&B logger this raises, so the model cannot be
used in a downstream training loop at all.

- [ ] Move `log_trajectories` (`model.py:324`) and `log_A_eigenvalues` (`model.py:407`) out of the
      model into a `pl.Callback` in `src/cell_mnn/callbacks.py`.
- [ ] Make `wandb` and `matplotlib` imports lazy / inside the callback.
- [ ] Verify: training runs to completion with `logger=False`.

### 2.2 Filesystem coupling

`get_data` (`lib/data/data_preprocessing.py:25-48`) hardcodes eight relative paths like
`data/ebdata/eb_velocity_v5.npz`, so any importer must `cd` to this repo's root.

- [ ] Extract a pure core: `prepare_marginals(adata_or_arrays, day_key, pca_dims) -> Marginals`
      with no filesystem access. Downstream projects bring their own AnnData and never touch the
      dataset registry.
- [ ] Reduce the `ds_name` registry to a thin convenience wrapper over a configurable root:
      `data_root` argument → `CELL_MNN_DATA` env var → default.
- [ ] Expose `pca_dims` (currently fixed at 5 and not on any CLI) and remove the hardcoded
      `latent_dim = 5` in `train_cfm.py`.
- [ ] Consider collapsing the per-dataset day-selection branches
      (`data_preprocessing.py:26-46` and `:71-80`) into one registry table — adding a dataset
      currently means editing two `if/elif` chains that must stay in sync.

### 2.3 Protocol coupling

`CellMNN.__init__` requires `days_w_data`, `skip_day`, `prev_day` — the leave-one-marginal-out
*evaluation protocol* is baked into the *model*. A downstream user who just wants "fit dynamics on
my timecourse, hand me `A`" has to invent a `skip_day`.

- [ ] Split into `CellMNNCore(nn.Module)` — `encode`, `forward`, `decode_trajectory`,
      `construct_A`, `compute_uncertainty`; no Lightning, no protocol args, no logging.
- [ ] `CellMNNModule(pl.LightningModule)` wraps the core with the skip-day loss, metrics, and
      optimizer. The paper reproduction uses this; downstream imports the core.
- [ ] Keep the `val_emd(skip_day={skip_day})` monitor string generated in **one** place — it is
      currently rebuilt independently in `model.py:318`, `train_mnn.py:124`, and `train_mnn.py:134`,
      and a mismatch breaks `EarlyStopping`/`ModelCheckpoint` silently.

### 2.4 Public API surface

`lib/__init__.py` exports four names, including the `FlowMatchingDataset` *base class*, but omits
`compute_wasserstein`, `MMDLoss`, `get_datasets`, and `predict_gene_interaction`.

- [ ] Curate `__init__.py` around the intended entry points; keep `__all__` accurate.
- [ ] Add `CellMNNModule.from_checkpoint(path, map_location)`. Checkpoint loading is the main
      downstream entry point and is currently four manual steps (`train_mnn.py:166-168`);
      `save_hyperparameters()` is already called at `model.py:71`, so `load_from_checkpoint` should
      work — verify and document it.
- [ ] Add a `predict_marginal(x, t_from, t_to)` convenience wrapper over the fixed-grid trajectory
      indexing, so callers don't reimplement `int((day - min_t) / dt)`.
- [ ] Add an ensemble loader for `predict_gene_interaction` (`lib/interpretability.py:5`) — it takes
      a bare `list[torch.nn.Module]` with no supported way to build one.
- [ ] Write down a deprecation policy (what's public, what can change without a major bump).

**Phase 2 acceptance:** a scratch project outside this repo can, from an arbitrary cwd, load a
checkpoint and get an `A` matrix for its own AnnData without importing `wandb` or setting a
`skip_day`.

---

## Phase 3 — Make it production

### 3.1 Tests (largest confidence gap — there are none today)

- [ ] Set up `tests/` + pytest config. The closest current thing to a test is the `__main__` block
      in `lib/data/data_preprocessing.py` and `lib/data/data_loading.py`.
- [ ] `forward` shape contract: `(B,1,D)`, `(B,1,1)`, `(B,T,1)` in → trajectory `(B,T,D)` out;
      assert the input-validation asserts fire on bad shapes.
- [ ] `construct_A(P_inv, Λ, P)` against a hand-computed diagonal case.
- [ ] `Δt = 0` round-trip: the trajectory evaluated at the initial time returns `x_t`.
- [ ] `compute_wasserstein(X, X) == 0`.
- [ ] Two-step CPU fit smoke test. **Blocked by** `use_cuda = True` hardcoded at `train_mnn.py:65` —
      the MNN path has no CPU fallback (`train_cfm.py` does).
- [ ] Dataset contract test: each `IterableDataset` yields the documented tuple arity and shapes.

### 3.2 Correctness

- [ ] Fix `OTFlowMatchingDataset.__iter__` using `self.rng`, which is never assigned
      (`lib/data/data_loading.py:233`) — `--method ot-cfm` raises `AttributeError`. Fix or remove
      before tagging a version.
- [ ] Validate integer-day / `dt` coupling at construction. `int((self.skip_day - self.min_t) /
      self.dt)` (`model.py:283`) and the `torch.isclose` train mask (`model.py:230`) assume integer
      day labels with `dt = 0.1`; non-integer days silently index the wrong timepoint. Raise instead.
- [ ] Reconcile the documented `skip_day_idx` range with the asserts: `MnnDataset` (`:324`) and
      `EmbryoidFlowMatchingTestDataset` (`:253`) require `0 < skip_day_idx < n_days - 1`, so the
      final day cannot be held out — narrower than the README's `1, ..., t_max - 1`. Fix the docs or
      the asserts.
- [ ] `train_mnn.py:15` sets `CUBLAS_WORKSPACE_CONFIG` "to force determinism" but line 64 calls
      `fix_seed(seed, use_det_algos=False)`. Pick one and make the CLI flag explicit.
- [ ] Surface the buried magic numbers as constructor params with current values as defaults:
      `dt = 0.1` and `ode_order = 1` (`model.py:94-95`), the `0.005` prior noise variance
      (`model.py:207`), the `1e-4` determinant epsilon (`model.py:264`), the `sigma = 0.1` flow
      matcher noise (`data_loading.py:134,140,165`), and the `10_000` `too_big` threshold
      (`data_loading.py:492`).

### 3.3 Device / dataloader model

- [ ] Decide per the open decision above: either document the device-owning `IterableDataset` +
      `batch_size=None` contract loudly as a single-process-only design, or move `.to(device)` out
      of `__iter__` and let Lightning handle transfer (required for DDP / `num_workers > 0`).
- [ ] Give the MNN path a CPU fallback (see 3.1).

### 3.4 CI & docs

- [ ] CI: lint (ruff) + the CPU test subset on push. No GPU needed for any of 3.1 except the
      currently-blocked fit test.
- [ ] Rewrite `README.md` split into *library usage* (install, import, checkpoint → `A`) and
      *paper reproduction* (conda env, `download_data.sh`, exact commands).
- [ ] Fix the README's documented-but-deleted scripts (removed in `c0a43ef`):
      `validate_on_trrust.py`, `pure_OT_interpolation.py`, `pre_trained_models/`. The surviving
      halves are `lib/interpretability.py` and `data/tf_targets_trrust/*.tsv`. Recover the driver
      with `git show c0a43ef^:validate_on_trrust.py` if it should come back.
- [ ] Document that W&B is currently mandatory in both training scripts, and what changes after 2.1.
- [ ] Tag `v0.1.0` once Phase 1 + 2 land; note the `lib` → `cell_mnn` rename as breaking.
