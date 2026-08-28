import numpy as np
import datetime
import os
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader
from cell_mnn.data.data_loading import build_datasets
from cell_mnn.data.sources import load_marginals
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from cell_mnn.metrics import compute_wasserstein, MMDLoss
from cell_mnn.utils import save_hyperparams_to_json, fix_seed


# Parse command-line arguments
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a Flow Matching model on embryoid data')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--skip_idx', type=int, default=1,
                        help='Index of timepoint to skip for evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--time_limit', type=int,
                        default=240, help='Time limit in minutes')
    parser.add_argument('--check_val_every_n_epoch', type=int,
                        default=10, help='Check validation every n epochs')
    parser.add_argument('--method', type=str, default='i-cfm',
                        choices=['ot-cfm', 'batch-ot-cfm', 'i-cfm'], help='Method to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--ds_name', type=str, default='embryoid',
                        help='Dataset name; a table in the config TOML')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Path to the dataset config TOML (default: ./datasets.toml)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for training')
    return parser.parse_args(argv)


class CFMVelocityMLP(torch.nn.Module):
    """Time-conditioned velocity field for the CFM baselines.

    Matches torchcfm.models.MLP(time_varying=True) exactly (3 hidden
    layers of width `w`, SELU activations) so the baseline stays
    unchanged; inlined here to avoid depending on torchcfm just for a
    generic MLP.
    """

    def __init__(
            self, 
            dim: int, 
            out_dim: Optional[int] = None, 
            w: int = 64,
            time_varying: bool = False
        ) -> None:
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowMatchingModel(pl.LightningModule):
    def __init__(
            self,
            dim: int,
            skip_idx: int,
            t_grid: Sequence[float],
            lr: float,
            w: int = 64
        ) -> None:
        super().__init__()
        self.ot_cfm_model = CFMVelocityMLP(dim=dim, time_varying=True, w=64)
        self.node = NeuralODE(torch_wrapper(
            self.ot_cfm_model), solver="rk4", sensitivity="adjoint")
        # index of timepoint to skip for evaluation [0, n_times - 1]
        self.skip_idx = skip_idx
        self.t_grid = t_grid
        self.t_skip = t_grid[skip_idx]
        self.t_prev = t_grid[skip_idx - 1]

        self.lr = lr
        self.dt = 0.01
        self.mmd_loss = MMDLoss(sigma=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        vt = self.ot_cfm_model(obs)  # obs contains both x and t here
        return vt

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        xt, t, ut = batch
        vt = self.forward(torch.cat([xt, t.unsqueeze(-1)], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        self.log("train_loss", loss)
        return loss

    def in_distribution_validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        xt, t, ut = batch
        vt = self.forward(torch.cat([xt, t.unsqueeze(-1)], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        self.log("val_loss", loss)
        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            num_iter_max: int = 200_000
        ) -> float:
        x_t_prev, t, x_t_skip, _ = batch
        t_span = torch.arange(self.t_prev, self.t_skip + self.dt, self.dt)
        traj = self.node.trajectory(
            x_t_prev,
            t_span=t_span,
        )
        # get predicted distribution at the last timepoint
        pred_dist = traj[-1, :, :]

        mmd = self.mmd_loss(pred_dist, x_t_skip)
        self.log(f"val_mmd(t_skip={self.t_skip})", mmd.cpu().item())

        emd = compute_wasserstein(
            pred_dist.cpu().numpy(),
            x_t_skip.cpu().numpy(),
            num_iter_max=num_iter_max)
        self.log(f"val_emd(t_skip={self.t_skip})", emd)

        return emd

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> float:
        return self.validation_step(batch, batch_idx, num_iter_max=1_000_000)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.ot_cfm_model.parameters(),
                                      self.lr,
                                      weight_decay=1e-5)
        return optimizer


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    fix_seed(args.seed, use_det_algos=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sigma = 0.1
    lr = 1e-3
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Instantiate the dataset
    marginals = load_marginals(ds_name=args.ds_name, config_path=args.datasets)
    latent_dim = marginals.n_features
    train_dataset, val_dataset = build_datasets(
        marginals,
        skip_idx=args.skip_idx,
        train_on_all_times=False,
        method=args.method,
        device=device,
        batch_size=args.batch_size,
    )

    # Use batch_size=None since the dataset is already returning a full batch.
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers)

    cfm_model = FlowMatchingModel(
        dim=latent_dim,
        skip_idx=args.skip_idx,
        lr=lr,
        t_grid=marginals.t_grid,
    ).to(device)
    model_name = f"{args.method}_5-dim_pca_skip_idx{args.skip_idx}_{timestamp}"

    wandb_logger = WandbLogger(
        project=f"OT-CFM-{args.ds_name}",
        name=model_name,
        save_dir="logs",
        log_model=False,
    )

    wandb_logger.experiment.config.update(vars(args))

    # Log all arguments to wandb
    additional_config = {
        "latent_dim": latent_dim,
        "sigma": sigma,
        "timestamp": timestamp,
        "use_cuda": use_cuda,
        "lr": lr,
        "wandb_url": wandb_logger.experiment.url,
    }
    wandb_logger.experiment.config.update(additional_config)

    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=f'val_emd(t_skip={cfm_model.t_skip})',
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode='min',
        check_on_train_epoch_end=False,  # Only check on validation epochs
    )

    # Create model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_emd(t_skip={cfm_model.t_skip})',
        dirpath=f'weights/checkpoints/{model_name}/',
        filename=f'best-model',
        save_top_k=1,
        mode='min',
        save_last=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        max_epochs=args.epochs,
        enable_checkpointing=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        max_time={"minutes": args.time_limit},
        fast_dev_run=args.debug,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(cfm_model, train_loader, val_loader)

    if args.debug:
        best_model = cfm_model
        print("Debug mode: Using the model from training (no checkpoint loading)")
    else:
        best_model = FlowMatchingModel.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                            dim=latent_dim,
                                                            skip_idx=args.skip_idx,
                                                            lr=1e-4,
                                                            t_grid=marginals.t_grid)

    # Evaluate the best model on the test dataset
    print("\nEvaluating best model on test dataset...")
    test_trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        logger=wandb_logger,
        enable_checkpointing=False,
    )
    test_results = test_trainer.test(best_model, val_loader)
    print(f"Test EMD: {test_results[0][f'val_emd(t_skip={cfm_model.t_skip})']:.4f}")

    # Log the final test EMD to wandb
    wandb_logger.experiment.summary["final_val_emd"] = test_results[0][f'val_emd(t_skip={cfm_model.t_skip})']

    # Save hyperparameters as a JSON file
    hyperparams_path = os.path.join(
        f'weights/checkpoints/{model_name}/', 'hyperparameters.json')
    save_hyperparams_to_json(wandb_logger, hyperparams_path)


if __name__ == "__main__":
    main()
