from cell_mnn.model import CellMNN
from cell_mnn.utils import fix_seed, save_hyperparams_to_json
from cell_mnn.data.data_loading import get_datasets
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
import datetime
import argparse
import os
import json
import uuid

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # force determinism

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an MNN prediction model on embryoid data')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--skip_day_idx', type=int, default=1,
                        help='Index of day to skip for evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--time_limit', type=int,
                        default=240, help='Time limit in minutes')
    parser.add_argument('--check_val_every_n_epoch', type=int,
                        default=10, help='Check validation every n epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=200, help='Batch size')
    parser.add_argument('--train_on_all_days', action='store_true',
                        help='Train on all days in the dataset (training on validation data)')
    parser.add_argument('--lambda_kinetic', type=float, default=0.1,
                        help='Weight factor for kinetic energy regularizer.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Weight factor for kinetic energy regularizer.')
    parser.add_argument('--width', type=int, default=96, help='Width of MLP')
    parser.add_argument('--depth', type=int, default=4, help='Depth of MLP')
    parser.add_argument('--init_scale', type=float, default=0.01,
                        help='Weight initialization scale for final layer')
    parser.add_argument('--mmd_sigma', type=float, default=1.0,
                        help='Sigma for MMD loss')
    parser.add_argument('--ds_name', type=str, default='embryoid',)
    parser.add_argument('--val_ds_name', type=str, default=None,)
    parser.add_argument('--num_const_dims', type=int, default=0,
                        help='Number of constant dimensions')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    fix_seed(args.seed, use_det_algos=False)
    use_cuda = True
    device = torch.device('cuda' if use_cuda else 'cpu')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    uid = str(uuid.uuid4())

    train_dataset, val_dataset = get_datasets(
        ds_name=args.ds_name,
        val_ds_name=args.val_ds_name,
        skip_day_idx=args.skip_day_idx,
        batch_size=args.batch_size,
        device=device,
        val_prop=0.0,
        train_on_all_days=args.train_on_all_days,
        method="mnn",
    )
    latent_dim = train_dataset.latent_dim
    dataloader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    model_name = f"mnn_{args.ds_name}_{timestamp}_skip_day{train_dataset.skip_day}_{uid}"

    model = CellMNN(
        latent_dim=latent_dim,
        lr=args.lr,
        days_w_data=train_dataset.days,
        skip_day=train_dataset.skip_day,
        prev_day=train_dataset.prev_day,
        log_every_n_epochs=3,
        n_trajectories=3,
        train_on_skip_day=args.train_on_all_days,
        num_const_dims=args.num_const_dims,
        lambda_kinetic=args.lambda_kinetic,
        gamma=args.gamma,
        depth=args.depth,
        width=args.width,
        init_scale=args.init_scale,
        weight_decay=args.weight_decay,
        mmd_sigma=args.mmd_sigma,
    )
    wandb_logger = WandbLogger(
        project=f"scrna-seq-full-decomp_{args.ds_name}->{args.val_ds_name}",
        name=model_name,
        save_dir="logs",
        log_model=False,
    )

    # Log all arguments to wandb
    wandb_logger.log_hyperparams(vars(args))
    additional_config = {
        "latent_dim": latent_dim,
        "timestamp": timestamp,
        "use_cuda": use_cuda,
        "wandb_url": wandb_logger.experiment.url,
    }
    wandb_logger.log_hyperparams(additional_config)

    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=f'val_emd(skip_day={train_dataset.skip_day})',
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode='min',
        check_on_train_epoch_end=False,  # Only check on validation epochs
    )

    # Create model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_emd(skip_day={train_dataset.skip_day})',
        dirpath=f'weights/mnn/{model_name}/',
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

    trainer.fit(
        model,
        train_dataloaders=dataloader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint
    )

    hp_path = os.path.join(
        f'weights/mnn/{model_name}/', 'hyperparameters.json')
    save_hyperparams_to_json(wandb_logger, hp_path)

    # Testing on the skip day
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = DataLoader(val_dataset, batch_size=None)

    test_trainer = pl.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        logger=wandb_logger,
        enable_checkpointing=False,
    )
    test_results = test_trainer.test(model, test_loader)

    # Log test results 
    for metric_name, value in test_results[0].items():
        wandb_logger.experiment.summary[f"{metric_name}"] = value

    test_results_path = os.path.join(f'weights/mnn/{model_name}/', 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results[0], f)

    print(f"Test results: {test_results}")