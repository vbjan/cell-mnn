import torch
import random
import os
import pytorch_lightning as pl
import numpy as np
import json
import argparse


def fix_seed(
    seed: int,
    use_det_algos: bool = False
):
    pl.seed_everything(seed, workers=True)           # Lightning helper
    torch.use_deterministic_algorithms(use_det_algos)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = use_det_algos
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Save hyperparameters as a JSON file
def save_hyperparams_to_json(
        wandb_logger,
        hyperparams_path: str
):
    with open(hyperparams_path, 'w') as f:
        # Convert wandb_logger.experiment.config to a dict first
        config_dict = dict(wandb_logger.experiment.config)
        json.dump(config_dict, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")


def str2bool(v: str) -> bool:
    if isinstance(v, bool):          # lets you pass a real bool when you call the script manually
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
